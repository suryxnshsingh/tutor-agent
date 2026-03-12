from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

from app.agent.attempt import run_attempt
from app.agent.events import (
    AgentEvent,
    CardsEvent,
    FollowUpEvent,
    ResponseDelta,
    ResponseEndEvent,
    ResponseStartEvent,
    StatusEvent,
)
from app.agent.prompt import build_teacher_prompt, filter_thinking_stream
from app.agent.subagents.base import SubAgentDispatch, SubAgentResult, TeacherDecision
from app.agent.subagents.content import run_content_agent
from app.agent.subagents.doubt import run_doubt_agent, stream_doubt_agent
from app.agent.subagents.guidance import run_guidance_agent, stream_guidance_agent
from app.config import settings
from app.gateway.models import UniversalMessage
from app.llm.openai_provider import OpenAIProvider
from app.messages.models import InternalMessage
from app.session.manager import SessionManager
from app.tools.registry import get_definitions_by_names

logger = logging.getLogger(__name__)

_JSON_FORMAT = {"type": "json_object"}

# Subagent timeout — if one subagent hangs, don't block the whole response
_SUBAGENT_TIMEOUT_S = 30.0


@dataclass
class AgentResponse:
    """Final structured response from the agent."""

    text: str
    cards: list[dict] = field(default_factory=list)
    buttons: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    thinking: str | None = None
    follow_up: str | None = None


# ── Singletons (set in main.py at startup) ──

_session_manager: SessionManager | None = None
_provider: OpenAIProvider | None = None


def init_session_manager(manager: SessionManager) -> None:
    global _session_manager
    _session_manager = manager


def init_provider(provider: OpenAIProvider) -> None:
    global _provider
    _provider = provider


def get_session_manager() -> SessionManager:
    if _session_manager is None:
        raise RuntimeError("SessionManager not initialized.")
    return _session_manager


def get_provider() -> OpenAIProvider:
    if _provider is None:
        raise RuntimeError("OpenAIProvider not initialized.")
    return _provider


# ── Teacher routing ──


async def _run_teacher(
    system_prompt: str,
    messages: list[InternalMessage],
    provider: OpenAIProvider,
    trace_id: str,
) -> TeacherDecision:
    """Run the teacher LLM call with resolve_chapter tool. Returns structured routing decision."""
    start = time.monotonic()
    teacher_tools = get_definitions_by_names(["resolve_chapter"])

    # IMPORTANT: copy messages so tool_call/tool_result don't leak into session
    teacher_messages = list(messages)

    # Run attempt — teacher may call resolve_chapter before producing JSON
    full_text = ""
    async for event in run_attempt(
        system_prompt=system_prompt,
        messages=teacher_messages,
        tools=teacher_tools,
        provider=provider,
        model=settings.fast_model,
        response_format=_JSON_FORMAT,
    ):
        if isinstance(event, ResponseDelta):
            full_text += event.content

    elapsed = round(time.monotonic() - start, 2)
    logger.info("[%s] Teacher routing completed in %.2fs", trace_id, elapsed)

    # Parse JSON response into TeacherDecision
    try:
        data = json.loads(full_text)
    except json.JSONDecodeError:
        logger.error("Teacher returned invalid JSON: %s", full_text[:500])
        return TeacherDecision(
            type="direct_response",
            direct_response="Kuch technical issue aa raha hai, thodi der mein try karo.",
        )

    # Build SubAgentDispatch list
    subagents = []
    for sa in data.get("subagents", []):
        subagents.append(SubAgentDispatch(
            agent=sa.get("agent", ""),
            input=sa.get("input", ""),
            language=sa.get("language", "hinglish"),
            nudge=sa.get("nudge"),
            content_types=sa.get("content_types", []),
            chapter_id=sa.get("chapter_id"),
            chapter_name=sa.get("chapter_name"),
        ))

    return TeacherDecision(
        type=data.get("type", "direct_response"),
        direct_response=data.get("direct_response"),
        chapter_id=data.get("chapter_id"),
        chapter_name=data.get("chapter_name"),
        subagents=subagents,
        follow_up=data.get("follow_up"),
    )


# ── Subagent dispatch ──


async def _dispatch_subagent(
    dispatch: SubAgentDispatch,
    message: UniversalMessage,
    provider: OpenAIProvider,
) -> SubAgentResult:
    """Run a single subagent based on the dispatch instructions."""
    common = {
        "input_text": dispatch.input,
        "language": dispatch.language,
        "class_": message.class_,
        "subject": message.subject,
        "course_id": message.course_id,
        "provider": provider,
    }

    if dispatch.agent == "doubt":
        return await run_doubt_agent(**common)

    if dispatch.agent == "content":
        return await run_content_agent(
            **common,
            content_types=dispatch.content_types,
            chapter_name=dispatch.chapter_name,
            chapter_id=dispatch.chapter_id,
        )

    if dispatch.agent == "guidance":
        return await run_guidance_agent(**common)

    logger.warning("Unknown subagent type: %s", dispatch.agent)
    return SubAgentResult(status="error", metadata={"error": f"Unknown agent: {dispatch.agent}"})


# ── Response assembly ──


def _assemble_response(
    decision: TeacherDecision,
    results: list[SubAgentResult],
) -> tuple[str, list[dict]]:
    """Programmatically assemble text and cards from subagent results."""
    text_parts: list[str] = []
    all_cards: list[dict] = []

    for result in results:
        if result.text:
            text_parts.append(result.text)
        if result.cards:
            all_cards.extend(result.cards)

    combined_text = "\n\n".join(text_parts) if text_parts else ""
    return combined_text, all_cards


# ── Public API ──


async def run(message: UniversalMessage) -> AgentResponse:
    """Teacher orchestrator: routes to subagents, assembles response."""
    trace_id = uuid.uuid4().hex[:12]
    session_mgr = get_session_manager()
    provider = get_provider()

    # 1. Load session
    session = await session_mgr.load(
        chat_id=message.chat_id,
        user_id=message.user_id,
        subject=message.subject,
        course_id=message.course_id,
    )

    # 2. Build teacher prompt
    system_prompt = build_teacher_prompt(
        course_id=message.course_id,
        class_=message.class_,
        subject=message.subject,
        language=message.language,
    )

    # 3. Append student message
    student_msg = InternalMessage(role="student", content=message.text)
    session.messages.append(student_msg)

    logger.info(
        "[%s] run() — chat_id=%s msg_count=%d",
        trace_id,
        message.chat_id,
        len(session.messages),
    )

    # 4. Teacher LLM call (routing + resolve_chapter)
    decision = await _run_teacher(
        system_prompt, session.messages, provider, trace_id,
    )

    logger.info(
        "[%s] Teacher decision: type=%s subagents=%s",
        trace_id,
        decision.type,
        [s.agent for s in decision.subagents],
    )

    # 5. Route
    if decision.type == "direct_response":
        text = decision.direct_response or ""
        cards: list[dict] = []
        follow_up = decision.follow_up
    else:
        # Dispatch subagents in parallel with timeout
        tasks = [
            asyncio.wait_for(
                _dispatch_subagent(d, message, provider),
                timeout=_SUBAGENT_TIMEOUT_S,
            )
            for d in decision.subagents
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions (including TimeoutError)
        clean_results: list[SubAgentResult] = []
        for i, r in enumerate(results):
            if isinstance(r, BaseException):
                agent_name = decision.subagents[i].agent
                logger.error("[%s] Subagent %s failed: %s", trace_id, agent_name, r)
                clean_results.append(
                    SubAgentResult(status="error", metadata={"error": str(r)})
                )
            else:
                clean_results.append(r)

        text, cards = _assemble_response(decision, clean_results)
        follow_up = decision.follow_up

        # If all subagents failed and no text, provide fallback
        if not text and not cards:
            text = "Abhi kuch technical issue aa raha hai, thodi der mein try karo."

    # 6. Save session
    session.messages.append(InternalMessage(role="agent", content=text))
    await session_mgr.save(session)

    return AgentResponse(
        text=text,
        cards=cards,
        follow_up=follow_up,
        metadata={
            "chat_id": message.chat_id,
            "trace_id": trace_id,
            "teacher_model": settings.fast_model,
            "routing": decision.type,
        },
    )


def _get_text_stream(
    dispatch: SubAgentDispatch,
    message: UniversalMessage,
    provider: OpenAIProvider,
) -> AsyncGenerator[str, None] | None:
    """Return a token stream for text-producing agents, or None for card agents."""
    common = {
        "input_text": dispatch.input,
        "language": dispatch.language,
        "class_": message.class_,
        "subject": message.subject,
        "course_id": message.course_id,
        "provider": provider,
    }
    if dispatch.agent == "doubt":
        return stream_doubt_agent(**common)
    if dispatch.agent == "guidance":
        return stream_guidance_agent(**common)
    return None


async def run_stream(message: UniversalMessage) -> AsyncGenerator[AgentEvent, None]:
    """Streaming teacher orchestrator: streams text tokens in real-time."""
    trace_id = uuid.uuid4().hex[:12]
    session_mgr = get_session_manager()
    provider = get_provider()

    # 1. Load session
    session = await session_mgr.load(
        chat_id=message.chat_id,
        user_id=message.user_id,
        subject=message.subject,
        course_id=message.course_id,
    )

    # 2. Build teacher prompt
    system_prompt = build_teacher_prompt(
        course_id=message.course_id,
        class_=message.class_,
        subject=message.subject,
        language=message.language,
    )

    # 3. Append student message
    student_msg = InternalMessage(role="student", content=message.text)
    session.messages.append(student_msg)

    # 4. Teacher routing
    yield StatusEvent(content="Samajh raha hoon kya chahiye...")

    decision = await _run_teacher(
        system_prompt, session.messages, provider, trace_id,
    )

    logger.info(
        "[%s] Teacher decision (stream): type=%s subagents=%s",
        trace_id,
        decision.type,
        [s.agent for s in decision.subagents],
    )

    # 5. Route and stream results
    if decision.type == "direct_response":
        text = decision.direct_response or ""
        yield ResponseStartEvent()
        yield ResponseDelta(content=text)
        yield ResponseEndEvent()

        if decision.follow_up:
            yield FollowUpEvent(content=decision.follow_up)

        session.messages.append(InternalMessage(role="agent", content=text))

    else:
        # Separate text-producing agents (doubt/guidance) from card agents (content)
        text_dispatches: list[SubAgentDispatch] = []
        card_dispatches: list[SubAgentDispatch] = []
        for d in decision.subagents:
            if d.agent in ("doubt", "guidance"):
                text_dispatches.append(d)
            else:
                card_dispatches.append(d)

        # Emit nudges
        for dispatch in decision.subagents:
            if dispatch.nudge:
                yield StatusEvent(content=dispatch.nudge)

        # Start card agents (content) as background tasks
        card_tasks = [
            asyncio.create_task(
                asyncio.wait_for(
                    _dispatch_subagent(d, message, provider),
                    timeout=_SUBAGENT_TIMEOUT_S,
                )
            )
            for d in card_dispatches
        ]

        # Stream text agents token-by-token
        collected_text = ""
        if text_dispatches:
            yield ResponseStartEvent()
            for i, dispatch in enumerate(text_dispatches):
                if i > 0:
                    sep = "\n\n"
                    yield ResponseDelta(content=sep)
                    collected_text += sep

                raw_stream = _get_text_stream(dispatch, message, provider)
                if raw_stream is not None:
                    async for token in filter_thinking_stream(raw_stream):
                        yield ResponseDelta(content=token)
                        collected_text += token
            yield ResponseEndEvent()

        # Await card agent results
        cards: list[dict] = []
        for i, task in enumerate(card_tasks):
            try:
                result = await task
                if result.cards:
                    cards.extend(result.cards)
            except BaseException as e:
                agent_name = card_dispatches[i].agent
                logger.error(
                    "[%s] Card agent %s failed: %s", trace_id, agent_name, e,
                )

        # Fallback if everything failed
        if not collected_text and not cards:
            collected_text = (
                "Abhi kuch technical issue aa raha hai, thodi der mein try karo."
            )
            yield ResponseStartEvent()
            yield ResponseDelta(content=collected_text)
            yield ResponseEndEvent()

        # Emit cards
        if cards:
            yield CardsEvent(content=cards)

        # Emit follow-up
        if decision.follow_up:
            yield FollowUpEvent(content=decision.follow_up)

        session.messages.append(
            InternalMessage(role="agent", content=collected_text),
        )

    # 6. Save session
    await session_mgr.save(session)
