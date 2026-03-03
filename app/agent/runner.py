from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

from app.agent.attempt import run_attempt
from app.agent.events import (
    AgentEvent,
    CardsEvent,
    ResponseDelta,
    ResponseEndEvent,
    ResponseStartEvent,
)
from app.agent.prompt import build_system_prompt, extract_thinking, strip_thinking
from app.config import settings
from app.gateway.models import UniversalMessage
from app.llm.openai_provider import OpenAIProvider
from app.messages.models import InternalMessage
from app.session.manager import SessionManager
from app.tools.registry import get_all_definitions

logger = logging.getLogger(__name__)

# Tool results that contain resource data to surface as cards.
# Maps tool name -> key in the result dict that holds the list of items.
_RESOURCE_TOOLS: dict[str, str] = {
    "search_lectures": "lectures",
    "search_topper_notes": "notes",
    "search_ppt_notes": "notes",
    "search_pyq_papers": "papers",
    "search_important_questions": "results",
    "search_tests": "tests",
    "search_ncert_solutions": "solutions",
}


def _normalize_language(language: str) -> str:
    """Map request language to CSV language value. Hinglish -> English."""
    lang = language.strip().lower()
    if lang in ("hindi", "hin", "hi", "हिंदी"):
        return "hindi"
    return "english"


def _filter_by_language(items: list[dict], language: str) -> list[dict]:
    """Keep only items matching the student's language, if they have one."""
    filtered = [
        item for item in items
        if "language" not in item
        or item["language"].strip().lower() == language
    ]
    # If filtering removes everything, return all (don't lose data)
    return filtered if filtered else items


def _extract_cards(
    messages: list[InternalMessage],
    language: str,
) -> list[dict]:
    """Extract resource cards from tool_result messages."""
    cards: list[dict] = []
    for msg in messages:
        if msg.role != "tool_result" or not isinstance(msg.result, dict):
            continue
        # Find the matching tool_call to get the tool name
        tool_name = None
        for prev in messages:
            if prev.role == "tool_call" and prev.call_id == msg.call_id:
                tool_name = prev.tool_name
                break
        if not tool_name or tool_name not in _RESOURCE_TOOLS:
            continue
        items_key = _RESOURCE_TOOLS[tool_name]
        items = msg.result.get(items_key, [])
        items = _filter_by_language(items, language)
        if items:
            cards.append({
                "type": tool_name,
                "data": items,
            })
    return cards


@dataclass
class AgentResponse:
    """Final structured response from the agent."""

    text: str
    cards: list[dict] = field(default_factory=list)
    buttons: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    thinking: str | None = None


# Module-level session manager — initialized in main.py on startup
_session_manager: SessionManager | None = None


def init_session_manager(manager: SessionManager) -> None:
    global _session_manager
    _session_manager = manager


def get_session_manager() -> SessionManager:
    if _session_manager is None:
        raise RuntimeError("SessionManager not initialized. Call init_session_manager() first.")
    return _session_manager


async def run(message: UniversalMessage) -> AgentResponse:
    """Session-aware runner: loads history, runs attempt, saves session."""

    session_mgr = get_session_manager()

    # 1. Load or create session
    session = await session_mgr.load(
        chat_id=message.chat_id,
        user_id=message.user_id,
        subject=message.subject,
        course_id=message.course_id,
    )

    # 2. Build system prompt
    system_prompt = build_system_prompt(
        course_id=message.course_id,
        class_=message.class_,
        subject=message.subject,
        language=message.language,
    )

    # 3. Append student message to session history
    student_msg = InternalMessage(role="student", content=message.text)
    session.messages.append(student_msg)

    # 4. Get tool definitions
    tools = get_all_definitions()

    # 5. Create provider
    provider = OpenAIProvider(api_key=settings.openai_api_key)

    logger.info(
        "run() — chat_id=%s model=%s tools=%s msg_count=%d",
        message.chat_id,
        settings.primary_model,
        [t.name for t in tools],
        len(session.messages),
    )

    # 6. Run attempt and collect events
    full_text = ""
    msg_count_before = len(session.messages)

    async for event in run_attempt(
        system_prompt=system_prompt,
        messages=session.messages,
        tools=tools,
        provider=provider,
        model=settings.primary_model,
    ):
        if isinstance(event, ResponseDelta):
            full_text += event.content

    # 7. Extract resource cards from this request's tool results only
    new_messages = session.messages[msg_count_before:]
    lang = _normalize_language(message.language)
    cards = _extract_cards(new_messages, lang)

    # 8. Strip thinking tags
    thinking = extract_thinking(full_text)
    clean_text = strip_thinking(full_text)

    # 9. Append agent response to session history
    session.messages.append(InternalMessage(role="agent", content=clean_text))

    # 10. Save session (Redis sync, DynamoDB fire-and-forget)
    await session_mgr.save(session)

    return AgentResponse(
        text=clean_text,
        cards=cards,
        metadata={"chat_id": message.chat_id, "model": settings.primary_model},
        thinking=thinking,
    )


async def run_stream(message: UniversalMessage) -> AsyncGenerator[AgentEvent, None]:
    """Streaming runner: yields AgentEvents in real-time, saves session after stream completes."""

    session_mgr = get_session_manager()

    # 1. Load or create session
    session = await session_mgr.load(
        chat_id=message.chat_id,
        user_id=message.user_id,
        subject=message.subject,
        course_id=message.course_id,
    )

    # 2. Build system prompt
    system_prompt = build_system_prompt(
        course_id=message.course_id,
        class_=message.class_,
        subject=message.subject,
        language=message.language,
    )

    # 3. Append student message
    student_msg = InternalMessage(role="student", content=message.text)
    session.messages.append(student_msg)

    # 4. Get tool definitions and provider
    tools = get_all_definitions()
    provider = OpenAIProvider(api_key=settings.openai_api_key)

    # 5. Stream events, collecting full text and stripping thinking from deltas
    full_text = ""
    started = False

    async for event in run_attempt(
        system_prompt=system_prompt,
        messages=session.messages,
        tools=tools,
        provider=provider,
        model=settings.primary_model,
    ):
        if isinstance(event, ResponseDelta):
            chunk = event.content
            full_text += chunk

            # Strip thinking tags from the delta stream
            clean = strip_thinking(full_text)
            if clean:
                if not started:
                    yield ResponseStartEvent()
                    started = True
                # Only yield the new clean content (delta)
                already_sent = len(full_text) - len(chunk)
                prev_clean = strip_thinking(full_text[:already_sent])
                new_clean = clean[len(prev_clean):]
                if new_clean:
                    yield ResponseDelta(content=new_clean)

        elif isinstance(event, (ResponseStartEvent, ResponseEndEvent)):
            if isinstance(event, ResponseEndEvent):
                if started:
                    yield ResponseEndEvent()
                else:
                    clean = strip_thinking(full_text)
                    if clean:
                        yield ResponseStartEvent()
                        yield ResponseDelta(content=clean)
                        yield ResponseEndEvent()
        elif isinstance(event, CardsEvent):
            yield event
        else:
            # StatusEvent, ErrorEvent — pass through
            yield event

    # 6. Finalize and save
    clean_text = strip_thinking(full_text)
    session.messages.append(InternalMessage(role="agent", content=clean_text))
    await session_mgr.save(session)
