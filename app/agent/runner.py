from __future__ import annotations

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

    # 6. Run attempt and collect events
    full_text = ""
    cards: list[dict] = []

    async for event in run_attempt(
        system_prompt=system_prompt,
        messages=session.messages,
        tools=tools,
        provider=provider,
        model=settings.primary_model,
    ):
        if isinstance(event, ResponseDelta):
            full_text += event.content
        elif isinstance(event, CardsEvent):
            cards.extend(event.content)

    # 7. Strip thinking tags
    thinking = extract_thinking(full_text)
    clean_text = strip_thinking(full_text)

    # 8. Append agent response to session history
    session.messages.append(InternalMessage(role="agent", content=clean_text))

    # 9. Save session (Redis sync, DynamoDB fire-and-forget)
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
