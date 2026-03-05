from __future__ import annotations

import json
import logging

from app.config import settings
from app.llm.base import LLMProvider
from app.llm.retry import chat_with_retry
from app.messages.models import ChatSession, InternalMessage

logger = logging.getLogger(__name__)

SUMMARIZATION_PROMPT = (
    "You are a conversation summarizer for an AI tutoring agent. "
    "Summarize the following conversation history concisely, preserving: "
    "1) Key topics discussed, 2) Important facts about the student, "
    "3) Any commitments or plans made, 4) Tool results that were important. "
    "Write in the same language mix (Hindi/English/Hinglish) the student used. "
    "Keep it under 500 words."
)


def needs_compaction(session: ChatSession) -> bool:
    """Check if session has too many messages and needs compaction."""
    return len(session.messages) > settings.compaction_threshold


def _build_summary_text(messages: list[InternalMessage]) -> str:
    """Build a text representation of messages for summarization."""
    parts: list[str] = []
    for msg in messages:
        if msg.role == "student":
            parts.append(f"Student: {msg.content}")
        elif msg.role == "agent":
            parts.append(f"Agent: {msg.content}")
        elif msg.role == "tool_call":
            parts.append(f"[Tool called: {msg.tool_name}]")
        elif msg.role == "tool_result":
            result_str = msg.result if isinstance(msg.result, str) else json.dumps(msg.result)
            truncated = result_str[:500] + "..." if len(result_str) > 500 else result_str
            parts.append(f"[Tool result: {truncated}]")
    return "\n".join(parts)


async def compact_session(
    session: ChatSession,
    provider: LLMProvider,
) -> ChatSession:
    """Compact session by summarizing old messages."""
    messages = session.messages
    keep_count = settings.recent_turns_to_keep * 2  # student+agent pairs

    # Find cutoff: count backwards to keep recent_turns_to_keep conversation turns
    role_count = 0
    cutoff_index = len(messages)
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role in ("student", "agent"):
            role_count += 1
            if role_count >= keep_count:
                cutoff_index = i
                break

    if cutoff_index <= 0:
        return session

    old_messages = messages[:cutoff_index]
    recent_messages = messages[cutoff_index:]

    summary_text = _build_summary_text(old_messages)

    logger.info(
        "Compacting session %s: %d old messages -> summary, keeping %d recent",
        session.chat_id, len(old_messages), len(recent_messages),
    )

    try:
        summary_response = await chat_with_retry(
            provider=provider,
            system_prompt=SUMMARIZATION_PROMPT,
            messages=[InternalMessage(role="student", content=summary_text)],
            tools=[],
            primary_model=settings.compaction_model,
            fallback_model=settings.fallback_model,
        )
        summary_content = summary_response.text or "[Summary unavailable]"
    except Exception:
        logger.exception("Compaction LLM call failed for chat_id=%s", session.chat_id)
        return session  # Keep original if compaction fails

    summary_msg = InternalMessage(
        role="agent",
        content=f"[Previous conversation summary]\n{summary_content}",
        metadata={"is_compaction_summary": True},
    )

    session.messages = [summary_msg] + recent_messages
    return session


def get_messages_for_llm(messages: list[InternalMessage]) -> list[InternalMessage]:
    """Return messages from the latest compaction summary onwards.

    If no compaction summary exists, returns all messages.
    Used when loading from DynamoDB to avoid sending the entire history to the LLM.
    """
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].metadata.get("is_compaction_summary"):
            return messages[i:]
    return messages
