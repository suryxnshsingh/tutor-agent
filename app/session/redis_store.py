from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import redis.asyncio as redis

from app.config import settings
from app.messages.models import ChatSession, InternalMessage


class RedisStore:
    def __init__(self) -> None:
        self.client: redis.Redis | None = None
        self._ttl_seconds = settings.redis_ttl_hours * 3600

    async def connect(self) -> None:
        self.client = redis.from_url(settings.redis_url, decode_responses=True)

    async def close(self) -> None:
        if self.client:
            await self.client.aclose()
            self.client = None

    def _key(self, chat_id: str) -> str:
        return f"chat:{chat_id}"

    async def get(self, chat_id: str) -> ChatSession | None:
        if not self.client:
            return None
        raw = await self.client.get(self._key(chat_id))
        if raw is None:
            return None
        return _deserialize_session(raw)

    async def save(self, session: ChatSession) -> None:
        if not self.client:
            return
        data = _serialize_session(session)
        await self.client.set(self._key(session.chat_id), data, ex=self._ttl_seconds)


# --- Serialization helpers ---


def _serialize_message(msg: InternalMessage) -> dict:
    d: dict[str, Any] = {"role": msg.role}
    if msg.content is not None:
        d["content"] = msg.content
    if msg.tool_name is not None:
        d["tool_name"] = msg.tool_name
    if msg.tool_input is not None:
        d["tool_input"] = msg.tool_input
    if msg.call_id is not None:
        d["call_id"] = msg.call_id
    if msg.result is not None:
        d["result"] = msg.result
    d["timestamp"] = msg.timestamp.isoformat()
    if msg.metadata:
        d["metadata"] = msg.metadata
    return d


def _deserialize_message(d: dict) -> InternalMessage:
    return InternalMessage(
        role=d["role"],
        content=d.get("content"),
        tool_name=d.get("tool_name"),
        tool_input=d.get("tool_input"),
        call_id=d.get("call_id"),
        result=d.get("result"),
        timestamp=datetime.fromisoformat(d["timestamp"]) if "timestamp" in d else datetime.utcnow(),
        metadata=d.get("metadata", {}),
    )


def _serialize_session(session: ChatSession) -> str:
    return json.dumps(
        {
            "chat_id": session.chat_id,
            "user_id": session.user_id,
            "subject": session.subject,
            "course_id": session.course_id,
            "messages": [_serialize_message(m) for m in session.messages],
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
        }
    )


def _deserialize_session(raw: str) -> ChatSession:
    d = json.loads(raw)
    return ChatSession(
        chat_id=d["chat_id"],
        user_id=d["user_id"],
        subject=d["subject"],
        course_id=d["course_id"],
        messages=[_deserialize_message(m) for m in d.get("messages", [])],
        created_at=datetime.fromisoformat(d["created_at"]),
        updated_at=datetime.fromisoformat(d["updated_at"]),
    )
