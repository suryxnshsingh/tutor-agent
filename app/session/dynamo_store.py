from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any

import boto3

from app.config import settings
from app.messages.models import ChatSession, InternalMessage


class DynamoStore:
    def __init__(self) -> None:
        self.table_name = settings.dynamo_table
        kwargs: dict[str, Any] = {"region_name": settings.aws_region}
        if settings.dynamo_endpoint:
            kwargs["endpoint_url"] = settings.dynamo_endpoint
        self._resource = boto3.resource("dynamodb", **kwargs)
        self._table = self._resource.Table(self.table_name)

    async def get(self, chat_id: str) -> ChatSession | None:
        response = await asyncio.to_thread(
            self._table.get_item, Key={"chat_id": chat_id}
        )
        item = response.get("Item")
        if item is None:
            return None
        return _deserialize_session(item)

    async def save(self, session: ChatSession) -> None:
        item = _serialize_session(session)
        await asyncio.to_thread(self._table.put_item, Item=item)


# --- Serialization helpers ---


def _serialize_message(msg: InternalMessage) -> dict[str, Any]:
    d: dict[str, Any] = {"role": msg.role}
    if msg.content is not None:
        d["content"] = msg.content
    if msg.tool_name is not None:
        d["tool_name"] = msg.tool_name
    if msg.tool_input is not None:
        d["tool_input"] = json.dumps(msg.tool_input)
    if msg.call_id is not None:
        d["call_id"] = msg.call_id
    if msg.result is not None:
        d["result"] = json.dumps(msg.result) if not isinstance(msg.result, str) else msg.result
    d["timestamp"] = msg.timestamp.isoformat()
    if msg.metadata:
        d["metadata"] = json.dumps(msg.metadata)
    return d


def _deserialize_message(d: dict) -> InternalMessage:
    tool_input = d.get("tool_input")
    if isinstance(tool_input, str):
        tool_input = json.loads(tool_input)

    result = d.get("result")
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except (json.JSONDecodeError, ValueError):
            pass

    metadata = d.get("metadata", {})
    if isinstance(metadata, str):
        metadata = json.loads(metadata)

    return InternalMessage(
        role=d["role"],
        content=d.get("content"),
        tool_name=d.get("tool_name"),
        tool_input=tool_input,
        call_id=d.get("call_id"),
        result=result,
        timestamp=datetime.fromisoformat(d["timestamp"]) if "timestamp" in d else datetime.utcnow(),
        metadata=metadata,
    )


def _serialize_session(session: ChatSession) -> dict[str, Any]:
    return {
        "chat_id": session.chat_id,
        "user_id": session.user_id,
        "subject": session.subject,
        "course_id": session.course_id,
        "messages": [_serialize_message(m) for m in session.messages],
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
    }


def _deserialize_session(item: dict) -> ChatSession:
    messages_raw = item.get("messages", [])
    return ChatSession(
        chat_id=item["chat_id"],
        user_id=item["user_id"],
        subject=item["subject"],
        course_id=item["course_id"],
        messages=[_deserialize_message(m) for m in messages_raw],
        created_at=datetime.fromisoformat(item["created_at"]),
        updated_at=datetime.fromisoformat(item["updated_at"]),
    )
