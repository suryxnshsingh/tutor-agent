from __future__ import annotations

import json
from dataclasses import asdict

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from app.agent.runner import run, run_stream
from app.gateway.models import ChatRequest, ChatResponse, UniversalMessage

router = APIRouter()


def _to_universal(request: ChatRequest) -> UniversalMessage:
    return UniversalMessage(
        user_id=request.user_id,
        text=request.text,
        course_id=request.course_id,
        class_=request.class_,
        subject=request.subject,
        language=request.language,
        chat_id=request.chat_id,
        channel=request.channel,
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Non-streaming chat endpoint. Accepts a message, returns full response."""
    message = _to_universal(request)
    response = await run(message)

    return ChatResponse(
        text=response.text,
        cards=response.cards,
        buttons=response.buttons,
        metadata=response.metadata,
        follow_up=response.follow_up,
    )


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> EventSourceResponse:
    """SSE streaming endpoint. Yields AgentEvents as server-sent events."""
    message = _to_universal(request)

    async def event_generator():
        async for event in run_stream(message):
            yield {"data": json.dumps(asdict(event))}

    return EventSourceResponse(event_generator())
