from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class InternalMessage:
    """Provider-agnostic message format for all conversation data."""

    role: str  # student | agent | tool_call | tool_result
    content: str | None = None
    tool_name: str | None = None
    tool_input: dict | None = None
    call_id: str | None = None
    result: Any | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)


@dataclass
class ChatSession:
    """A single chat session between a student and the agent."""

    chat_id: str
    user_id: str
    subject: str
    course_id: str
    messages: list[InternalMessage] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
