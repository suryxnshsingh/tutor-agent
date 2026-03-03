from dataclasses import dataclass

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Incoming chat request from any channel."""

    user_id: str
    text: str
    course_id: str
    class_: str = Field(alias="class")
    subject: str
    language: str = "hinglish"
    chat_id: str
    channel: str = "web"

    model_config = {"populate_by_name": True}


class ChatResponse(BaseModel):
    """Structured response sent back to client."""

    text: str
    cards: list[dict] = Field(default_factory=list)
    buttons: list[dict] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


@dataclass
class UniversalMessage:
    """Channel-agnostic message normalized by gateway."""

    user_id: str
    text: str
    course_id: str
    class_: str
    subject: str
    language: str
    chat_id: str
    channel: str
