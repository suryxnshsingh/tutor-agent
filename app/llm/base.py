from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

from app.messages.models import InternalMessage


@dataclass
class ToolDefinition:
    """Provider-agnostic tool definition."""

    name: str
    description: str
    parameters: dict  # JSON Schema
    required_params: list[str] = field(default_factory=list)


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""

    text: str | None = None
    tool_calls: list[dict] = field(default_factory=list)
    has_tool_calls: bool = False
    raw: Any = None


@dataclass
class LLMStreamDelta:
    """A single chunk from a streaming LLM response."""

    text_delta: str | None = None
    tool_call_index: int | None = None
    tool_call_id: str | None = None
    tool_call_name: str | None = None
    tool_call_arguments_delta: str | None = None
    finish_reason: str | None = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def chat(
        self,
        system_prompt: str,
        messages: list[InternalMessage],
        tools: list[ToolDefinition],
        model: str,
    ) -> LLMResponse:
        """Send a chat request to the LLM provider."""
        ...

    @abstractmethod
    async def chat_stream(
        self,
        system_prompt: str,
        messages: list[InternalMessage],
        tools: list[ToolDefinition],
        model: str,
    ) -> AsyncGenerator[LLMStreamDelta, None]:
        """Send a streaming chat request to the LLM provider."""
        ...

    @abstractmethod
    def to_provider_messages(self, messages: list[InternalMessage]) -> list[dict]:
        """Convert internal messages to provider format."""
        ...

    @abstractmethod
    def to_provider_tools(self, tools: list[ToolDefinition]) -> list[dict]:
        """Convert internal tool definitions to provider format."""
        ...

    @abstractmethod
    def from_provider_response(self, response: Any) -> list[InternalMessage]:
        """Convert provider response to internal messages."""
        ...
