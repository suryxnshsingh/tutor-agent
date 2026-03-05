import json
from collections.abc import AsyncGenerator
from typing import Any

from openai import AsyncOpenAI

from app.llm.base import LLMProvider, LLMResponse, LLMStreamDelta, ToolDefinition
from app.messages.models import InternalMessage


class OpenAIProvider(LLMProvider):
    """OpenAI chat completions adapter."""

    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)

    def to_provider_messages(self, messages: list[InternalMessage]) -> list[dict]:
        """Convert InternalMessage list to OpenAI message format."""
        provider_msgs: list[dict] = []

        i = 0
        while i < len(messages):
            msg = messages[i]

            if msg.role == "student":
                provider_msgs.append({"role": "user", "content": msg.content})

            elif msg.role == "agent":
                provider_msgs.append({"role": "assistant", "content": msg.content})

            elif msg.role == "tool_call":
                # Collect consecutive tool_calls into a single assistant message
                tool_calls = []
                while i < len(messages) and messages[i].role == "tool_call":
                    tc = messages[i]
                    tool_calls.append({
                        "id": tc.call_id,
                        "type": "function",
                        "function": {
                            "name": tc.tool_name,
                            "arguments": json.dumps(tc.tool_input or {}),
                        },
                    })
                    i += 1
                provider_msgs.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls,
                })
                # Don't increment i again at bottom — continue to avoid double increment
                continue

            elif msg.role == "tool_result":
                result = msg.result
                if not isinstance(result, str):
                    result = json.dumps(result)
                provider_msgs.append({
                    "role": "tool",
                    "tool_call_id": msg.call_id,
                    "content": result,
                })

            i += 1

        return provider_msgs

    def to_provider_tools(self, tools: list[ToolDefinition]) -> list[dict]:
        """Convert ToolDefinition list to OpenAI function calling format."""
        if not tools:
            return []

        provider_tools = []
        for tool in tools:
            schema = dict(tool.parameters)
            # Ensure required is set from our required_params if not in schema
            if tool.required_params and "required" not in schema:
                schema["required"] = tool.required_params

            provider_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": schema,
                },
            })
        return provider_tools

    def from_provider_response(self, response: Any) -> LLMResponse:
        """Parse OpenAI ChatCompletion response into LLMResponse."""
        choice = response.choices[0]
        message = choice.message

        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": json.loads(tc.function.arguments),
                })

        return LLMResponse(
            text=message.content,
            tool_calls=tool_calls,
            has_tool_calls=bool(tool_calls),
            raw=response,
        )

    async def chat(
        self,
        system_prompt: str,
        messages: list[InternalMessage],
        tools: list[ToolDefinition],
        model: str,
    ) -> LLMResponse:
        """Send a non-streaming chat request to OpenAI."""
        provider_messages = [
            {"role": "system", "content": system_prompt},
            *self.to_provider_messages(messages),
        ]
        provider_tools = self.to_provider_tools(tools)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": provider_messages,
        }
        if provider_tools:
            kwargs["tools"] = provider_tools

        response = await self.client.chat.completions.create(**kwargs)
        return self.from_provider_response(response)

    async def chat_stream(
        self,
        system_prompt: str,
        messages: list[InternalMessage],
        tools: list[ToolDefinition],
        model: str,
    ) -> AsyncGenerator[LLMStreamDelta, None]:
        """Send a streaming chat request to OpenAI. Yields LLMStreamDelta chunks."""
        provider_messages = [
            {"role": "system", "content": system_prompt},
            *self.to_provider_messages(messages),
        ]
        provider_tools = self.to_provider_tools(tools)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": provider_messages,
            "stream": True,
        }
        if provider_tools:
            kwargs["tools"] = provider_tools

        stream = await self.client.chat.completions.create(**kwargs)

        async for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            finish_reason = choice.finish_reason

            # Text delta
            if delta.content:
                yield LLMStreamDelta(
                    text_delta=delta.content,
                    finish_reason=finish_reason,
                )

            # Tool call deltas
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    yield LLMStreamDelta(
                        tool_call_index=tc_delta.index,
                        tool_call_id=tc_delta.id or None,
                        tool_call_name=tc_delta.function.name if tc_delta.function and tc_delta.function.name else None,
                        tool_call_arguments_delta=tc_delta.function.arguments if tc_delta.function and tc_delta.function.arguments else None,
                        finish_reason=finish_reason,
                    )

            # Finish reason without content (e.g., stop or tool_calls)
            if finish_reason and not delta.content and not delta.tool_calls:
                yield LLMStreamDelta(finish_reason=finish_reason)
