import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from openai import AsyncOpenAI

from app.llm.base import LLMProvider, LLMResponse, ToolDefinition
from app.messages.models import InternalMessage

logger = logging.getLogger(__name__)

# 30s connect, 60s read (LLM calls can take a while)
_DEFAULT_TIMEOUT = httpx.Timeout(connect=30.0, read=60.0, write=30.0, pool=30.0)


class OpenAIProvider(LLMProvider):
    """OpenAI chat completions adapter."""

    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key, timeout=_DEFAULT_TIMEOUT)

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
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    logger.error(
                        "Malformed tool args for %s: %s",
                        tc.function.name,
                        tc.function.arguments[:200],
                    )
                    args = {}
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": args,
                })

        # Extract token usage
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            logger.info(
                "LLM usage — prompt=%d completion=%d total=%d",
                usage["prompt_tokens"],
                usage["completion_tokens"],
                usage["total_tokens"],
            )

        return LLMResponse(
            text=message.content,
            tool_calls=tool_calls,
            has_tool_calls=bool(tool_calls),
            usage=usage,
            raw=response,
        )

    async def chat(
        self,
        system_prompt: str,
        messages: list[InternalMessage],
        tools: list[ToolDefinition],
        model: str,
        response_format: dict | None = None,
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
        if response_format:
            kwargs["response_format"] = response_format

        response = await self.client.chat.completions.create(**kwargs)
        return self.from_provider_response(response)

    async def chat_stream(
        self,
        system_prompt: str,
        messages: list[InternalMessage],
        model: str,
    ) -> AsyncGenerator[str, None]:
        """Stream token-by-token text from OpenAI. No tool support."""
        provider_messages = [
            {"role": "system", "content": system_prompt},
            *self.to_provider_messages(messages),
        ]

        stream = await self.client.chat.completions.create(
            model=model,
            messages=provider_messages,
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content
