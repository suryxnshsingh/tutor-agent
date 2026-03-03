from collections.abc import AsyncGenerator

from app.agent.events import (
    AgentEvent,
    ResponseDelta,
    ResponseEndEvent,
    ResponseStartEvent,
    StatusEvent,
)
from app.llm.base import LLMProvider, LLMResponse, ToolDefinition
from app.messages.models import InternalMessage
from app.tools.registry import execute_tool


async def run_attempt(
    system_prompt: str,
    messages: list[InternalMessage],
    tools: list[ToolDefinition],
    provider: LLMProvider,
    model: str,
) -> AsyncGenerator[AgentEvent, None]:
    """Run a single LLM attempt with tool loop. Stateless — no session or retry logic."""
    while True:
        response: LLMResponse = await provider.chat(
            system_prompt=system_prompt,
            messages=messages,
            tools=tools,
            model=model,
        )

        if response.has_tool_calls:
            for tool_call in response.tool_calls:
                yield StatusEvent(content=f"Using {tool_call['name']}...")

                # Execute the tool
                result = await execute_tool(tool_call["name"], tool_call["input"])

                # Append tool_call and tool_result to messages for next LLM iteration
                messages.append(
                    InternalMessage(
                        role="tool_call",
                        tool_name=tool_call["name"],
                        tool_input=tool_call["input"],
                        call_id=tool_call["id"],
                    )
                )
                messages.append(
                    InternalMessage(
                        role="tool_result",
                        call_id=tool_call["id"],
                        result=result,
                    )
                )
        else:
            # Final text response — yield as single delta (non-streaming Phase 1)
            text = response.text or ""
            yield ResponseStartEvent()
            yield ResponseDelta(content=text)
            yield ResponseEndEvent()
            return
