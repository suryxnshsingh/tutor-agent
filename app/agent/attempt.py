import logging
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

logger = logging.getLogger(__name__)


MAX_TOOL_ITERATIONS = 10


async def run_attempt(
    system_prompt: str,
    messages: list[InternalMessage],
    tools: list[ToolDefinition],
    provider: LLMProvider,
    model: str,
    response_format: dict | None = None,
    max_iterations: int = MAX_TOOL_ITERATIONS,
) -> AsyncGenerator[AgentEvent, None]:
    """Run a single LLM attempt with tool loop. Stateless — no session or retry logic."""
    logger.info(
        "Attempt started — %d tools registered: %s",
        len(tools),
        [t.name for t in tools],
    )
    iteration = 0
    while True:
        iteration += 1
        if iteration > max_iterations:
            logger.error(
                "Tool loop hit max iterations (%d), forcing stop",
                max_iterations,
            )
            yield ResponseStartEvent()
            yield ResponseDelta(
                content="Abhi request complete nahi ho payi, "
                "please dobara try karo."
            )
            yield ResponseEndEvent()
            return

        response: LLMResponse = await provider.chat(
            system_prompt=system_prompt,
            messages=messages,
            tools=tools,
            model=model,
            response_format=response_format,
        )

        if response.has_tool_calls:
            for tool_call in response.tool_calls:
                logger.info(
                    "Iteration %d — tool call: %s(%s)",
                    iteration,
                    tool_call["name"],
                    tool_call["input"],
                )
                yield StatusEvent(content=f"Using {tool_call['name']}...")

                # Execute the tool
                result = await execute_tool(tool_call["name"], tool_call["input"])
                logger.info(
                    "Tool %s returned %d chars",
                    tool_call["name"],
                    len(str(result)),
                )

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
            logger.info(
                "Iteration %d — final response (%d chars)",
                iteration,
                len(text),
            )
            yield ResponseStartEvent()
            yield ResponseDelta(content=text)
            yield ResponseEndEvent()
            return
