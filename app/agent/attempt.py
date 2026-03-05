import json
import logging
from collections.abc import AsyncGenerator

from app.agent.events import (
    AgentEvent,
    ResponseDelta,
    ResponseEndEvent,
    ResponseStartEvent,
    StatusEvent,
)
from app.agent.pruning import prune_tool_results
from app.config import settings
from app.llm.base import LLMProvider, LLMResponse, LLMStreamDelta, ToolDefinition
from app.llm.retry import chat_with_retry
from app.messages.models import InternalMessage
from app.tools.registry import execute_tool

logger = logging.getLogger(__name__)

# Tool loop detection constants
MAX_ITERATIONS = 10
MAX_PER_TOOL = 3


async def run_attempt(
    system_prompt: str,
    messages: list[InternalMessage],
    tools: list[ToolDefinition],
    provider: LLMProvider,
    model: str,
) -> AsyncGenerator[AgentEvent, None]:
    """Run a single LLM attempt with tool loop. Uses retry + failover and pruning."""
    logger.info(
        "Attempt started — %d tools registered: %s",
        len(tools),
        [t.name for t in tools],
    )
    iteration = 0
    tool_call_counts: dict[str, int] = {}

    while True:
        iteration += 1

        # Check total iteration limit
        if iteration > MAX_ITERATIONS:
            logger.warning("Max iterations (%d) reached, stopping tool loop", MAX_ITERATIONS)
            yield ResponseStartEvent()
            yield ResponseDelta(
                content="Bahut zyada steps ho gaye, let me answer with what I have so far. "
                "Kya aap apna question thoda aur specific kar sakte hain?"
            )
            yield ResponseEndEvent()
            return

        # Prune old tool results before sending to LLM
        pruned = prune_tool_results(messages)

        response: LLMResponse = await chat_with_retry(
            provider=provider,
            system_prompt=system_prompt,
            messages=pruned,
            tools=tools,
            primary_model=model,
            fallback_model=settings.fallback_model,
        )

        if response.has_tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]

                # Check per-tool limit
                tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
                if tool_call_counts[tool_name] > MAX_PER_TOOL:
                    logger.warning(
                        "Tool %s called %d times (limit %d), skipping",
                        tool_name, tool_call_counts[tool_name], MAX_PER_TOOL,
                    )
                    messages.append(
                        InternalMessage(
                            role="tool_call",
                            tool_name=tool_name,
                            tool_input=tool_call["input"],
                            call_id=tool_call["id"],
                        )
                    )
                    messages.append(
                        InternalMessage(
                            role="tool_result",
                            call_id=tool_call["id"],
                            result=f"Error: {tool_name} has been called too many times. "
                            "Please use the information you already have to answer the student's question.",
                        )
                    )
                    continue

                logger.info(
                    "Iteration %d — tool call: %s(%s)",
                    iteration,
                    tool_name,
                    tool_call["input"],
                )
                yield StatusEvent(content=f"Using {tool_name}...")

                result = await execute_tool(tool_name, tool_call["input"])
                logger.info(
                    "Tool %s returned %d chars",
                    tool_name,
                    len(str(result)),
                )

                messages.append(
                    InternalMessage(
                        role="tool_call",
                        tool_name=tool_name,
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


async def run_attempt_streaming(
    system_prompt: str,
    messages: list[InternalMessage],
    tools: list[ToolDefinition],
    provider: LLMProvider,
    model: str,
) -> AsyncGenerator[AgentEvent, None]:
    """Streaming attempt: yields ResponseDelta token-by-token. Same tool loop as non-streaming."""
    logger.info(
        "Streaming attempt started — %d tools registered: %s",
        len(tools),
        [t.name for t in tools],
    )
    iteration = 0
    tool_call_counts: dict[str, int] = {}

    while True:
        iteration += 1

        if iteration > MAX_ITERATIONS:
            logger.warning("Max iterations (%d) reached in streaming attempt", MAX_ITERATIONS)
            yield ResponseStartEvent()
            yield ResponseDelta(
                content="Bahut zyada steps ho gaye, let me answer with what I have so far. "
                "Kya aap apna question thoda aur specific kar sakte hain?"
            )
            yield ResponseEndEvent()
            return

        pruned = prune_tool_results(messages)

        # Accumulate stream: text and tool calls
        accumulated_text = ""
        # tool_calls_by_index: {index: {id, name, arguments}}
        tool_calls_by_index: dict[int, dict] = {}
        response_started = False
        finish_reason: str | None = None

        try:
            stream = provider.chat_stream(
                system_prompt=system_prompt,
                messages=pruned,
                tools=tools,
                model=model,
            )

            async for delta in stream:
                # Text delta — yield immediately
                if delta.text_delta:
                    if not response_started:
                        yield ResponseStartEvent()
                        response_started = True
                    accumulated_text += delta.text_delta
                    yield ResponseDelta(content=delta.text_delta)

                # Tool call delta — accumulate
                if delta.tool_call_index is not None:
                    idx = delta.tool_call_index
                    if idx not in tool_calls_by_index:
                        tool_calls_by_index[idx] = {"id": "", "name": "", "arguments": ""}
                    tc = tool_calls_by_index[idx]
                    if delta.tool_call_id:
                        tc["id"] = delta.tool_call_id
                    if delta.tool_call_name:
                        tc["name"] = delta.tool_call_name
                    if delta.tool_call_arguments_delta:
                        tc["arguments"] += delta.tool_call_arguments_delta

                if delta.finish_reason:
                    finish_reason = delta.finish_reason

        except Exception as exc:
            # Retry once on stream connection failure
            logger.warning("Stream connection failed (attempt %d), retrying: %s", iteration, exc)
            try:
                stream = provider.chat_stream(
                    system_prompt=system_prompt,
                    messages=pruned,
                    tools=tools,
                    model=model,
                )
                async for delta in stream:
                    if delta.text_delta:
                        if not response_started:
                            yield ResponseStartEvent()
                            response_started = True
                        accumulated_text += delta.text_delta
                        yield ResponseDelta(content=delta.text_delta)

                    if delta.tool_call_index is not None:
                        idx = delta.tool_call_index
                        if idx not in tool_calls_by_index:
                            tool_calls_by_index[idx] = {"id": "", "name": "", "arguments": ""}
                        tc = tool_calls_by_index[idx]
                        if delta.tool_call_id:
                            tc["id"] = delta.tool_call_id
                        if delta.tool_call_name:
                            tc["name"] = delta.tool_call_name
                        if delta.tool_call_arguments_delta:
                            tc["arguments"] += delta.tool_call_arguments_delta

                    if delta.finish_reason:
                        finish_reason = delta.finish_reason
            except Exception:
                raise  # Let runner handle it

        # Process accumulated result
        if tool_calls_by_index:
            # Parse and execute tool calls
            for idx in sorted(tool_calls_by_index):
                tc = tool_calls_by_index[idx]
                tool_name = tc["name"]

                try:
                    tool_input = json.loads(tc["arguments"]) if tc["arguments"] else {}
                except json.JSONDecodeError:
                    tool_input = {}

                tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
                if tool_call_counts[tool_name] > MAX_PER_TOOL:
                    logger.warning("Tool %s called too many times in stream, skipping", tool_name)
                    messages.append(
                        InternalMessage(
                            role="tool_call",
                            tool_name=tool_name,
                            tool_input=tool_input,
                            call_id=tc["id"],
                        )
                    )
                    messages.append(
                        InternalMessage(
                            role="tool_result",
                            call_id=tc["id"],
                            result=f"Error: {tool_name} has been called too many times. "
                            "Please use the information you already have to answer the student's question.",
                        )
                    )
                    continue

                logger.info("Streaming iteration %d — tool call: %s", iteration, tool_name)
                yield StatusEvent(content=f"Using {tool_name}...")

                result = await execute_tool(tool_name, tool_input)
                logger.info("Tool %s returned %d chars", tool_name, len(str(result)))

                messages.append(
                    InternalMessage(
                        role="tool_call",
                        tool_name=tool_name,
                        tool_input=tool_input,
                        call_id=tc["id"],
                    )
                )
                messages.append(
                    InternalMessage(
                        role="tool_result",
                        call_id=tc["id"],
                        result=result,
                    )
                )
            # Continue tool loop
        else:
            # Text response — we're done
            if response_started:
                yield ResponseEndEvent()
            else:
                # Edge case: empty response
                yield ResponseStartEvent()
                yield ResponseDelta(content=accumulated_text)
                yield ResponseEndEvent()
            return
