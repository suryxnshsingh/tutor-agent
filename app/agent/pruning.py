import json

from app.messages.models import InternalMessage

KEEP_RECENT_ASSISTANT_TURNS = 3
MAX_RESULT_CHARS = 1000
TRUNCATED_PREVIEW_CHARS = 200


def prune_tool_results(messages: list[InternalMessage]) -> list[InternalMessage]:
    """Prune old tool results to reduce context size.

    Keeps the last KEEP_RECENT_ASSISTANT_TURNS agent messages as boundaries.
    Tool results before the cutoff that exceed MAX_RESULT_CHARS are truncated.
    Returns a new list — never mutates originals.
    """
    # Find cutoff: index of the Nth-from-last agent message
    agent_indices = [i for i, m in enumerate(messages) if m.role == "agent"]

    if len(agent_indices) <= KEEP_RECENT_ASSISTANT_TURNS:
        return list(messages)

    cutoff_index = agent_indices[-KEEP_RECENT_ASSISTANT_TURNS]

    pruned: list[InternalMessage] = []
    for i, msg in enumerate(messages):
        if i < cutoff_index and msg.role == "tool_result":
            serialized = msg.result if isinstance(msg.result, str) else json.dumps(msg.result)
            if len(serialized) > MAX_RESULT_CHARS:
                # Find the tool name from a preceding tool_call
                tool_name = "unknown"
                for prev in reversed(messages[:i]):
                    if prev.role == "tool_call" and prev.call_id == msg.call_id:
                        tool_name = prev.tool_name or "unknown"
                        break
                preview = serialized[:TRUNCATED_PREVIEW_CHARS]
                truncated_result = f"[Tool result: {tool_name} — {preview}... trimmed]"
                pruned.append(InternalMessage(
                    role=msg.role,
                    call_id=msg.call_id,
                    result=truncated_result,
                    timestamp=msg.timestamp,
                    metadata=msg.metadata,
                ))
                continue
        pruned.append(msg)

    return pruned
