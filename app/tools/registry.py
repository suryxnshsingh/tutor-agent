import logging
from typing import Any, Callable, Coroutine

from app.llm.base import ToolDefinition

logger = logging.getLogger(__name__)

# Registry mapping tool names to their async handler functions
_tool_handlers: dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {}

# Registry mapping tool names to their definitions
_tool_definitions: dict[str, ToolDefinition] = {}


def register_tool(definition: ToolDefinition, handler: Callable[..., Coroutine[Any, Any, Any]]):
    """Register a tool with its definition and handler."""
    _tool_handlers[definition.name] = handler
    _tool_definitions[definition.name] = definition


def get_all_definitions() -> list[ToolDefinition]:
    """Return all registered tool definitions."""
    return list(_tool_definitions.values())


def get_definitions_by_names(names: list[str]) -> list[ToolDefinition]:
    """Return tool definitions for a specific list of tool names."""
    return [_tool_definitions[n] for n in names if n in _tool_definitions]


async def execute_tool(tool_name: str, tool_input: dict) -> Any:
    """Execute a registered tool by name."""
    handler = _tool_handlers.get(tool_name)
    if not handler:
        return {"error": f"Unknown tool: {tool_name}"}

    try:
        return await handler(**tool_input)
    except Exception as e:
        logger.exception("Tool %s failed with %s", tool_name, e)
        return {"error": str(e)}
