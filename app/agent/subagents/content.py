from __future__ import annotations

import logging
import time

from app.agent.attempt import run_attempt
from app.agent.events import ResponseDelta
from app.agent.subagents.base import SubAgentResult
from app.config import settings
from app.llm.base import LLMProvider
from app.messages.models import InternalMessage
from app.tools.registry import get_definitions_by_names

logger = logging.getLogger(__name__)

_PROMPTS_DIR = __import__("pathlib").Path(__file__).parent.parent.parent.parent / "prompts"
_PROMPT_PATH = _PROMPTS_DIR / "subagents" / "content_prompt.txt"
_PROMPT_TEMPLATE = _PROMPT_PATH.read_text()

# Content tools available to this agent
_CONTENT_TOOLS = [
    "search_lectures",
    "search_topper_notes",
    "search_ppt_notes",
    "search_pyq_papers",
    "search_important_questions",
    "search_tests",
    "search_ncert_solutions",
]

# Maps tool name -> key in result dict that holds the list of items
_RESOURCE_KEYS: dict[str, str] = {
    "search_lectures": "lectures",
    "search_topper_notes": "notes",
    "search_ppt_notes": "notes",
    "search_pyq_papers": "papers",
    "search_important_questions": "results",
    "search_tests": "tests",
    "search_ncert_solutions": "solutions",
}


def _build_prompt(language: str, class_: str, subject: str, course_id: str) -> str:
    prompt = _PROMPT_TEMPLATE
    prompt = prompt.replace("{language}", language)
    prompt = prompt.replace("{class}", class_)
    prompt = prompt.replace("{subject}", subject)
    prompt = prompt.replace("{course}", course_id)
    return prompt


def _normalize_language(language: str) -> str:
    lang = language.strip().lower()
    if lang in ("hindi", "hin", "hi", "\u0939\u093f\u0902\u0926\u0940"):
        return "hindi"
    return "english"


def _filter_by_language(items: list[dict], language: str) -> list[dict]:
    filtered = [
        item for item in items
        if "language" not in item
        or item["language"].strip().lower() == language
    ]
    return filtered if filtered else items


def _extract_cards(messages: list[InternalMessage], language: str) -> list[dict]:
    """Extract resource cards from tool_result messages."""
    cards: list[dict] = []
    for msg in messages:
        if msg.role != "tool_result" or not isinstance(msg.result, dict):
            continue
        tool_name = None
        for prev in messages:
            if prev.role == "tool_call" and prev.call_id == msg.call_id:
                tool_name = prev.tool_name
                break
        if not tool_name or tool_name not in _RESOURCE_KEYS:
            continue
        items_key = _RESOURCE_KEYS[tool_name]
        items = msg.result.get(items_key, [])
        items = _filter_by_language(items, language)
        if items:
            cards.append({"type": tool_name, "data": items})
    return cards


async def run_content_agent(
    *,
    input_text: str,
    content_types: list[str],
    language: str,
    class_: str,
    subject: str,
    course_id: str,
    chapter_name: str | None = None,
    chapter_id: int | None = None,
    provider: LLMProvider,
) -> SubAgentResult:
    """Run the content research subagent. Uses search tools via attempt loop."""
    start = time.monotonic()

    system_prompt = _build_prompt(language, class_, subject, course_id)

    # Build the user message with all context the LLM needs to pick the right tools
    parts = [f"Fetch the following content: {input_text}"]
    parts.append(f"Content types requested: {', '.join(content_types)}")
    parts.append(f"Subject: {subject}, Course ID: {course_id}, Class: {class_}")
    if chapter_name:
        parts.append(f"Chapter: {chapter_name}")
    if chapter_id is not None:
        parts.append(f"Chapter number: {chapter_id}")
    parts.append(f"Language filter: {language}")

    user_msg = "\n".join(parts)
    messages: list[InternalMessage] = [InternalMessage(role="student", content=user_msg)]

    tools = get_definitions_by_names(_CONTENT_TOOLS)

    # Run the attempt loop — LLM will call search tools then produce a final response
    async for event in run_attempt(
        system_prompt=system_prompt,
        messages=messages,
        tools=tools,
        provider=provider,
        model=settings.fast_model,
    ):
        # We only care about collecting the messages; events are ignored
        if isinstance(event, ResponseDelta):
            pass

    # Extract cards from tool results
    lang = _normalize_language(language)
    cards = _extract_cards(messages, lang)

    elapsed = round(time.monotonic() - start, 2)
    logger.info("Content agent completed in %.2fs (%d cards)", elapsed, len(cards))

    return SubAgentResult(
        status="success" if cards else "partial",
        cards=cards,
        metadata={"agent": "content", "model": settings.fast_model, "duration_s": elapsed},
    )
