from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator

from app.agent.prompt import strip_thinking
from app.agent.subagents.base import SubAgentResult
from app.config import settings
from app.llm.base import LLMProvider
from app.messages.models import InternalMessage

logger = logging.getLogger(__name__)

_PROMPTS_DIR = __import__("pathlib").Path(__file__).parent.parent.parent.parent / "prompts"
_PROMPT_PATH = _PROMPTS_DIR / "subagents" / "guidance_prompt.txt"
_PROMPT_TEMPLATE = _PROMPT_PATH.read_text()


def _build_prompt(language: str, class_: str, subject: str, course_id: str) -> str:
    prompt = _PROMPT_TEMPLATE
    prompt = prompt.replace("{language}", language)
    prompt = prompt.replace("{class}", class_)
    prompt = prompt.replace("{subject}", subject)
    prompt = prompt.replace("{course}", course_id)
    return prompt


async def run_guidance_agent(
    *,
    input_text: str,
    language: str,
    class_: str,
    subject: str,
    course_id: str,
    provider: LLMProvider,
) -> SubAgentResult:
    """Run the guidance subagent. No tools — pure LLM advice."""
    start = time.monotonic()

    system_prompt = _build_prompt(language, class_, subject, course_id)
    messages = [InternalMessage(role="student", content=input_text)]

    response = await provider.chat(
        system_prompt=system_prompt,
        messages=messages,
        tools=[],
        model=settings.primary_model,
    )

    elapsed = round(time.monotonic() - start, 2)
    text = strip_thinking(response.text or "")

    logger.info("Guidance agent completed in %.2fs (%d chars)", elapsed, len(text))

    return SubAgentResult(
        status="success" if text else "error",
        text=text or None,
        metadata={
            "agent": "guidance",
            "model": settings.primary_model,
            "duration_s": elapsed,
        },
    )


async def stream_guidance_agent(
    *,
    input_text: str,
    language: str,
    class_: str,
    subject: str,
    course_id: str,
    provider: LLMProvider,
) -> AsyncGenerator[str, None]:
    """Stream tokens from the guidance agent. Yields raw tokens (including thinking tags)."""
    system_prompt = _build_prompt(language, class_, subject, course_id)
    messages = [InternalMessage(role="student", content=input_text)]

    async for token in provider.chat_stream(
        system_prompt=system_prompt,
        messages=messages,
        model=settings.primary_model,
    ):
        yield token
