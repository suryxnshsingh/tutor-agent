import asyncio
import logging
import random

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)

from app.llm.base import LLMProvider, LLMResponse, ToolDefinition
from app.messages.models import InternalMessage

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
BASE_DELAY = 1.0
MAX_DELAY = 10.0


class LLMCallFailedError(Exception):
    """Raised when all retry attempts (primary + fallback) are exhausted."""


def _should_retry(exc: Exception) -> bool:
    """Classify exception: return True if retryable, False if not."""
    if isinstance(exc, AuthenticationError):
        return False
    if isinstance(exc, (RateLimitError, APIConnectionError, APIStatusError)):
        return True
    if isinstance(exc, APITimeoutError):
        return True
    return False


def _backoff_delay(attempt: int) -> float:
    """Exponential backoff with jitter."""
    delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
    return delay + random.uniform(0, delay * 0.5)


async def _try_model(
    provider: LLMProvider,
    system_prompt: str,
    messages: list[InternalMessage],
    tools: list[ToolDefinition],
    model: str,
) -> LLMResponse:
    """Try a single model with up to MAX_ATTEMPTS retries."""
    last_exc: Exception | None = None

    for attempt in range(MAX_ATTEMPTS):
        try:
            return await provider.chat(
                system_prompt=system_prompt,
                messages=messages,
                tools=tools,
                model=model,
            )
        except Exception as exc:
            last_exc = exc
            if not _should_retry(exc):
                logger.warning("Non-retryable error on model %s: %s", model, exc)
                raise
            if isinstance(exc, APITimeoutError) and attempt >= 1:
                logger.warning("Timeout retry exhausted for model %s", model)
                break
            delay = _backoff_delay(attempt)
            logger.warning(
                "Retryable error on model %s (attempt %d/%d), retrying in %.1fs: %s",
                model, attempt + 1, MAX_ATTEMPTS, delay, exc,
            )
            await asyncio.sleep(delay)

    raise last_exc  # type: ignore[misc]


async def chat_with_retry(
    provider: LLMProvider,
    system_prompt: str,
    messages: list[InternalMessage],
    tools: list[ToolDefinition],
    primary_model: str,
    fallback_model: str | None = None,
) -> LLMResponse:
    """Call LLM with retry + failover to fallback model."""
    try:
        return await _try_model(provider, system_prompt, messages, tools, primary_model)
    except Exception as primary_exc:
        if not fallback_model or fallback_model == primary_model:
            raise LLMCallFailedError(
                f"Primary model {primary_model} failed: {primary_exc}"
            ) from primary_exc

        logger.warning(
            "Primary model %s exhausted, falling back to %s",
            primary_model, fallback_model,
        )
        try:
            return await _try_model(provider, system_prompt, messages, tools, fallback_model)
        except Exception as fallback_exc:
            raise LLMCallFailedError(
                f"Both models failed. Primary ({primary_model}): {primary_exc}, "
                f"Fallback ({fallback_model}): {fallback_exc}"
            ) from fallback_exc
