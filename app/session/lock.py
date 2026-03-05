from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

import redis.asyncio as redis

logger = logging.getLogger(__name__)


class ChatLockError(Exception):
    """Raised when a chat lock cannot be acquired."""


class ChatLock:
    """Per-chat concurrency lock backed by Redis."""

    def __init__(self, redis_client: redis.Redis, ttl_seconds: int = 120) -> None:
        self.client = redis_client
        self.ttl_seconds = ttl_seconds

    def _key(self, chat_id: str) -> str:
        return f"lock:chat:{chat_id}"

    async def acquire(self, chat_id: str) -> bool:
        """Try to acquire the lock. Returns True if acquired."""
        result = await self.client.set(
            self._key(chat_id), "1", nx=True, ex=self.ttl_seconds
        )
        return result is not None

    async def release(self, chat_id: str) -> None:
        """Release the lock."""
        await self.client.delete(self._key(chat_id))

    @asynccontextmanager
    async def hold(self, chat_id: str) -> AsyncGenerator[None, None]:
        """Async context manager: acquire lock, yield, release in finally."""
        acquired = await self.acquire(chat_id)
        if not acquired:
            raise ChatLockError(f"Chat {chat_id} is already being processed")
        try:
            yield
        finally:
            await self.release(chat_id)
