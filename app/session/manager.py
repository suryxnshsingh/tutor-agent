from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from app.messages.models import ChatSession
from app.session.dynamo_store import DynamoStore
from app.session.redis_store import RedisStore

logger = logging.getLogger(__name__)


class SessionManager:
    def __init__(self, redis_store: RedisStore, dynamo_store: DynamoStore) -> None:
        self.redis = redis_store
        self.dynamo = dynamo_store

    async def load(
        self, chat_id: str, user_id: str, subject: str, course_id: str
    ) -> ChatSession:
        """Load session: try Redis → fallback DynamoDB → create new."""
        # 1. Try Redis
        session = await self.redis.get(chat_id)
        if session is not None:
            logger.debug("Session loaded from Redis: %s", chat_id)
            return session

        # 2. Fallback to DynamoDB
        try:
            session = await self.dynamo.get(chat_id)
        except Exception:
            logger.exception("DynamoDB read failed for chat_id=%s", chat_id)
            session = None

        if session is not None:
            logger.debug("Session loaded from DynamoDB, populating Redis: %s", chat_id)
            await self.redis.save(session)
            return session

        # 3. Create new session
        logger.debug("Creating new session: %s", chat_id)
        return ChatSession(
            chat_id=chat_id,
            user_id=user_id,
            subject=subject,
            course_id=course_id,
        )

    async def save(self, session: ChatSession) -> None:
        """Save session: write Redis (awaited), fire-and-forget DynamoDB."""
        session.updated_at = datetime.utcnow()

        # Redis — synchronous write (awaited)
        await self.redis.save(session)

        # DynamoDB — fire-and-forget
        asyncio.create_task(self._save_dynamo(session))

    async def _save_dynamo(self, session: ChatSession) -> None:
        try:
            await self.dynamo.save(session)
        except Exception:
            logger.exception("DynamoDB write failed for chat_id=%s", session.chat_id)
