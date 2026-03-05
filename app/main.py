import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

import app.tools.fetch_coupons  # noqa: F401
import app.tools.fetch_subscription_plans  # noqa: F401
import app.tools.resolve_chapter  # noqa: F401
import app.tools.search_important_questions  # noqa: F401
import app.tools.search_lectures  # noqa: F401
import app.tools.search_ncert_solutions  # noqa: F401
import app.tools.search_ppt_notes  # noqa: F401
import app.tools.search_pyq_papers  # noqa: F401
import app.tools.search_tests  # noqa: F401
import app.tools.search_topper_notes  # noqa: F401
from app.agent.runner import init_chat_lock, init_session_manager
from app.gateway.routes import router
from app.session.dynamo_store import DynamoStore
from app.session.lock import ChatLock
from app.session.manager import SessionManager
from app.session.redis_store import RedisStore

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

_redis_store = RedisStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await _redis_store.connect()
    dynamo_store = DynamoStore()
    session_manager = SessionManager(redis_store=_redis_store, dynamo_store=dynamo_store)
    init_session_manager(session_manager)
    chat_lock = ChatLock(redis_client=_redis_store.client)
    init_chat_lock(chat_lock)
    yield
    # Shutdown
    await _redis_store.close()


app = FastAPI(title="Arivihan Agent", version="0.1.0", lifespan=lifespan)
app.include_router(router)


@app.get("/health")
async def health():
    return {"status": "ok"}
