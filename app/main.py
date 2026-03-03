from contextlib import asynccontextmanager

from fastapi import FastAPI

import app.tools.resolve_chapter  # noqa: F401
from app.agent.runner import init_session_manager
from app.gateway.routes import router
from app.session.dynamo_store import DynamoStore
from app.session.manager import SessionManager
from app.session.redis_store import RedisStore

_redis_store = RedisStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await _redis_store.connect()
    dynamo_store = DynamoStore()
    session_manager = SessionManager(redis_store=_redis_store, dynamo_store=dynamo_store)
    init_session_manager(session_manager)
    yield
    # Shutdown
    await _redis_store.close()


app = FastAPI(title="Arivihan Agent", version="0.1.0", lifespan=lifespan)
app.include_router(router)


@app.get("/health")
async def health():
    return {"status": "ok"}
