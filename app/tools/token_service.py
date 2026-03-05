"""Shared token service for internal platform APIs.

Fetches auth token from DynamoDB (MLServiceToken table), checks expiry,
falls back to ML_TOKEN_GENERATE_ENDPOINT if missing/expired.
Caches token in-memory to avoid repeated DynamoDB calls.
"""

import logging
import time
from datetime import datetime, timezone

import boto3
import httpx

from app.config import settings

logger = logging.getLogger(__name__)

# In-memory cache: (token_string, expiry_epoch_seconds)
_cached_token: tuple[str, float] | None = None


def _is_expired(expires_at_iso: str) -> bool:
    """Check if an ISO 8601 expiry string is in the past."""
    try:
        expiry = datetime.fromisoformat(expires_at_iso.replace("Z", "+00:00"))
        return datetime.now(timezone.utc) >= expiry
    except (ValueError, AttributeError):
        return True


async def _fetch_from_dynamo() -> str | None:
    """Scan MLServiceToken table for the first token. Returns None if missing/expired."""
    try:
        dynamodb = boto3.client(
            "dynamodb",
            region_name=settings.platform_token_dynamo_region,
        )
        response = dynamodb.scan(
            TableName=settings.platform_token_dynamo_table,
            Limit=1,
        )
        items = response.get("Items", [])
        if not items:
            return None

        token_item = items[0]
        token = token_item.get("token", {}).get("S")
        expires_at = token_item.get("expiresAt", {}).get("S")

        if not token:
            return None
        if expires_at and _is_expired(expires_at):
            logger.info("DynamoDB token expired at %s", expires_at)
            return None

        return token
    except Exception:
        logger.exception("Failed to fetch token from DynamoDB")
        return None


async def _generate_new_token() -> str | None:
    """Generate a new token via the ML token endpoint."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                settings.platform_token_endpoint,
                headers={"token": settings.platform_token_auth},
            )
            response.raise_for_status()
            data = response.json()
            if data.get("success") and data.get("data", {}).get("token"):
                return data["data"]["token"]
            return None
    except Exception:
        logger.exception("Failed to generate new token")
        return None


async def get_token() -> str:
    """Get a valid auth token, using cache → DynamoDB → endpoint fallback.

    Raises RuntimeError if no token can be obtained.
    """
    global _cached_token

    # Check in-memory cache first
    if _cached_token is not None:
        token, expiry = _cached_token
        if time.time() < expiry:
            return token

    # Try DynamoDB
    token = await _fetch_from_dynamo()
    if token:
        # Cache for 50 minutes (tokens typically last ~1 hour)
        _cached_token = (token, time.time() + 3000)
        return token

    # Fallback: generate new token
    token = await _generate_new_token()
    if token:
        _cached_token = (token, time.time() + 3000)
        return token

    raise RuntimeError("Unable to obtain auth token from DynamoDB or endpoint")
