"""Tool: fetch_coupons — retrieves the best active coupon for a student."""

import logging
from datetime import datetime, timezone

import httpx

from app.config import settings
from app.llm.base import ToolDefinition
from app.tools.registry import register_tool
from app.tools.token_service import get_token

logger = logging.getLogger(__name__)


def _epoch_ms_to_date(epoch_ms: int | float) -> str:
    """Convert epoch milliseconds to human-readable UTC datetime string."""
    return datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _get_max_discount_coupon(coupons: list[dict]) -> dict | None:
    """Return the coupon with the highest maxDiscountAmount."""
    best = None
    best_value = 0
    for coupon in coupons:
        discount = coupon.get("maxDiscountAmount", 0)
        if discount > best_value:
            best_value = discount
            best = coupon
    return best


async def fetch_coupons(user_id: str, language: str = "english") -> dict:
    """Fetch the best active coupon for a user from the platform API."""
    token = await get_token()

    headers = {
        "accept": "*/*",
        "Accept-Language": language.upper(),
        "token": token,
        "userId": user_id,
    }

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(settings.platform_coupon_url, headers=headers)
        response.raise_for_status()
        data = response.json()

    coupons = data.get("data", [])
    if not coupons:
        return {"coupon": None, "message": "No active coupons found for this user"}

    best = _get_max_discount_coupon(coupons)
    if not best:
        return {"coupon": None, "message": "No active coupons found for this user"}

    # Convert epoch timestamps to readable dates
    for field in ("validFrom", "validTo", "modifiedAt"):
        if field in best and isinstance(best[field], (int, float)):
            best[field] = _epoch_ms_to_date(best[field])

    return {"coupon": best}


_definition = ToolDefinition(
    name="fetch_coupons",
    description=(
        "Fetches the best active discount coupon for a student. "
        "Call this when the student asks about coupons, discounts, or offers. "
        "Returns the coupon with the highest discount amount including its code, "
        "validity dates, and discount details."
    ),
    parameters={
        "type": "object",
        "properties": {
            "user_id": {
                "type": "string",
                "description": "The student's user ID (available in student context)",
            },
            "language": {
                "type": "string",
                "description": "Language: 'english' or 'hindi'. Defaults to 'english'.",
                "enum": ["english", "hindi"],
            },
        },
        "required": ["user_id"],
    },
    required_params=["user_id"],
)

register_tool(_definition, fetch_coupons)
