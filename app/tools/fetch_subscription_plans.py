"""Tool: fetch_subscription_plans — retrieves subscription plan details for a student."""

import logging

import httpx

from app.config import settings
from app.llm.base import ToolDefinition
from app.tools.registry import register_tool
from app.tools.token_service import get_token

logger = logging.getLogger(__name__)


async def fetch_subscription_plans(user_id: str, language: str = "english") -> dict:
    """Fetch subscription plans from the platform API."""
    token = await get_token()

    headers = {
        "accept": "*/*",
        "Accept-Language": language,
        "token": token,
        "userId": user_id,
    }
    params = {
        "planLevel": "COURSE",
        "boardSelected": "true",
        "comboSelected": "false",
    }

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            settings.platform_subscription_plan_url,
            headers=headers,
            params=params,
        )
        response.raise_for_status()
        data = response.json()

    return {
        "title": data.get("title"),
        "subscriptionType": data.get("subscriptionType"),
        "features": data.get("features"),
        "plans": [
            {"name": plan.get("name"), "finalPrice": plan.get("finalPrice")}
            for plan in data.get("plans", [])
        ],
    }


_definition = ToolDefinition(
    name="fetch_subscription_plans",
    description=(
        "Fetches available subscription plans for a student. "
        "Call this when the student asks about subscription plans, pricing, "
        "or wants to know what plans are available. "
        "Returns plan titles, types, features, and prices."
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

register_tool(_definition, fetch_subscription_plans)
