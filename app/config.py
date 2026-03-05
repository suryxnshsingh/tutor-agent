import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


class Settings:
    """Application settings loaded from environment variables."""

    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    primary_model: str = os.getenv("PRIMARY_MODEL", "gpt-5-mini")
    fallback_model: str = os.getenv("FALLBACK_MODEL", "gpt-4.1-mini")
    compaction_model: str = os.getenv("COMPACTION_MODEL", "gpt-5")

    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    redis_ttl_hours: int = int(os.getenv("REDIS_TTL_HOURS", "48"))

    dynamo_table: str = os.getenv("DYNAMO_TABLE", "arivihan-sessions")
    dynamo_endpoint: str | None = os.getenv("DYNAMO_ENDPOINT")
    aws_region: str = os.getenv("AWS_REGION", "us-east-1")

    compaction_threshold: int = int(os.getenv("COMPACTION_THRESHOLD", "20"))
    recent_turns_to_keep: int = int(os.getenv("RECENT_TURNS_TO_KEEP", "5"))

    agent_name: str = os.getenv("AGENT_NAME", "Arivihan Agent")

    # Platform APIs (coupon & subscription)
    platform_coupon_url: str = os.getenv("PLATFORM_COUPON_URL", "")
    platform_subscription_plan_url: str = os.getenv("PLATFORM_SUBSCRIPTION_PLAN_URL", "")
    platform_token_endpoint: str = os.getenv("PLATFORM_TOKEN_ENDPOINT", "")
    platform_token_auth: str = os.getenv("PLATFORM_TOKEN_AUTH", "")
    platform_token_dynamo_table: str = os.getenv("PLATFORM_TOKEN_DYNAMO_TABLE", "MLServiceToken")
    platform_token_dynamo_region: str = os.getenv("PLATFORM_TOKEN_DYNAMO_REGION", "ap-south-1")


settings = Settings()
