from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SubAgentResult:
    """Standardized result returned by every subagent."""

    status: str  # "success" | "partial" | "error"
    text: str | None = None
    cards: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class SubAgentDispatch:
    """Instructions for a single subagent, produced by the teacher."""

    agent: str  # "doubt" | "content" | "guidance"
    input: str  # English context for the subagent
    language: str = "hinglish"
    nudge: str | None = None
    content_types: list[str] = field(default_factory=list)
    chapter_id: int | None = None
    chapter_name: str | None = None


@dataclass
class TeacherDecision:
    """Structured routing decision from the teacher LLM call."""

    type: str  # "dispatch" | "direct_response"
    direct_response: str | None = None
    chapter_id: int | None = None
    chapter_name: str | None = None
    subagents: list[SubAgentDispatch] = field(default_factory=list)
    follow_up: str | None = None
