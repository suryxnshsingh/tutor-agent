import re
from pathlib import Path

from app.config import settings

_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "system_prompt.txt"
_PROMPT_TEMPLATE = _PROMPT_PATH.read_text()
_COURSE_SNIPPETS_DIR = Path(__file__).parent.parent.parent / "prompts" / "courses"


def build_system_prompt(
    course_id: str,
    class_: str,
    subject: str,
    language: str,
    agent_name: str | None = None,
) -> str:
    """Build the full system prompt with student context filled in."""
    name = agent_name or settings.agent_name
    # Use manual replacement to avoid issues with curly braces in prompt
    prompt = _PROMPT_TEMPLATE
    prompt = prompt.replace("{agent_name}", name)
    prompt = prompt.replace("{course}", course_id)
    prompt = prompt.replace("{class}", class_)
    prompt = prompt.replace("{subject}", subject)
    prompt = prompt.replace("{language}", language)

    # Append course-specific notes if a snippet exists
    snippet_path = _COURSE_SNIPPETS_DIR / f"{course_id}.txt"
    if snippet_path.exists():
        course_notes = snippet_path.read_text().strip()
        prompt += f"\n\n## Course-Specific Notes\n\n{course_notes}"

    return prompt


def strip_thinking(response_text: str) -> str:
    """Remove <thinking> blocks from agent response."""
    return re.sub(r"<thinking>.*?</thinking>", "", response_text, flags=re.DOTALL).strip()


def extract_thinking(response_text: str) -> str | None:
    """Extract thinking content for analytics/logging."""
    match = re.search(r"<thinking>(.*?)</thinking>", response_text, flags=re.DOTALL)
    return match.group(1).strip() if match else None
