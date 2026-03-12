import re
from pathlib import Path

from app.config import settings

_PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"
_TEACHER_PROMPT_PATH = _PROMPTS_DIR / "teacher_prompt.txt"
_TEACHER_PROMPT_TEMPLATE = _TEACHER_PROMPT_PATH.read_text()
_COURSE_SNIPPETS_DIR = _PROMPTS_DIR / "courses"


def build_teacher_prompt(
    course_id: str,
    class_: str,
    subject: str,
    language: str,
    agent_name: str | None = None,
) -> str:
    """Build the teacher agent routing prompt with student context."""
    name = agent_name or settings.agent_name
    prompt = _TEACHER_PROMPT_TEMPLATE
    prompt = prompt.replace("{agent_name}", name)
    prompt = prompt.replace("{course}", course_id)
    prompt = prompt.replace("{class}", class_)
    prompt = prompt.replace("{subject}", subject)
    prompt = prompt.replace("{language}", language)

    # Append course-specific notes if available
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
