import re
from collections.abc import AsyncGenerator
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


async def filter_thinking_stream(
    token_stream: AsyncGenerator[str, None],
) -> AsyncGenerator[str, None]:
    """Filter <thinking>...</thinking> blocks from a token stream.

    Buffers tokens while inside a thinking block. Yields clean content only.
    Handles the case where tags are split across multiple tokens.
    """
    buffer = ""
    inside_thinking = False

    async for token in token_stream:
        buffer += token

        while buffer:
            if inside_thinking:
                # Look for closing tag
                end_idx = buffer.find("</thinking>")
                if end_idx != -1:
                    # Skip everything up to and including </thinking>
                    buffer = buffer[end_idx + len("</thinking>"):]
                    inside_thinking = False
                    continue
                # Might have a partial closing tag at the end — keep buffering
                if "</thinking>"[:1] in buffer[-len("</thinking>"):]:
                    break
                # No closing tag fragment — discard and keep buffering
                buffer = ""
                break
            else:
                # Look for opening tag
                start_idx = buffer.find("<thinking>")
                if start_idx != -1:
                    # Yield everything before the tag
                    before = buffer[:start_idx]
                    if before:
                        yield before
                    buffer = buffer[start_idx + len("<thinking>"):]
                    inside_thinking = True
                    continue
                # Check for partial opening tag at end of buffer
                # e.g. buffer ends with "<thin" — could be start of <thinking>
                partial = False
                tag = "<thinking>"
                for i in range(1, len(tag)):
                    if buffer.endswith(tag[:i]):
                        # Yield everything before the partial match
                        safe = buffer[: -i]
                        if safe:
                            yield safe
                        buffer = buffer[-i:]
                        partial = True
                        break
                if partial:
                    break
                # No tag or partial — yield entire buffer
                yield buffer
                buffer = ""
                break

    # Flush remaining buffer (if not inside thinking)
    if buffer and not inside_thinking:
        yield buffer
