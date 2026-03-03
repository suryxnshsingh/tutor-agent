import csv
import re
from pathlib import Path

from app.llm.base import ToolDefinition
from app.tools.registry import register_tool

_COURSES_DIR = Path(__file__).parent.parent / "data" / "courses"
_ppt_cache: dict[str, list[dict]] = {}

_DROP_COLUMNS = {"Exam_Type"}


def _normalize_class(raw: str) -> str:
    """Extract digits from class string, e.g. 'Class 12th' -> '12'."""
    match = re.search(r"\d+", raw)
    return match.group() if match else raw


def _load_ppt_notes(course_id: str) -> list[dict]:
    """Load and cache PPT notes data for a given course_id."""
    if course_id in _ppt_cache:
        return _ppt_cache[course_id]

    csv_path = _COURSES_DIR / course_id / "ppt_notes.csv"
    if not csv_path.exists():
        _ppt_cache[course_id] = []
        return []

    rows: list[dict] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for col in _DROP_COLUMNS:
                row.pop(col, None)
            row["Class"] = _normalize_class(row.get("Class", ""))
            row["Subject"] = (row.get("Subject") or "").strip().title()
            rows.append(row)

    _ppt_cache[course_id] = rows
    return rows


async def search_ppt_notes(
    subject: str,
    course_id: str,
    chapter_name: str | None = None,
    class_: str | None = None,
) -> dict:
    """Find PPT notes for a given chapter, subject, and course."""
    notes = _load_ppt_notes(course_id)
    if not notes:
        return {"notes": [], "error": "No PPT notes data for this course"}

    subject_lower = subject.strip().lower()
    chapter_lower = chapter_name.strip().lower() if chapter_name else None

    results = []
    for row in notes:
        if row["Subject"].lower() != subject_lower:
            continue
        if chapter_lower and row["Chapter_Name"].strip().lower() != chapter_lower:
            continue
        if class_ and row["Class"] != class_:
            continue

        results.append({
            "chapter_code": row.get("Chapter_Code", ""),
            "chapter_name": row.get("Chapter_Name", ""),
            "chapter_number": row.get("Chapter_Number", ""),
            "teacher_name": row.get("Teacher_Name", ""),
            "language": row.get("language", ""),
            "subscription_type": row.get("Subscription_Type", ""),
            "ppt_notes_link": row.get("PPT_Notes_Link", ""),
        })

    return {"notes": results}


_definition = ToolDefinition(
    name="search_ppt_notes",
    description=(
        "Returns PPT/presentation notes PDFs for a chapter. Use AFTER "
        "resolve_chapter to get notes for a specific chapter. When a "
        "student asks for 'notes', call both search_topper_notes and "
        "search_ppt_notes to give them all available material."
    ),
    parameters={
        "type": "object",
        "properties": {
            "subject": {
                "type": "string",
                "description": "The subject (Physics, Chemistry, etc.)",
            },
            "course_id": {
                "type": "string",
                "description": "The course ID (e.g., '4' for MPBSE)",
            },
            "chapter_name": {
                "type": "string",
                "description": (
                    "The English chapter name from resolve_chapter "
                    "results (e.g. 'Human Reproduction')"
                ),
            },
            "class_": {
                "type": "string",
                "description": (
                    "The student's class number (e.g., '12'). "
                    "Filters to that class. Omit to search all."
                ),
            },
        },
        "required": ["subject", "course_id"],
    },
    required_params=["subject", "course_id"],
)

register_tool(_definition, search_ppt_notes)
