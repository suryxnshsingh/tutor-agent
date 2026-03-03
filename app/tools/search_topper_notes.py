import csv
import re
from pathlib import Path

from app.llm.base import ToolDefinition
from app.tools.registry import register_tool

_COURSES_DIR = Path(__file__).parent.parent / "data" / "courses"
_notes_cache: dict[str, list[dict]] = {}

_DROP_COLUMNS = {"Exam_Type", "subjectId"}


def _normalize_class(raw: str) -> str:
    """Extract digits from class string, e.g. 'Class 12th' -> '12'."""
    match = re.search(r"\d+", raw)
    return match.group() if match else raw


def _load_notes(course_id: str) -> list[dict]:
    """Load and cache topper notes data for a given course_id."""
    if course_id in _notes_cache:
        return _notes_cache[course_id]

    csv_path = _COURSES_DIR / course_id / "toppers_notes.csv"
    if not csv_path.exists():
        _notes_cache[course_id] = []
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

    _notes_cache[course_id] = rows
    return rows


async def search_topper_notes(
    subject: str,
    course_id: str,
    chapter_name: str | None = None,
    toppers_name: str | None = None,
    class_: str | None = None,
) -> dict:
    """Find topper notes by chapter and/or topper name."""
    notes = _load_notes(course_id)
    if not notes:
        return {"notes": [], "error": "No topper notes data for this course"}

    subject_lower = subject.strip().lower()
    chapter_lower = chapter_name.strip().lower() if chapter_name else None
    toppers_lower = toppers_name.strip().lower() if toppers_name else None

    results = []
    for row in notes:
        if row["Subject"].lower() != subject_lower:
            continue
        if chapter_lower and row["Chapter_Name"].strip().lower() != chapter_lower:
            continue
        if toppers_lower and row.get("Toppers_Name", "").strip().lower() != toppers_lower:
            continue
        if class_ and row["Class"] != class_:
            continue

        results.append({
            "notes_id": row.get("notes_id", ""),
            "chapter_name": row.get("Chapter_Name", ""),
            "chapter_number": row.get("Chapter_Number", ""),
            "title": row.get("final_title", ""),
            "toppers_name": row.get("Toppers_Name", ""),
            "language": row.get("language", ""),
            "subscription_type": row.get("Subscription_Type", ""),
            "notes_link": row.get("Toppers_Notes_link", ""),
        })

    return {"notes": results}


_definition = ToolDefinition(
    name="search_topper_notes",
    description=(
        "Returns topper notes PDFs filtered by chapter and/or topper "
        "name. Use AFTER resolve_chapter to get notes for a chapter, "
        "or pass toppers_name to find all notes by a specific topper. "
        "At least one of chapter_name or toppers_name must be provided."
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
                    "results (e.g. 'Relations and Functions')"
                ),
            },
            "toppers_name": {
                "type": "string",
                "description": (
                    "Filter by topper name (e.g. 'Priyal Dwivedi'). "
                    "Returns all notes by this topper."
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

register_tool(_definition, search_topper_notes)
