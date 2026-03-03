import csv
import re
from pathlib import Path

from app.llm.base import ToolDefinition
from app.tools.registry import register_tool

_COURSES_DIR = Path(__file__).parent.parent / "data" / "courses"
_ncert_cache: dict[str, list[dict]] = {}

_DROP_COLUMNS = {"Exam_Type"}


def _normalize_class(raw: str) -> str:
    """Extract digits from class string, e.g. 'Class 12th' -> '12'."""
    match = re.search(r"\d+", raw)
    return match.group() if match else raw


def _load_ncert_solutions(course_id: str) -> list[dict]:
    """Load and cache NCERT solutions data for a given course_id."""
    if course_id in _ncert_cache:
        return _ncert_cache[course_id]

    csv_path = _COURSES_DIR / course_id / "ncert_solutions.csv"
    if not csv_path.exists():
        _ncert_cache[course_id] = []
        return []

    rows: list[dict] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for col in _DROP_COLUMNS:
                row.pop(col, None)
            row["Class"] = _normalize_class(row.get("Class", ""))
            row["Subject"] = (row.get("Subject") or "").strip().title()
            try:
                row["Chapter_Number"] = int(
                    row.get("Chapter_Number", 0) or 0
                )
            except (ValueError, TypeError):
                row["Chapter_Number"] = 0
            rows.append(row)

    _ncert_cache[course_id] = rows
    return rows


async def search_ncert_solutions(
    subject: str,
    course_id: str,
    chapter_name: str | None = None,
    chapter_number: int | None = None,
    class_: str | None = None,
) -> dict:
    """Find NCERT solutions by subject and optional chapter."""
    data = _load_ncert_solutions(course_id)
    if not data:
        return {
            "solutions": [],
            "error": "No NCERT solutions data for this course",
        }

    subject_lower = subject.strip().lower()
    # Match against both Chapter_Name and title (English) for flexibility
    chapter_lower = chapter_name.strip().lower() if chapter_name else None

    results = []
    for row in data:
        if row["Subject"].lower() != subject_lower:
            continue
        if class_ and row["Class"] != class_:
            continue
        if chapter_number and row["Chapter_Number"] != chapter_number:
            continue
        if chapter_lower:
            name_match = row["Chapter_Name"].strip().lower() == chapter_lower
            title_match = (
                row.get("title", "").strip().lower() == chapter_lower
            )
            if not name_match and not title_match:
                continue

        results.append({
            "chapter_name": row.get("Chapter_Name", ""),
            "title": row.get("title", ""),
            "chapter_number": row["Chapter_Number"],
            "language": row.get("language", ""),
            "subscription_type": row.get("Subscription_Type", ""),
            "solution_link": row.get("solution_link", ""),
        })

    results.sort(key=lambda x: x["chapter_number"])
    return {"solutions": results}


_definition = ToolDefinition(
    name="search_ncert_solutions",
    description=(
        "Returns NCERT solution PDFs for a subject, optionally "
        "filtered by chapter. Use when the student asks for NCERT "
        "solutions, textbook solutions, or exercise solutions. "
        "Currently available for Maths."
    ),
    parameters={
        "type": "object",
        "properties": {
            "subject": {
                "type": "string",
                "description": "The subject (currently Maths)",
            },
            "course_id": {
                "type": "string",
                "description": "The course ID (e.g., '4' for MPBSE)",
            },
            "chapter_name": {
                "type": "string",
                "description": (
                    "Chapter name (English). Matches against both "
                    "chapter name and title fields."
                ),
            },
            "chapter_number": {
                "type": "integer",
                "description": "Chapter number to filter by.",
            },
            "class_": {
                "type": "string",
                "description": (
                    "Class number (e.g., '12'). "
                    "Omit to search all classes."
                ),
            },
        },
        "required": ["subject", "course_id"],
    },
    required_params=["subject", "course_id"],
)

register_tool(_definition, search_ncert_solutions)
