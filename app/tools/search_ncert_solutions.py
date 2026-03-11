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
    """Search NCERT solutions at two granularity levels.

    - subject only → subject-level: lists all chapters with NCERT solutions available
    - subject + chapter → chapter-level: returns the chapter's full NCERT solutions PDF

    No topic-level exists — NCERT solutions are organised at chapter level only.
    If a student asks for a specific topic's solutions, return the chapter PDF.
    """
    data = _load_ncert_solutions(course_id)
    if not data:
        return {
            "solutions": [],
            "error": "No NCERT solutions data for this course",
        }

    subject_lower = subject.strip().lower()

    # ── Subject-level: no chapter specified ──
    if not chapter_name and not chapter_number:
        subject_rows = [
            r for r in data
            if r["Subject"].lower() == subject_lower
            and (not class_ or r["Class"] == class_)
        ]
        if not subject_rows:
            return {"error": f"No NCERT solutions found for {subject}"}

        subject_rows.sort(key=lambda x: x["Chapter_Number"])

        chapters = []
        for row in subject_rows:
            chapters.append({
                "chapter_name": row.get("Chapter_Name", ""),
                "title": row.get("title", ""),
                "chapter_number": row["Chapter_Number"],
                "language": row.get("language", ""),
            })

        return {
            "type": "subject",
            "subject": subject,
            "course_id": course_id,
            "total_chapters": len(chapters),
            "chapters": chapters,
        }

    # ── Chapter-level: chapter specified ──
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

    if not results:
        return {"solutions": [], "error": f"No NCERT solutions found for chapter '{chapter_name or chapter_number}'"}

    results.sort(key=lambda x: x["chapter_number"])
    return {
        "type": "chapter",
        "solutions": results,
    }


_definition = ToolDefinition(
    name="search_ncert_solutions",
    description=(
        "Search for NCERT solutions at two levels: "
        "1) Subject-level (just subject) → lists all chapters with NCERT solutions available. "
        "2) Chapter-level (subject + chapter_name or chapter_number) → returns the chapter's full NCERT solutions PDF. "
        "No topic-level exists — if a student asks for a specific topic's NCERT solution, "
        "resolve to the chapter and return the chapter PDF. "
        "Use resolve_chapter first when the student mentions a chapter or topic name."
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
                    "chapter name and title fields. "
                    "Omit for subject-level results."
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
