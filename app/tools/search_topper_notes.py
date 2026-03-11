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
    class_: str | None = None,
    topic: str | None = None,
) -> dict:
    """Search topper notes at three granularity levels.

    - subject only → subject-level: lists all chapters with topper notes available
    - subject + chapter → chapter-level: returns the chapter's full topper notes PDF
    - subject + chapter + topic → topic-level: returns the chapter's topper notes PDF
      (toppers notes are per-chapter, so topic resolves to the chapter PDF)
    """
    notes = _load_notes(course_id)
    if not notes:
        return {"notes": [], "error": "No topper notes data for this course"}

    subject_lower = subject.strip().lower()

    # ── Subject-level: no chapter specified ──
    if not chapter_name:
        subject_rows = [
            r for r in notes
            if r["Subject"].lower() == subject_lower
            and (not class_ or r["Class"] == class_)
        ]
        if not subject_rows:
            return {"error": f"No topper notes found for {subject}"}

        chapters = []
        for row in subject_rows:
            chapters.append({
                "chapter_name": row.get("Chapter_Name", ""),
                "chapter_number": row.get("Chapter_Number", ""),
                "notes_id": row.get("notes_id", ""),
                "language": row.get("language", ""),
            })

        return {
            "type": "subject",
            "subject": subject,
            "course_id": course_id,
            "total_chapters": len(chapters),
            "chapters": chapters,
        }

    chapter_lower = chapter_name.strip().lower()

    # Filter to chapter + subject + optional class
    chapter_rows = [
        r for r in notes
        if r["Subject"].lower() == subject_lower
        and r["Chapter_Name"].strip().lower() == chapter_lower
        and (not class_ or r["Class"] == class_)
    ]

    if not chapter_rows:
        return {"notes": [], "error": f"No topper notes found for chapter '{chapter_name}'"}

    # Build chapter-level results (used for both chapter and topic levels)
    results = []
    for row in chapter_rows:
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

    # ── Topic-level: chapter + topic specified ──
    # Toppers notes are per-chapter PDFs, so topic resolves to the same chapter PDF.
    if topic:
        return {
            "type": "topic",
            "topic": topic,
            "notes": results,
            "note": f"Toppers notes are chapter-level PDFs. Showing notes for chapter '{chapter_name}' which covers '{topic}'.",
        }

    # ── Chapter-level: chapter specified, no topic ──
    return {
        "type": "chapter",
        "notes": results,
    }


_definition = ToolDefinition(
    name="search_topper_notes",
    description=(
        "Search for topper notes at three levels: "
        "1) Subject-level (just subject) → lists all chapters with topper notes available. "
        "2) Chapter-level (subject + chapter_name) → returns the chapter's full topper notes PDF. "
        "3) Topic-level (subject + chapter_name + topic) → returns the chapter's topper notes PDF "
        "(toppers notes are per-chapter, so topic resolves to the relevant chapter). "
        "Use resolve_chapter first when the student mentions a chapter or topic name."
    ),
    parameters={
        "type": "object",
        "properties": {
            "subject": {
                "type": "string",
                "description": "The subject (Physics, Chemistry, Biology, etc.)",
            },
            "course_id": {
                "type": "string",
                "description": "The course ID (e.g., '4' for MPBSE)",
            },
            "chapter_name": {
                "type": "string",
                "description": (
                    "The English chapter name from resolve_chapter. "
                    "Omit for subject-level results."
                ),
            },
            "class_": {
                "type": "string",
                "description": (
                    "The student's class number (e.g., '12'). "
                    "Filters to that class. Omit to search all."
                ),
            },
            "topic": {
                "type": "string",
                "description": (
                    "A specific topic within a chapter (e.g. 'Coulomb\\'s Law', 'Fermentation'). "
                    "Only pass when the student asks for notes on a specific topic, not the whole chapter."
                ),
            },
        },
        "required": ["subject", "course_id"],
    },
    required_params=["subject", "course_id"],
)

register_tool(_definition, search_topper_notes)
