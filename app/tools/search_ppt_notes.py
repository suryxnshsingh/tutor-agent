import csv
import re
from pathlib import Path

from app.llm.base import ToolDefinition
from app.tools.registry import register_tool

_COURSES_DIR = Path(__file__).parent.parent / "data" / "courses"
_ppt_cache: dict[str, list[dict]] = {}
_lecture_cache: dict[str, list[dict]] = {}

_DROP_COLUMNS = {"Exam_Type"}


def _normalize_class(raw: str) -> str:
    """Extract digits from class string, e.g. 'Class 12th' -> '12'."""
    match = re.search(r"\d+", raw)
    return match.group() if match else raw


def _load_ppt_notes(course_id: str) -> list[dict]:
    """Load and cache chapter-level PPT notes from ppt_notes.csv."""
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


def _load_lectures(course_id: str) -> list[dict]:
    """Load and cache topic-level lecture data from lecture_data.csv."""
    if course_id in _lecture_cache:
        return _lecture_cache[course_id]

    csv_path = _COURSES_DIR / course_id / "lecture_data.csv"
    if not csv_path.exists():
        _lecture_cache[course_id] = []
        return []

    rows: list[dict] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["Class"] = _normalize_class(row.get("Class", ""))
            row["Subject"] = (row.get("Subject") or "").strip().title()
            rows.append(row)

    _lecture_cache[course_id] = rows
    return rows


async def search_ppt_notes(
    subject: str,
    course_id: str,
    chapter_name: str | None = None,
    class_: str | None = None,
    topic: str | None = None,
) -> dict:
    """Search PPT notes at three granularity levels.

    - subject only → subject-level: lists all available chapters with notes
    - subject + chapter → chapter-level: returns the chapter's full PPT notes PDF
    - subject + chapter + topic → topic-level: returns topic-specific PPT PDFs from lecture data
    """
    notes = _load_ppt_notes(course_id)
    if not notes:
        return {"notes": [], "error": "No PPT notes data for this course"}

    subject_lower = subject.strip().lower()

    # ── Subject-level: no chapter specified ──
    if not chapter_name:
        subject_rows = [
            r for r in notes
            if r["Subject"].lower() == subject_lower
            and (not class_ or r["Class"] == class_)
        ]
        if not subject_rows:
            return {"error": f"No PPT notes found for {subject}"}

        chapters = []
        for row in subject_rows:
            chapters.append({
                "chapter_name": row.get("Chapter_Name", ""),
                "chapter_number": row.get("Chapter_Number", ""),
                "chapter_code": row.get("Chapter_Code", ""),
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

    # ── Topic-level: chapter + topic specified ──
    if topic:
        lectures = _load_lectures(course_id)
        topic_lower = topic.strip().lower()

        chapter_topics = [
            r for r in lectures
            if r["Subject"].lower() == subject_lower
            and r["Chapter_Name"].strip().lower() == chapter_lower
            and (not class_ or r["Class"] == class_)
            and (r.get("PPT Notes Link Microlecture wise") or "").strip()
        ]

        if not chapter_topics:
            return {"error": f"No topic-level PPT notes found for chapter '{chapter_name}'"}

        matched = [
            r for r in chapter_topics
            if topic_lower in (r.get("Microlecture_Name") or "").lower()
            or topic_lower in (r.get("final_microlecture_title") or "").lower()
        ]

        def _build_topic(row: dict) -> dict:
            return {
                "microlecture_name": row.get("Microlecture_Name", ""),
                "microlecture_code": row.get("Microlecture_Code", ""),
                "hindi_title": row.get("final_microlecture_title", ""),
                "chapter_name": row.get("Chapter_Name", ""),
                "chapter_code": row.get("Chapter_Code", ""),
                "language": row.get("language", ""),
                "subscription_type": row.get("Subscription_Type", ""),
                "ppt_notes_link": row.get("PPT Notes Link Microlecture wise", ""),
            }

        if matched:
            return {
                "type": "topic",
                "notes": [_build_topic(r) for r in matched],
            }

        return {
            "type": "topic",
            "notes": [_build_topic(r) for r in chapter_topics],
            "note": f"No exact match for topic '{topic}', showing all topic notes for this chapter.",
        }

    # ── Chapter-level: chapter specified, no topic ──
    chapter_rows = [
        r for r in notes
        if r["Subject"].lower() == subject_lower
        and r["Chapter_Name"].strip().lower() == chapter_lower
        and (not class_ or r["Class"] == class_)
    ]

    if not chapter_rows:
        return {"notes": [], "error": f"No PPT notes found for chapter '{chapter_name}'"}

    results = []
    for row in chapter_rows:
        results.append({
            "chapter_code": row.get("Chapter_Code", ""),
            "chapter_name": row.get("Chapter_Name", ""),
            "chapter_number": row.get("Chapter_Number", ""),
            "teacher_name": row.get("Teacher_Name", ""),
            "language": row.get("language", ""),
            "subscription_type": row.get("Subscription_Type", ""),
            "ppt_notes_link": row.get("PPT_Notes_Link", ""),
        })

    return {
        "type": "chapter",
        "notes": results,
    }


_definition = ToolDefinition(
    name="search_ppt_notes",
    description=(
        "Search for PPT/presentation notes at three levels: "
        "1) Subject-level (just subject) → lists all chapters with PPT notes available. "
        "2) Chapter-level (subject + chapter_name) → returns the chapter's full PPT notes PDF. "
        "3) Topic-level (subject + chapter_name + topic) → returns topic-specific PPT notes PDFs. "
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

register_tool(_definition, search_ppt_notes)
