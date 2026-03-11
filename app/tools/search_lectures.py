import csv
import re
from pathlib import Path

from app.llm.base import ToolDefinition
from app.tools.registry import register_tool

_COURSES_DIR = Path(__file__).parent.parent / "data" / "courses"
_lecture_cache: dict[str, list[dict]] = {}

_DROP_COLUMNS = {"board", "completed", "class_id", "Exam_Type"}


def _normalize_class(raw: str) -> str:
    """Extract digits from class string, e.g. 'Class 12th' -> '12'."""
    match = re.search(r"\d+", raw)
    return match.group() if match else raw


def _load_lectures(course_id: str) -> list[dict]:
    """Load and cache lecture data for a given course_id."""
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
            # Drop redundant columns
            for col in _DROP_COLUMNS:
                row.pop(col, None)
            # Normalize class and subject at load time
            row["Class"] = _normalize_class(row.get("Class", ""))
            row["Subject"] = (row.get("Subject") or "").strip().title()
            rows.append(row)

    _lecture_cache[course_id] = rows
    return rows


def _build_lecture(row: dict) -> dict:
    return {
        "microlecture_code": row.get("Microlecture_Code", ""),
        "chapter_id": row.get("chapter_id", ""),
        "microlecture_name": row.get("Microlecture_Name", ""),
        "hindi_title": row.get("final_microlecture_title", ""),
        "chapter_name": row.get("Chapter_Name", ""),
        "teacher_name": row.get("Teacher_Name", ""),
        "language": row.get("language", ""),
        "subscription_type": row.get("Subscription_Type", ""),
        "thumbnail": row.get("Lecture_Thumbnail", ""),
        "ppt_notes_link": row.get("PPT Notes Link Microlecture wise", ""),
        "video_start_time": int(row.get("Video_Start_Time", 0) or 0),
        "video_end_time": int(row.get("Video_End_Time", 0) or 0),
    }


async def search_lectures(
    subject: str,
    course_id: str,
    chapter_name: str | None = None,
    class_: str | None = None,
    topic: str | None = None,
) -> dict:
    """Find microlectures at three granularity levels.

    - subject only → subject-level params
    - subject + chapter → chapter-level params
    - subject + chapter + topic → exact matching microlectures
    """
    lectures = _load_lectures(course_id)
    if not lectures:
        return {"lectures": [], "error": "No lecture data for this course"}

    subject_lower = subject.strip().lower()

    # Subject-level: no chapter specified
    if not chapter_name:
        subject_rows = [
            r for r in lectures
            if r["Subject"].lower() == subject_lower
            and (not class_ or r["Class"] == class_)
        ]
        if not subject_rows:
            return {"error": f"No lectures found for {subject}"}
        return {
            "level": "subject",
            "subject": subject,
            "course_id": course_id,
            "total_chapters": len({r["Chapter_Name"] for r in subject_rows}),
            "total_microlectures": len(subject_rows),
        }

    # Filter to chapter + subject + optional class
    chapter_lower = chapter_name.strip().lower()
    chapter_rows = [
        r for r in lectures
        if r["Chapter_Name"].strip().lower() == chapter_lower
        and r["Subject"].lower() == subject_lower
        and (not class_ or r["Class"] == class_)
    ]

    if not chapter_rows:
        return {"lectures": [], "error": "No lectures found for this chapter"}

    # Topic-level: return matching microlectures
    if topic:
        topic_lower = topic.strip().lower()
        results = [
            _build_lecture(row) for row in chapter_rows
            if topic_lower in (row.get("Microlecture_Name") or "").lower()
            or topic_lower in (row.get("final_microlecture_title") or "").lower()
        ]
        if results:
            return {"level": "topic", "lectures": results}
        return {
            "level": "topic",
            "lectures": [_build_lecture(r) for r in chapter_rows],
            "note": f"No exact match for topic '{topic}', showing all lectures for this chapter.",
        }

    # Chapter-level: return chapter params
    sample = chapter_rows[0]
    return {
        "level": "chapter",
        "chapter_name": sample.get("Chapter_Name", ""),
        "chapter_id": sample.get("chapter_id", ""),
        "hindi_title": sample.get("final_chapter_title", ""),
        "subject": sample.get("Subject", ""),
        "course_id": course_id,
        "class": sample.get("Class", ""),
        "thumbnail": sample.get("Lecture_Thumbnail", ""),
        "total_microlectures": len(chapter_rows),
    }


_definition = ToolDefinition(
    name="search_lectures",
    description=(
        "Search for lectures at three levels: "
        "1) Subject-level (just subject) → returns subject navigation info. "
        "2) Chapter-level (subject + chapter_name) → returns chapter navigation info. "
        "3) Topic-level (subject + chapter_name + topic) → returns exact microlectures. "
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
                "description": "The student's class number (e.g., '12'). Omit to search all.",
            },
            "topic": {
                "type": "string",
                "description": (
                    "A specific topic within a chapter (e.g. 'Dipole', 'Fermentation'). "
                    "Only pass when the student asks for a specific topic."
                ),
            },
        },
        "required": ["subject", "course_id"],
    },
    required_params=["subject", "course_id"],
)

register_tool(_definition, search_lectures)
