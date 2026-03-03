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


async def search_lectures(
    chapter_name: str,
    subject: str,
    course_id: str,
    class_: str | None = None,
) -> dict:
    """Find microlectures for a given chapter, subject, and course."""
    lectures = _load_lectures(course_id)
    if not lectures:
        return {"lectures": [], "error": "No lecture data for this course"}

    chapter_lower = chapter_name.strip().lower()
    subject_lower = subject.strip().lower()

    results = []
    for row in lectures:
        if row["Chapter_Name"].strip().lower() != chapter_lower:
            continue
        if row["Subject"].lower() != subject_lower:
            continue
        if class_ and row["Class"] != class_:
            continue

        results.append({
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
        })

    return {"lectures": results}


_definition = ToolDefinition(
    name="search_lectures",
    description=(
        "Given a chapter name and subject, returns all microlectures for that chapter. "
        "Use this AFTER resolve_chapter has identified the chapter. Returns microlecture "
        "names (English + Hindi), video timestamps, teacher info, and deep-link metadata. "
        "The chapter_name must be the English chapter name from resolve_chapter results."
    ),
    parameters={
        "type": "object",
        "properties": {
            "chapter_name": {
                "type": "string",
                "description": (
                    "The English chapter name from resolve_chapter "
                    "results (e.g. 'Microbes in Human Welfare')"
                ),
            },
            "subject": {
                "type": "string",
                "description": "The subject (Physics, Chemistry, Biology, etc.)",
            },
            "course_id": {
                "type": "string",
                "description": "The course ID (e.g., '4' for MPBSE)",
            },
            "class_": {
                "type": "string",
                "description": (
                    "The student's class number (e.g., '12'). "
                    "Filters to that class. Omit to search all."
                ),
            },
        },
        "required": ["chapter_name", "subject", "course_id"],
    },
    required_params=["chapter_name", "subject", "course_id"],
)

register_tool(_definition, search_lectures)
