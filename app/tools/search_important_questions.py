import csv
import re
from pathlib import Path

from app.llm.base import ToolDefinition
from app.tools.registry import register_tool

_COURSES_DIR = Path(__file__).parent.parent / "data" / "courses"
_iq_cache: dict[str, list[dict]] = {}

_DROP_COLUMNS = {"Exam_Type", "Question_Type"}

# Question type columns present in the CSV
_QUESTION_TYPE_COLS = [
    "CONCEPTUAL_QUESTION",
    "LONG_ANSWERS",
    "SHORT_ANSWERS",
    "VERY_SHORT_ANSWERS",
    "FILL_IN_THE_BLANKS",
    "ONE_WORD_ANSWER",
    "MULTIPLE_CHOICE_QUESTION",
    "TRUE_FALSE",
    "NUMERICAL_QUESTION",
    "MATCH_THE_COLUMN",
]


def _normalize_class(raw: str) -> str:
    """Extract digits from class string, e.g. 'Class 12th' -> '12'."""
    match = re.search(r"\d+", raw)
    return match.group() if match else raw


def _load_important_questions(course_id: str) -> list[dict]:
    """Load and cache important questions data for a given course_id."""
    if course_id in _iq_cache:
        return _iq_cache[course_id]

    csv_path = _COURSES_DIR / course_id / "important_questions.csv"
    if not csv_path.exists():
        _iq_cache[course_id] = []
        return []

    rows: list[dict] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for col in _DROP_COLUMNS:
                row.pop(col, None)
            row["Class"] = _normalize_class(row.get("Class", ""))
            row["Subject"] = (row.get("Subject") or "").strip().title()
            # Parse chapter_number to int for range filtering
            try:
                row["chapter_number"] = int(row.get("chapter_number", 0) or 0)
            except (ValueError, TypeError):
                row["chapter_number"] = 0
            rows.append(row)

    _iq_cache[course_id] = rows
    return rows


def _build_question_content(row: dict, question_type: str | None) -> dict:
    """Extract question content for the requested type(s)."""
    questions: dict[str, str] = {}
    if question_type:
        # Single type requested
        col = question_type.upper()
        if col in _QUESTION_TYPE_COLS:
            val = row.get(col, "")
            if val and val.strip():
                questions[col.lower()] = val.strip()
    else:
        # Return all available types
        for col in _QUESTION_TYPE_COLS:
            val = row.get(col, "")
            if val and val.strip():
                questions[col.lower()] = val.strip()
    return questions


async def search_important_questions(
    subject: str,
    course_id: str,
    chapter_name: str | None = None,
    chapter_number: int | None = None,
    chapter_number_from: int | None = None,
    chapter_number_to: int | None = None,
    exam_schedule: str | None = None,
    question_type: str | None = None,
    class_: str | None = None,
) -> dict:
    """Find important questions by subject, chapter, schedule, and type."""
    data = _load_important_questions(course_id)
    if not data:
        return {
            "results": [],
            "error": "No important questions data for this course",
        }

    subject_lower = subject.strip().lower()
    chapter_lower = chapter_name.strip().lower() if chapter_name else None
    schedule = (exam_schedule or "finalexam").strip().lower()

    results = []
    for row in data:
        if row["Subject"].lower() != subject_lower:
            continue
        if class_ and row["Class"] != class_:
            continue
        if row.get("Exam_Schedule", "").strip().lower() != schedule:
            continue
        # Chapter filtering
        if chapter_lower and row["Chapter_Name"].strip().lower() != chapter_lower:
            continue
        if chapter_number and row["chapter_number"] != chapter_number:
            continue
        if chapter_number_from and row["chapter_number"] < chapter_number_from:
            continue
        if chapter_number_to and row["chapter_number"] > chapter_number_to:
            continue

        questions = _build_question_content(row, question_type)
        if not questions and question_type:
            # Requested type has no content for this row, skip
            continue

        results.append({
            "chapter_name": row.get("Chapter_Name", ""),
            "chapter_number": row["chapter_number"],
            "title": row.get("Final_Title", ""),
            "language": row.get("language", ""),
            "download_link": row.get("download_link", ""),
            "exam_schedule": row.get("Exam_Schedule", ""),
            "questions": questions,
        })

    # Sort by chapter number
    results.sort(key=lambda x: x["chapter_number"])
    return {"results": results}


_definition = ToolDefinition(
    name="search_important_questions",
    description=(
        "Returns important questions for a subject, optionally filtered "
        "by chapter, exam schedule, and question type. Each result "
        "includes a PDF download link and the actual question content "
        "grouped by type (MCQ, short answer, long answer, etc.). "
        "Use when the student asks for important questions, expected "
        "questions, or question-type-specific practice material."
    ),
    parameters={
        "type": "object",
        "properties": {
            "subject": {
                "type": "string",
                "description": (
                    "The subject (Physics, Chemistry, Biology, Maths)"
                ),
            },
            "course_id": {
                "type": "string",
                "description": "The course ID (e.g., '4' for MPBSE)",
            },
            "chapter_name": {
                "type": "string",
                "description": (
                    "English chapter name from resolve_chapter. "
                    "Omit to get all chapters."
                ),
            },
            "chapter_number": {
                "type": "integer",
                "description": (
                    "Specific chapter number. "
                    "Use chapter_number_from/to for ranges."
                ),
            },
            "chapter_number_from": {
                "type": "integer",
                "description": (
                    "Start of chapter range (inclusive). "
                    "E.g. 1 for 'chapter 1 se 5'."
                ),
            },
            "chapter_number_to": {
                "type": "integer",
                "description": (
                    "End of chapter range (inclusive). "
                    "E.g. 5 for 'chapter 1 se 5'."
                ),
            },
            "exam_schedule": {
                "type": "string",
                "description": (
                    "Exam schedule filter: 'finalexam' (default), "
                    "'quarterly', or 'halfyearly'."
                ),
            },
            "question_type": {
                "type": "string",
                "description": (
                    "Filter by question type. One of: "
                    "MULTIPLE_CHOICE_QUESTION, SHORT_ANSWERS, "
                    "LONG_ANSWERS, VERY_SHORT_ANSWERS, "
                    "FILL_IN_THE_BLANKS, ONE_WORD_ANSWER, "
                    "TRUE_FALSE, NUMERICAL_QUESTION, "
                    "CONCEPTUAL_QUESTION, MATCH_THE_COLUMN. "
                    "Omit to return all types."
                ),
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

register_tool(_definition, search_important_questions)
