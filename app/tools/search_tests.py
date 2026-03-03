import csv
import re
from pathlib import Path

from app.llm.base import ToolDefinition
from app.tools.registry import register_tool

_COURSES_DIR = Path(__file__).parent.parent / "data" / "courses"
_test_cache: dict[str, list[dict]] = {}

_DROP_COLUMNS = {"Exam_Type", "user_subject"}

_CSV_FILES = {
    "chapterwise": "chapterwise_test.csv",
    "fulllength": "fulllength_test.csv",
}


def _normalize_class(raw: str) -> str:
    """Extract digits from class string, e.g. 'Class 12th' -> '12'."""
    match = re.search(r"\d+", raw)
    return match.group() if match else raw


def _load_tests(course_id: str, test_type: str) -> list[dict]:
    """Load and cache test data for a given course_id and test_type."""
    cache_key = f"{course_id}:{test_type}"
    if cache_key in _test_cache:
        return _test_cache[cache_key]

    filename = _CSV_FILES.get(test_type)
    if not filename:
        _test_cache[cache_key] = []
        return []

    csv_path = _COURSES_DIR / course_id / filename
    if not csv_path.exists():
        _test_cache[cache_key] = []
        return []

    rows: list[dict] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for col in _DROP_COLUMNS:
                row.pop(col, None)
            row["Class"] = _normalize_class(row.get("Class", ""))
            row["Subject"] = (row.get("Subject") or "").strip().title()
            # Parse chapter number if present
            try:
                row["Chapter_Number"] = int(row.get("Chapter_Number", 0) or 0)
            except (ValueError, TypeError):
                row["Chapter_Number"] = 0
            rows.append(row)

    _test_cache[cache_key] = rows
    return rows


def _load_all_tests(course_id: str) -> list[tuple[str, dict]]:
    """Load both test types, returning (test_type, row) tuples."""
    results = []
    for test_type in _CSV_FILES:
        for row in _load_tests(course_id, test_type):
            results.append((test_type, row))
    return results


async def search_tests(
    subject: str,
    course_id: str,
    test_type: str | None = None,
    chapter_name: str | None = None,
    chapter_number: int | None = None,
    class_: str | None = None,
) -> dict:
    """Find chapterwise or full-length tests."""
    # Load based on test_type
    if test_type and test_type in _CSV_FILES:
        entries = [(test_type, row) for row in _load_tests(course_id, test_type)]
    else:
        entries = _load_all_tests(course_id)

    if not entries:
        return {"tests": [], "error": "No test data for this course"}

    subject_lower = subject.strip().lower()
    chapter_lower = chapter_name.strip().lower() if chapter_name else None

    results = []
    for t_type, row in entries:
        if row["Subject"].lower() != subject_lower:
            continue
        if class_ and row["Class"] != class_:
            continue
        # Chapter filters only apply to chapterwise
        if t_type == "chapterwise":
            if chapter_lower and row["Chapter_Name"].strip().lower() != chapter_lower:
                continue
            if chapter_number and row["Chapter_Number"] != chapter_number:
                continue

        result = {
            "test_id": row.get("test_id", ""),
            "test_type": t_type,
            "test_name": row.get("Chapter_Name", ""),
            "subject": row.get("Subject", ""),
            "language": row.get("language", ""),
            "subscription_type": row.get("Subscription_Type", ""),
            "category": row.get("category", ""),
        }
        if t_type == "chapterwise":
            result["chapter_number"] = row["Chapter_Number"]

        results.append(result)

    # Sort: chapterwise by chapter number, fulllength by test_id
    results.sort(key=lambda x: (x["test_type"], x.get("chapter_number", 0)))
    return {"tests": results}


_definition = ToolDefinition(
    name="search_tests",
    description=(
        "Returns practice tests — either chapterwise (per-chapter) or "
        "full-length (subject-level mock tests). Use when the student "
        "asks for tests, mock tests, practice tests, or test series. "
        "Pass test_type='chapterwise' for chapter tests or "
        "'fulllength' for mock exams. Omit test_type to return both."
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
            "test_type": {
                "type": "string",
                "description": (
                    "'chapterwise' for chapter tests, 'fulllength' "
                    "for mock exams. Omit to search both."
                ),
            },
            "chapter_name": {
                "type": "string",
                "description": (
                    "Chapter name to filter chapterwise tests. "
                    "From resolve_chapter results."
                ),
            },
            "chapter_number": {
                "type": "integer",
                "description": (
                    "Chapter number to filter chapterwise tests."
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

register_tool(_definition, search_tests)
