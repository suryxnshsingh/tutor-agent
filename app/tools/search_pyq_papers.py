import csv
import re
from pathlib import Path

from app.llm.base import ToolDefinition
from app.tools.registry import register_tool

_COURSES_DIR = Path(__file__).parent.parent / "data" / "courses"
_pyq_cache: dict[str, list[dict]] = {}

_DROP_COLUMNS = {"Exam_Type", "board_type", "sample_paper_id"}


def _normalize_class(raw: str) -> str:
    """Extract digits from class string, e.g. 'Class 12th' -> '12'."""
    match = re.search(r"\d+", raw)
    return match.group() if match else raw


def _load_pyq_papers(course_id: str) -> list[dict]:
    """Load and cache PYQ paper data for a given course_id."""
    if course_id in _pyq_cache:
        return _pyq_cache[course_id]

    csv_path = _COURSES_DIR / course_id / "pyq_pdf.csv"
    if not csv_path.exists():
        _pyq_cache[course_id] = []
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

    _pyq_cache[course_id] = rows
    return rows


async def search_pyq_papers(
    subject: str,
    course_id: str,
    year_from: str | None = None,
    year_to: str | None = None,
    class_: str | None = None,
) -> dict:
    """Find PYQ paper PDFs by subject and optional year range."""
    papers = _load_pyq_papers(course_id)
    if not papers:
        return {"papers": [], "error": "No PYQ paper data for this course"}

    subject_lower = subject.strip().lower()

    # Parse year bounds
    y_from = int(year_from) if year_from else None
    y_to = int(year_to) if year_to else None

    results = []
    for row in papers:
        if row["Subject"].lower() != subject_lower:
            continue
        if class_ and row["Class"] != class_:
            continue

        row_year = int(row.get("year", 0) or 0)
        if y_from and row_year < y_from:
            continue
        if y_to and row_year > y_to:
            continue

        results.append({
            "id": row.get("id", ""),
            "title": row.get("Title", ""),
            "subject": row.get("Subject", ""),
            "year": row_year,
            "language": row.get("language", ""),
            "subscription_type": row.get("Subscription_Type", ""),
            "download_link": row.get("download_link", ""),
        })

    # Sort by year descending (most recent first)
    results.sort(key=lambda x: x["year"], reverse=True)
    return {"papers": results}


_definition = ToolDefinition(
    name="search_pyq_papers",
    description=(
        "Returns previous year question paper PDFs for a subject. "
        "Supports filtering by year range. Use when the student asks "
        "for PYQ papers, previous papers, or mentions a specific year "
        "(e.g. '2023 ka paper dedo', 'last 3 years ke papers')."
    ),
    parameters={
        "type": "object",
        "properties": {
            "subject": {
                "type": "string",
                "description": (
                    "The subject (Physics, Chemistry, Biology, "
                    "Mathematics, Hindi, English)"
                ),
            },
            "course_id": {
                "type": "string",
                "description": "The course ID (e.g., '4' for MPBSE)",
            },
            "year_from": {
                "type": "string",
                "description": (
                    "Start year for range filter (inclusive). "
                    "E.g. '2020' to get papers from 2020 onwards."
                ),
            },
            "year_to": {
                "type": "string",
                "description": (
                    "End year for range filter (inclusive). "
                    "E.g. '2023' to get papers up to 2023."
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

register_tool(_definition, search_pyq_papers)
