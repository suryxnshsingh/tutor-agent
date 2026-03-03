import csv
from pathlib import Path

from app.llm.base import ToolDefinition
from app.tools.registry import register_tool

# Per-course curriculum cache: course_id -> list of row dicts
_COURSES_DIR = Path(__file__).parent.parent / "data" / "courses"
_curriculum_cache: dict[str, list[dict]] = {}


def _load_curriculum(course_id: str) -> list[dict]:
    """Load and cache curriculum data for a given course_id."""
    if course_id in _curriculum_cache:
        return _curriculum_cache[course_id]

    csv_path = _COURSES_DIR / course_id / "curriculum.csv"
    if not csv_path.exists():
        _curriculum_cache[course_id] = []
        return []

    rows: list[dict] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["keywords"] = [kw.strip().lower() for kw in row["keywords"].split(",")]
            rows.append(row)

    _curriculum_cache[course_id] = rows
    return rows


async def resolve_chapter(
    query: str, subject: str, course_id: str, class_: str | None = None,
) -> dict:
    """Match a student query to curriculum chapters using keyword overlap."""
    curriculum = _load_curriculum(course_id)
    if not curriculum:
        return {"chapters": [], "error": "No curriculum data for this course"}

    query_tokens = set(query.lower().split())

    scored = []
    for row in curriculum:
        if row["subject"].lower() != subject.lower():
            continue
        if class_ and row["class"] != class_:
            continue

        # Score by keyword overlap
        overlap = query_tokens & set(row["keywords"])
        if not overlap:
            continue

        relevance = round(len(overlap) / max(len(query_tokens), 1), 2)
        scored.append({
            "class": row["class"],
            "chapter_name": row["chapter_name"],
            "chapter_number": int(row["chapter_number"]),
            "relevance": relevance,
        })

    # Sort by relevance descending, return top 5
    scored.sort(key=lambda x: x["relevance"], reverse=True)
    return {"chapters": scored[:5]}


# Register tool with the registry
_definition = ToolDefinition(
    name="resolve_chapter",
    description=(
        "Given a student query and subject, returns the most likely matching chapters "
        "from the curriculum. Use this when the student mentions a topic but no specific "
        "chapter has been identified. IMPORTANT: The query parameter must be in English. "
        "If the student wrote in Hindi or Hinglish, translate the topic to English keywords "
        "before calling this tool (e.g. 'vidyut dhara' → 'current electricity')."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The topic or keywords translated to English (e.g. 'current electricity', not 'vidyut dhara')",
            },
            "subject": {
                "type": "string",
                "description": "The subject (Physics, Chemistry, etc.)",
            },
            "course_id": {
                "type": "string",
                "description": "The course ID (e.g., '6' for CBSE)",
            },
            "class_": {
                "type": "string",
                "description": "The student's class (e.g., '11', '12'). Filters results to that class. Omit to search across all classes.",
            },
        },
        "required": ["query", "subject", "course_id"],
    },
    required_params=["query", "subject", "course_id"],
)

register_tool(_definition, resolve_chapter)
