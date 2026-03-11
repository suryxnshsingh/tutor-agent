# Arivihan Agent — Project Guidelines

Autonomous AI tutoring agent for Indian students (Tier-2/3 cities). Converses in Hindi, English, and Hinglish. Fetches learning materials, guides through concepts, maintains chat-scoped context.

## Tech Stack

- **Runtime:** Python 3.12+, FastAPI, async everywhere
- **LLM:** OpenAI SDK (provider-agnostic via adapter layer)
- **Storage:** Redis (hot cache, 48h TTL), DynamoDB (persistent, async fire-and-forget writes)
- **Transport:** SSE for streaming responses (via `sse-starlette`)
- **Dev tooling:** uv (package management), ruff (lint + format), pytest, asyncio
- **Deployment:** Docker + docker-compose (FastAPI + Redis)

## Architecture

3-layer execution model: **Gateway → Runner → Attempt**

- **Gateway** — FastAPI entry. Normalizes request to `UniversalMessage`. Pydantic models for request/response schemas only. Auth and rate limiting not yet implemented.
- **Runner** — Orchestration. Session load/save, system prompt assembly, response assembly (text + resource cards), thinking tag stripping/extraction.
- **Attempt** — Single LLM call cycle. Stateless. Tool loop until final response. Yields `AgentEvent` stream.

See `ARCHITECTURE.md` for full details.

## Key Conventions

### Models & Types
- **Dataclasses** for all internal types: `InternalMessage`, `ChatSession`, `AgentEvent` variants, `ToolDefinition`.
- **Pydantic** only for API request/response schemas in `gateway/models.py`.

### LLM Layer
- Provider-agnostic. All conversation data in `InternalMessage` format — never store or pass provider-specific formats.
- Adapter converts internal format ↔ provider format. Currently only OpenAI adapter implemented.
- Swap provider = implement new adapter class + change config.

### Tools
- One tool per data source file. Each tool loads its own CSV from `app/data/courses/{course_id}/`.
- LLM chains tools autonomously — no hardcoded routing or flow sequences.
- **Implemented tools (8):**
  - `resolve_chapter` — keyword-based chapter matching from curriculum CSV
  - `search_lectures` — video microlecture search
  - `search_topper_notes` — topper-written notes PDFs
  - `search_ppt_notes` — PowerPoint notes PDFs
  - `search_pyq_papers` — previous year question papers
  - `search_important_questions` — important questions with type/exam filtering
  - `search_tests` — chapterwise and full-length practice tests
  - `search_ncert_solutions` — NCERT solutions (Maths only)

### Data Organization
- CSV files organized by course: `app/data/courses/{course_id}/`
- Known courses: `4` (CBSE, full data), `6` (curriculum only)
- Tools filter by `language` field in CSV for Hindi/English content

### System Prompt
- Template at `prompts/system_prompt.txt`, filled with student context (course, class, subject, language, exam year).
- Course-specific prompt snippets in `prompts/courses/{course_id}.txt`.
- Static block first (~1500 tokens, cacheable), dynamic student context last.

### Chain-of-Thought
- Agent uses `<thinking>` tags before every response.
- Runner strips thinking before sending to student. Never forwarded in SSE stream.
- Thinking content extracted and logged for analytics.

### Error Handling
- Student-facing errors: always friendly, in their language. Never expose technical details.
- Tool failures: returned as tool result dict, LLM decides how to respond gracefully.
- Redis/DynamoDB failures: handled with fallbacks (Redis miss → DynamoDB lookup → new session).

### General Rules
- **No mocks.** Never create mock services. Fix the real thing.
- **Async everywhere.** All IO-bound operations use async/await.
- **No hardcoded flows.** Agent reasons about intent and chains tools autonomously.

## Project Structure

```
app/
├── main.py                         # FastAPI app, lifespan (startup/shutdown)
├── config.py                       # Environment config, model settings
├── gateway/
│   ├── routes.py                   # POST /chat, POST /chat/stream
│   ├── models.py                   # Pydantic: ChatRequest, ChatResponse, UniversalMessage
│   └── adapters/                   # Channel adapters (placeholder)
├── agent/
│   ├── runner.py                   # Orchestrator: session, prompt, attempt, response
│   ├── attempt.py                  # Single LLM cycle + tool loop
│   ├── events.py                   # AgentEvent types (Status, Delta, Cards, Error, etc.)
│   └── prompt.py                   # System prompt builder + thinking extraction
├── llm/
│   ├── base.py                     # LLMProvider ABC, LLMResponse, ToolDefinition
│   └── openai_provider.py          # OpenAI adapter
├── tools/
│   ├── registry.py                 # Tool registration & execution
│   ├── resolve_chapter.py          # Curriculum chapter resolution
│   ├── search_lectures.py          # Video microlecture search
│   ├── search_topper_notes.py      # Topper notes PDF search
│   ├── search_ppt_notes.py         # PowerPoint notes search
│   ├── search_pyq_papers.py        # Previous year papers search
│   ├── search_important_questions.py # Important questions search
│   ├── search_tests.py             # Practice test search
│   └── search_ncert_solutions.py   # NCERT solutions search
├── session/
│   ├── manager.py                  # Session load/save (Redis → DynamoDB fallback)
│   ├── redis_store.py              # Hot cache (48h TTL)
│   └── dynamo_store.py             # Persistent storage (async via to_thread)
├── messages/
│   └── models.py                   # InternalMessage, ChatSession dataclasses
└── data/
    └── courses/{course_id}/        # CSV data files per course
tests/                              # Test suite (empty, not yet written)
prompts/
├── system_prompt.txt               # Main system prompt template
└── courses/                        # Course-specific prompt snippets
```

## Not Yet Implemented

- Retry with backoff and fallback model on LLM failures
- Context compaction (summarize old turns when history grows)
- Token-by-token streaming (currently yields full response as one delta)
- Rate limiting and authentication on gateway
- `generate_coupon` tool
- Test suite
- Additional LLM adapters (Claude, etc.)

## Development Setup

1. Python 3.12+ required
2. `uv` for dependency management (`uv sync` to install)
3. `ruff` for linting and formatting (100-char line length, Python 3.12 target)
4. `pytest` for tests (all async tests use `pytest-asyncio`)
5. `docker-compose up` for local dev (FastAPI + Redis)
