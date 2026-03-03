# Arivihan Agent — Project Guidelines

Autonomous AI tutoring agent for Indian students (Tier-2/3 cities). Converses in Hindi, English, and Hinglish. Fetches learning materials, guides through concepts, maintains chat-scoped context.

## Tech Stack

- **Runtime:** Python 3.12+, FastAPI, async everywhere
- **LLM:** OpenAI SDK (provider-agnostic via adapter layer)
- **Storage:** Redis (hot cache, 24-48h TTL), DynamoDB (persistent, async writes)
- **Transport:** SSE for streaming responses
- **Dev tooling:** uv (package management), ruff (lint + format), pytest, asyncio

## Architecture

3-layer execution model: **Gateway → Runner → Attempt**

- **Gateway** — FastAPI entry. Auth, rate limit, normalize to `UniversalMessage`. Pydantic models for request/response schemas only.
- **Runner** — Orchestration. Session load/save, context compaction, system prompt assembly, retry/fallback, response assembly (text + cards + buttons), thinking tag stripping.
- **Attempt** — Single LLM call cycle. Stateless. Tool loop until final response. Yields `AgentEvent` stream.

See `ARCHITECTURE.md` for full details.

## Key Conventions

### Models & Types
- **Dataclasses** for all internal types: `InternalMessage`, `ChatSession`, `AgentEvent` variants, `ToolDefinition`, `AgentResponse`.
- **Pydantic** only for API request/response schemas in `gateway/models.py`.

### LLM Layer
- Provider-agnostic. All conversation data in `InternalMessage` format — never store or pass provider-specific formats.
- Adapter converts internal format ↔ provider format. Swap provider = implement new adapter class + change config.

### Tools
- Grouped by data source, not use case.
- LLM chains tools autonomously — no hardcoded routing or flow sequences.
- Tools: `resolve_chapter`, `search_study_material`, `search_exam_content`, `generate_coupon`.

### System Prompt
- Static block first (~1500 tokens, cacheable across all requests), dynamic student context last.
- OpenAI prompt caching gives 50% discount on cached prefix.

### Chain-of-Thought
- Agent uses `<thinking>` tags before every response.
- Runner strips thinking before sending to student. Never forwarded in SSE stream.
- Thinking content logged for analytics.

### Error Handling
- Student-facing errors: always friendly, in their language. Never expose technical details.
- LLM failures: retry with backoff → fallback model → friendly error message.
- Tool failures: returned as tool result, LLM decides how to respond.

### General Rules
- **No mocks.** Never create mock services. Fix the real thing.
- **Async everywhere.** All IO-bound operations use async/await.
- **No hardcoded flows.** Agent reasons about intent and chains tools autonomously.

## Project Structure

```
app/
├── main.py                     # FastAPI app, startup, shutdown
├── config.py                   # Environment config, model settings
├── gateway/
│   ├── routes.py               # POST /chat, POST /chat/stream
│   ├── models.py               # Pydantic request/response models
│   └── adapters/               # Channel adapters (web, whatsapp)
├── agent/
│   ├── runner.py               # Orchestrator
│   ├── attempt.py              # Single LLM cycle + tool loop
│   ├── events.py               # AgentEvent types
│   └── prompt.py               # System prompt builder
├── llm/
│   ├── base.py                 # LLMProvider ABC, LLMResponse, ToolDefinition
│   └── openai_provider.py      # OpenAI adapter
├── tools/
│   ├── registry.py             # Tool registry and executor
│   ├── resolve_chapter.py
│   ├── search_study.py
│   ├── search_exam.py
│   └── generate_coupon.py
├── session/
│   ├── manager.py              # Session load/save orchestration
│   ├── redis_store.py
│   ├── dynamo_store.py
│   └── compaction.py
├── messages/
│   └── models.py               # InternalMessage, ChatSession dataclasses
└── data/                       # CSV data files
tests/
prompts/
    └── system_prompt.txt       # Full system prompt template
```

## Development Setup

1. Python 3.12+ required
2. `uv` for dependency management
3. `ruff` for linting and formatting
4. `pytest` for tests (all async tests use `pytest-asyncio`)
