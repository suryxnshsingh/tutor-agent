# Arivihan Educational AI Agent — System Architecture

> **Version:** 1.0 | **Date:** March 2026
> **Stack:** Python (FastAPI) · OpenAI (primary, provider-agnostic) · Redis · DynamoDB · SSE · AWS/K8s

---

## 1. Overview

An autonomous, conversational tutoring agent for students in Tier-2/Tier-3 Indian cities. The agent converses in Hindi, English, and Hinglish, fetches learning materials, guides students through concepts, and maintains chat-scoped context. No rigid classification or routing — the LLM reasons about intent and chains tools autonomously.

### Design Principles

- **Autonomy over classification** — agent reasons about intent, no hardcoded routing
- **Channel-agnostic core** — web today, WhatsApp tomorrow, zero core changes
- **Provider-agnostic LLM layer** — OpenAI primary, swap via adapters
- **Tools as capabilities** — agent chains tools naturally, no hardcoded flows
- **Cost-conscious** — prompt caching, lean context, smart compaction

---

## 2. Three-Layer Execution Model

Inspired by OpenClaw's runner → attempt pattern and Pi's agent-core state machine.

```
Student Message (web / WhatsApp)
         │
         ▼
┌──────────────────────────────────────┐
│  GATEWAY (gateway.py)                │
│  Auth → Rate Limit → Normalize      │
│  → UniversalMessage                  │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  RUNNER (runner.py)                  │
│  1. Load session (Redis / DynamoDB)  │
│  2. Check context, compact if needed │
│  3. Build system prompt              │
│  4. Call Attempt                     │
│  5. Receive events                   │
│  6. Append to session history        │
│  7. Save session (Redis + async DB)  │
│  8. Build response (text + cards)    │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  ATTEMPT (attempt.py)                │
│  1. Call LLM via adapter             │
│  2. ┌── TOOL LOOP ───────────┐      │
│     │ LLM wants tool_use?    │      │
│     │  YES → Execute → Feed  │      │
│     │      → Call LLM again  │      │
│     │  NO  → Final response  │      │
│     └────────────────────────┘      │
│  3. Yield events as they happen      │
└──────────────────────────────────────┘
```

### 2.1 Gateway (`gateway.py`)

FastAPI entry point. Authenticates, normalizes incoming messages into `UniversalMessage`, routes to Runner.

```python
@dataclass
class UniversalMessage:
    student_id: str       # unique student identifier
    text: str             # student's message
    board: str            # CBSE, UP Board, etc.
    class_: str           # 11, 12, etc.
    stream: str           # PCM, PCB, Commerce
    subject: str          # Physics, Chemistry, etc.
    language: str         # hi, en, hinglish
    chat_id: str          # frontend-generated chat ID
    channel: str          # web, whatsapp
```

**Frontend always sends:** `student_id`, `board`, `class_`, `stream`, `subject`, `language`, `chat_id`, `channel`

**Agent figures out:** chapter, topic, what the student needs — via tools and reasoning.

Adding a new channel (WhatsApp, Telegram) = one new adapter. Zero changes to Runner or Attempt.

### 2.2 Runner (`runner.py`)

The orchestration layer. Everything the Attempt shouldn't worry about:

- **Session management:** Load from Redis (or DynamoDB on cache miss). Save after each turn.
- **Context window management:** Check token count. If too long → compact older turns via cheap model call, keep recent turns in full detail.
- **System prompt assembly:** Static block first (for prompt caching) + dynamic student context last.
- **Retry & fallback:** If LLM call fails → retry with backoff → fallback to secondary model.
- **Response assembly:** Combine Attempt's text output + tool results → structured response (text + cards + buttons).
- **Event forwarding:** For streaming requests, forward Attempt events to SSE transport.
- **Thinking tag stripping:** Remove `<thinking>` blocks from agent response before sending to student.

### 2.3 Attempt (`attempt.py`)

One single LLM call cycle. Stateless. Receives fully built context, runs tool loop, yields events.

```python
from typing import AsyncGenerator

async def run_attempt(
    system_prompt: str,
    messages: list[InternalMessage],
    tools: list[ToolDefinition],
    provider: LLMProvider,
    model: str
) -> AsyncGenerator[AgentEvent, None]:

    while True:
        response = await provider.chat(
            system_prompt=system_prompt,
            messages=messages,
            tools=tools,
            model=model
        )

        if response.has_tool_calls:
            for tool_call in response.tool_calls:
                yield StatusEvent(f"Using {tool_call.name}...")
                result = await execute_tool(tool_call)
                messages.append(make_tool_call_message(tool_call))
                messages.append(make_tool_result_message(tool_call.id, result))
        else:
            for token in response.text_stream:
                yield ResponseDelta(token)
            yield ResponseEnd()
            return
```

Knows nothing about sessions, retries, students, or channels. Pure function in, events out.

---

## 3. LLM Abstraction Layer

Business logic never touches provider SDKs. A clean adapter sits between Agent Core and any LLM.

### 3.1 Provider Interface

```python
from abc import ABC, abstractmethod

class LLMProvider(ABC):

    @abstractmethod
    async def chat(
        self,
        system_prompt: str,
        messages: list[InternalMessage],
        tools: list[ToolDefinition],
        model: str
    ) -> LLMResponse:
        """Send a chat request to the LLM provider."""
        ...

    @abstractmethod
    def to_provider_messages(self, messages: list[InternalMessage]) -> list[dict]:
        """Convert internal messages to provider format."""
        ...

    @abstractmethod
    def to_provider_tools(self, tools: list[ToolDefinition]) -> list[dict]:
        """Convert internal tool definitions to provider format."""
        ...

    @abstractmethod
    def from_provider_response(self, response) -> list[InternalMessage]:
        """Convert provider response to internal messages."""
        ...
```

### 3.2 Adapter Implementation (OpenAI)

```python
class OpenAIProvider(LLMProvider):

    async def chat(self, system_prompt, messages, tools, model):
        provider_messages = self.to_provider_messages(messages)
        provider_tools = self.to_provider_tools(tools)

        response = await self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                *provider_messages
            ],
            tools=provider_tools,
            stream=True
        )

        return self.from_provider_response(response)
```

### 3.3 Design Decisions

- **Lowest common denominator:** Only text messages, tool/function calling, streaming. No extended thinking, no provider-specific features.
- **One class per provider.** Swap from OpenAI to Claude = implement new class + change config.
- **Tool schemas are provider-agnostic.** Defined in our own format, adapter converts to provider-specific (OpenAI function calling, Claude tool_use, etc.)

---

## 4. Internal Message Format

All conversation data stored in Arivihan's own format — not OpenAI's, not Claude's. Zero vendor lock-in at data layer.

### 4.1 Message Roles

| Role | Description |
|------|-------------|
| `student` | Student's message |
| `agent` | Agent's final text response |
| `tool_call` | Agent requesting a tool execution |
| `tool_result` | Result of a tool execution |

### 4.2 Message Schema

```python
from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import datetime

@dataclass
class InternalMessage:
    role: str                          # student | agent | tool_call | tool_result
    content: Optional[str] = None      # text content (for student/agent messages)
    tool_name: Optional[str] = None    # for tool_call
    tool_input: Optional[dict] = None  # for tool_call
    call_id: Optional[str] = None      # links tool_call ↔ tool_result
    result: Optional[Any] = None       # for tool_result
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)
```

### 4.3 Example Conversation

```json
[
  {
    "role": "student",
    "content": "potential energy samajh nahi aa raha",
    "timestamp": "2026-03-02T10:30:00Z",
    "metadata": { "student_id": "stu_123", "subject": "Physics", "board": "CBSE" }
  },
  {
    "role": "tool_call",
    "tool_name": "resolve_chapter",
    "tool_input": { "query": "potential energy", "subject": "Physics", "board": "CBSE" },
    "call_id": "call_abc123",
    "timestamp": "2026-03-02T10:30:01Z"
  },
  {
    "role": "tool_result",
    "call_id": "call_abc123",
    "result": { "chapters": [{ "name": "Work Energy and Power", "relevance": 0.95 }] },
    "timestamp": "2026-03-02T10:30:01Z"
  },
  {
    "role": "tool_call",
    "tool_name": "search_study_material",
    "tool_input": { "board": "CBSE", "subject": "Physics", "chapters": ["Work Energy and Power"], "type": "lecture" },
    "call_id": "call_def456",
    "timestamp": "2026-03-02T10:30:02Z"
  },
  {
    "role": "tool_result",
    "call_id": "call_def456",
    "result": { "materials": [{ "title": "Potential Energy Derivation", "url": "...", "duration": "12 min" }] },
    "timestamp": "2026-03-02T10:30:02Z"
  },
  {
    "role": "agent",
    "content": "Potential energy basically ek body ki position ke wajah se hoti hai...",
    "timestamp": "2026-03-02T10:30:03Z",
    "metadata": { "model": "gpt-4o", "tokens_in": 1200, "tokens_out": 350, "latency_ms": 2100 }
  }
]
```

Metadata wraps messages for analytics/debugging but gets stripped before sending to LLM.

---

## 5. Tool System

Tools are grouped by data source, not by use case. The LLM chains them autonomously based on tool descriptions.

### 5.1 Tool Inventory

| Tool | Purpose | Data Source |
|------|---------|-------------|
| `resolve_chapter` | Maps query keywords to curriculum chapters. Returns ranked list. | Curriculum table |
| `search_study_material` | Searches lectures, notes, PPTs by board/subject/chapter/type. | Study materials table |
| `search_exam_content` | Searches PYQs, important questions, past papers by board/subject/chapter/year/type. | Exam content table |
| `generate_coupon` | Generates coupon codes via external API. | External API |

### 5.2 Tool Definition Schema (Provider-Agnostic)

```python
@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict       # JSON Schema
    required_params: list[str]
```

Tools defined in this format. LLM adapter converts to provider-specific format (OpenAI function calling, Claude tool_use, etc.)

### 5.3 Tool Schemas

#### resolve_chapter

```json
{
  "name": "resolve_chapter",
  "description": "Given a student query and subject, returns the most likely matching chapters from the curriculum. Use this when the student mentions a topic but no specific chapter has been identified.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": { "type": "string", "description": "The topic or keywords from the student's message" },
      "subject": { "type": "string", "description": "The subject (Physics, Chemistry, etc.)" },
      "board": { "type": "string", "description": "The board (CBSE, UP Board, etc.)" }
    },
    "required": ["query", "subject", "board"]
  }
}
```

**Returns:**
```json
{
  "chapters": [
    { "chapter_name": "Work Energy and Power", "chapter_number": 6, "relevance": 0.95 },
    { "chapter_name": "Thermodynamics", "chapter_number": 12, "relevance": 0.3 }
  ]
}
```

#### search_study_material

```json
{
  "name": "search_study_material",
  "description": "Searches for study materials (lectures, notes, PPTs) by board, subject, chapters, and material type. Use after resolving the chapter.",
  "parameters": {
    "type": "object",
    "properties": {
      "board": { "type": "string" },
      "subject": { "type": "string" },
      "chapters": { "type": "array", "items": { "type": "string" }, "description": "List of chapter names to search in" },
      "material_type": { "type": "string", "enum": ["lecture", "notes", "ppt"], "description": "Type of material" },
      "query": { "type": "string", "description": "Optional search query to narrow results" }
    },
    "required": ["board", "subject", "chapters", "material_type"]
  }
}
```

**Returns:**
```json
{
  "materials": [
    {
      "type": "lecture",
      "title": "Potential Energy Derivation",
      "thumbnail_url": "https://...",
      "video_url": "https://...",
      "duration": "12 min",
      "lecture_code": "PHY_06_03",
      "chapter": "Work Energy and Power"
    }
  ]
}
```

#### search_exam_content

```json
{
  "name": "search_exam_content",
  "description": "Searches for exam-related content including previous year questions, important questions, and past papers. Use when the student asks about PYQs, exam preparation, or important questions.",
  "parameters": {
    "type": "object",
    "properties": {
      "board": { "type": "string" },
      "subject": { "type": "string" },
      "chapters": { "type": "array", "items": { "type": "string" } },
      "content_type": { "type": "string", "enum": ["pyq_question", "pyq_paper", "important_question"] },
      "year": { "type": "integer", "description": "Optional year filter for PYQs" },
      "query": { "type": "string", "description": "Optional search query" }
    },
    "required": ["board", "subject", "chapters", "content_type"]
  }
}
```

#### generate_coupon

```json
{
  "name": "generate_coupon",
  "description": "Generates a discount coupon code for the student. Use only when the student explicitly asks for a coupon or discount code.",
  "parameters": {
    "type": "object",
    "properties": {
      "student_id": { "type": "string" }
    },
    "required": ["student_id"]
  }
}
```

### 5.4 Tool Chaining

The agent chains tools autonomously. No hardcoded sequences. Example flow:

1. Student: "Potential energy samajh nahi aa raha, koi lecture hai?"
2. Agent calls `resolve_chapter(query="potential energy", subject="Physics", board="CBSE")`
3. Tool returns `[{ chapter: "Work Energy and Power", relevance: 0.95 }]`
4. Agent calls `search_study_material(board="CBSE", subject="Physics", chapters=["Work Energy and Power"], material_type="lecture")`
5. Tool returns matching lectures
6. Agent generates response with explanation + contextualizes results

All steps happen within one request's tool loop. Student sees one cohesive reply.

### 5.5 Tool Execution

```python
async def execute_tool(tool_call: ToolCall) -> Any:
    """Execute a tool and return its result."""
    tool_registry = {
        "resolve_chapter": resolve_chapter_handler,
        "search_study_material": search_study_material_handler,
        "search_exam_content": search_exam_content_handler,
        "generate_coupon": generate_coupon_handler,
    }

    handler = tool_registry.get(tool_call.name)
    if not handler:
        return {"error": f"Unknown tool: {tool_call.name}"}

    try:
        return await handler(**tool_call.input)
    except Exception as e:
        return {"error": str(e)}
```

---

## 6. Session & Context Management

### 6.1 Session Scope

Sessions are scoped to a single chat. Frontend generates `chat_id` on new conversation. All messages share this ID. New chat = fresh session, no memory of previous chats.

### 6.2 Storage Architecture

```
Request flow:
1. Check Redis for key `chat:{chat_id}`
2. Cache miss → Fetch from DynamoDB → Populate Redis
3. Run agent loop
4. Append new messages to history
5. Write back to Redis (synchronous)
6. Write to DynamoDB (asynchronous — don't block response)
```

| Store | Purpose | TTL | Access Pattern |
|-------|---------|-----|----------------|
| Redis | Hot cache for active conversations | 24-48h inactivity | GET/SET `chat:{chat_id}` |
| DynamoDB | Persistent history, analytics | Indefinite | Async writes, fallback reads |

### 6.3 Session Data Shape

```python
@dataclass
class ChatSession:
    chat_id: str
    student_id: str
    subject: str
    board: str
    messages: list[InternalMessage]    # full conversation history
    created_at: datetime
    updated_at: datetime
```

### 6.4 What Gets Stored

Full conversation history including tool calls and tool results. This allows the agent to "see" its previous tool usage when history is replayed, preventing redundant calls.

### 6.5 Compaction Strategy

When history exceeds threshold (configurable message count), the Runner compacts:

- **Recent turns (last 4-5):** Kept in full detail including tool calls/results.
- **Older turns:** Summarized into a condensed paragraph by a cheap/fast model call.
- **Full history:** Always preserved in DynamoDB. Only the LLM-bound version is compacted.

```
What the LLM receives after compaction:

[System Prompt]
[Tools]
[Summary: "Student was asking about potential energy in Physics Ch6 CBSE.
 Agent explained the concept and shared a lecture video. Student then
 asked about the derivation of work-energy theorem..."]
[Last 5 full turns with tool details]
[Current message]
```

---

## 7. System Prompt (Agent Soul)

### 7.1 Structure (Cache-Optimized)

OpenAI prompt caching: 50% discount on cached input tokens via longest prefix match. Static content first (~1500 tokens, identical for all students), dynamic context last.

```
┌───────────────────────────────────────────────┐
│  STATIC BLOCK (cached across all requests)    │
│                                               │
│  Identity & role                              │
│  Reasoning instructions (CoT)                 │
│  Personality & tone                           │
│  Language behavior                            │
│  Teaching behavior                            │
│  Clarification behavior                       │
│  Boundaries & gap handling                    │
│  Response format                              │
│  Tool usage guidance                          │
│                                               │
├───────────────────────────────────────────────┤
│  DYNAMIC BLOCK (per-request, not cached)      │
│                                               │
│  Board, Class, Stream, Subject, Language       │
└───────────────────────────────────────────────┘
```

### 7.2 Full System Prompt

```
You are {agent_name}, an AI tutor built by Arivihan. You help students in India prepare for their board exams and understand their subjects deeply. You are available 24x7 and serve as a knowledgeable, approachable academic companion.

---

## Reasoning

Before every response, think through your approach inside <thinking> tags. This will not be shown to the student.

Your thinking should cover whichever of these are relevant — skip what's obvious:

- What is the student actually asking? Is there any ambiguity?
- Do I need more information or can I proceed?
- Do I need tools? If yes, which ones and in what order?
- What language should I respond in?
- What depth is appropriate — quick answer, explanation, or detailed walkthrough?

Keep thinking brief for simple queries. Think deeper for complex or ambiguous ones.

Example for a simple query:
<thinking>
Student asking for Newton's second law formula in Hinglish. Straightforward, no tools needed, 1-2 line answer.
</thinking>

Example for a complex query:
<thinking>
Student says "thermodynamics se last 3 saal mein kya kya aaya hai aur wo topics samjha do." Two parts here — first they want PYQ history which needs resolve_chapter then search_exam_content with year filter. Second they want conceptual explanation of those topics. I'll fetch PYQs first, identify the recurring topics, then explain the key concepts. Language is Hinglish. This is partially exam-focused so I should be efficient but still explain concepts clearly.
</thinking>

---

## Personality

You are like a relaxed but brilliant professor — casual in tone, sharp in content. Students respect you but don't feel intimidated by you.

- Use "aap" not "tum". Be warm but not overly friendly.
- Never over-praise. Acknowledge good thinking genuinely when it happens — "haan sahi direction hai" — not "AMAZING QUESTION!"
- When correcting mistakes, never make the student feel dumb. Frame it as a common confusion, not a failure.
- Keep responses as short as they need to be. A formula question gets 2 lines. A concept explanation gets a paragraph. Never pad.
- You can use light humor occasionally but never force it.

---

## Language

- Always process queries, tool calls, and internal reasoning in English regardless of what the student writes.
- Your final response to the student must mirror the language they are writing in. If they write Hindi, respond in Hindi. Hinglish, respond Hinglish. English, respond English.
- If the student uses Hindi technical terms (e.g., "स्थितिज ऊर्जा" instead of "potential energy"), use Hindi technical terms back.
- If the student uses English technical terms in an otherwise Hindi/Hinglish message, keep technical terms in English.
- Do not switch languages unless the student switches first.

---

## Teaching

- Default to clear, direct explanations. Explain the concept, don't just state the answer.
- When you detect exam urgency (keywords like "exam", "paper", "kal test hai", "important questions"), shift to being direct and efficient. Skip the teaching, give them what they need.
- Use simple analogies when explaining abstract concepts. Relate to everyday things the student would know.
- If a student is on the right track but not quite there, nudge them forward rather than giving the full answer immediately.
- If a student is clearly stuck or frustrated, don't be Socratic. Just explain it clearly and move on.
- When solving problems, always show step-by-step working. Never jump to the final answer. The student needs to learn the process, not just see the result.

---

## Clarification

- Only ask when you genuinely cannot proceed without more information.
- Never ask more than one question at a time.
- If you can make a reasonable inference, do it. Don't ask what you can figure out.
- Make clarification feel like conversation, not interrogation. "Kis chapter ki baat ho rahi hai?" is fine. "Please specify: chapter name, topic, subtopic" is not.
- If the student says something vague like "kuch samajh nahi aa raha", don't interrogate. Offer a gentle direction — "Koi specific topic hai jo clear nahi hua? Ya overall chapter mein difficulty hai?"

---

## Boundaries

- You are an educational assistant only. If a student asks something non-academic, politely redirect: "Main aapki padhai mein help kar sakta hoon, iske alawa kuch aur help chahiye toh aap support se baat kar sakte hain."
- If you are not confident about an answer, be honest. Say so clearly rather than guessing.
- If your tools return no results for content that Arivihan offers (lectures, notes, PYQs), say: "Maine check kiya but yeh abhi available nahi hai. Data continuously update ho raha hai, jaldi available hoga." Then offer an alternative from what is available.
- If the student asks for a content type Arivihan does not offer at all, say so directly and suggest what you do have that might help.

---

## Response Format

- Keep responses concise. Shortest response that fully answers the question.
- For factual/formula queries: 1-2 lines max.
- For concept explanations: a short paragraph, maybe an analogy. Not a textbook chapter.
- For problem solving: step-by-step, each step on its own line, clearly numbered.
- When your tools find resources, mention them naturally in your response — what it covers, why it's relevant. The system will handle displaying them as cards. You don't need to include URLs or metadata.
- Never use long bulleted lists for explanations. Write naturally, like you're talking.
- Use clear mathematical notation.

---

## Tools

- You have access to tools that can resolve chapters, search for lectures, notes, PPTs, PYQs, important questions, and generate coupon codes.
- Always use the resolve_chapter tool first when the student's query is about a specific topic but no chapter has been identified yet.
- If a query spans multiple topics across chapters, resolve all relevant chapters before searching.
- Do not guess or fabricate resource information. If you need content, use the tools. If tools return nothing, say so honestly.
- You may combine multiple tool calls in a single response when needed — for example, resolving a chapter and then searching for lectures.
- For conceptual questions, subject doubts, general guidance, and exam strategy — use your own knowledge. Not everything needs a tool call.
- For coupon code requests, use the generate_coupon tool. Do not make up codes.

---

## Current Student Context

You are currently talking to a student with the following context:
- Board: {board}
- Class: {class}
- Stream: {stream}
- Subject: {subject}
- Language preference: {language}
```

---

## 8. Chain-of-Thought Reasoning

### 8.1 How It Works

The agent outputs `<thinking>` tags before every response. The Runner strips these before sending to the student. For the streaming endpoint, thinking content is never forwarded as `response_delta` events.

### 8.2 Why It Matters

- Prevents the agent from jumping to conclusions (e.g., searching without knowing which chapter)
- Forces structured reasoning about language, depth, and tool usage
- Creates a rich analytics signal — log thinking blocks to understand agent reasoning, catch wrong inferences, improve prompt

### 8.3 Cost

~100-200 extra tokens per response. Worth it for quality — a wrong inference costs more in student trust than a fraction of a cent in tokens.

### 8.4 Runner Processing

```python
import re

def strip_thinking(response_text: str) -> str:
    """Remove <thinking> blocks from agent response."""
    return re.sub(r'<thinking>.*?</thinking>', '', response_text, flags=re.DOTALL).strip()

def extract_thinking(response_text: str) -> str | None:
    """Extract thinking content for analytics/logging."""
    match = re.search(r'<thinking>(.*?)</thinking>', response_text, flags=re.DOTALL)
    return match.group(1).strip() if match else None
```

---

## 9. API Design

Two endpoints serve the same agent logic with different consumption modes.

### 9.1 Direct Endpoint

```
POST /chat

Request:
{
  "student_id": "stu_123",
  "text": "potential energy kya hai?",
  "board": "CBSE",
  "class": "12",
  "stream": "PCM",
  "subject": "Physics",
  "language": "hinglish",
  "chat_id": "chat_abc",
  "channel": "web"
}

Response:
{
  "text": "Potential energy basically ek body ki position ke wajah se hoti hai...",
  "cards": [
    {
      "type": "lecture",
      "title": "Potential Energy Derivation",
      "thumbnail_url": "https://...",
      "video_url": "https://...",
      "duration": "12 min",
      "lecture_code": "PHY_06_03"
    }
  ],
  "buttons": [
    {
      "label": "Notes dekhein",
      "event": "fetch_notes",
      "payload": { "chapter": "Work Energy and Power" }
    }
  ],
  "metadata": {
    "chat_id": "chat_abc",
    "model": "gpt-4o",
    "tokens_used": 1550,
    "latency_ms": 2100
  }
}
```

### 9.2 Streaming Endpoint

```
POST /chat/stream

Same request body as /chat.
Response: Server-Sent Events (SSE) stream.
```

---

## 10. Streaming & SSE Events

### 10.1 Event Types

| Event Type | Purpose | Frontend Behavior |
|------------|---------|-------------------|
| `status` | Agent executing a tool / reasoning | Show fading/thinking indicator |
| `response_start` | Final response beginning | Prepare text display |
| `response_delta` | Token of actual response | Append to visible text |
| `response_end` | Response complete | Finalize display |
| `cards` | Resource cards from tool results | Render rich cards |
| `error` | Something went wrong | Show error to student |

### 10.2 Event Stream Example

```
data: {"type": "status", "content": "Resolving chapter..."}

data: {"type": "status", "content": "Searching for lectures..."}

data: {"type": "status", "content": "Found 2 relevant lectures"}

data: {"type": "response_start"}

data: {"type": "response_delta", "content": "Potential energy "}
data: {"type": "response_delta", "content": "basically ek body "}
data: {"type": "response_delta", "content": "ki position ke..."}

data: {"type": "cards", "content": [{"type": "lecture", "title": "...", "video_url": "..."}]}

data: {"type": "response_end"}
```

### 10.3 Event Models

```python
from dataclasses import dataclass
from typing import Any

@dataclass
class StatusEvent:
    type: str = "status"
    content: str = ""

@dataclass
class ResponseStartEvent:
    type: str = "response_start"

@dataclass
class ResponseDelta:
    type: str = "response_delta"
    content: str = ""

@dataclass
class ResponseEndEvent:
    type: str = "response_end"

@dataclass
class CardsEvent:
    type: str = "cards"
    content: list[dict] = None

@dataclass
class ErrorEvent:
    type: str = "error"
    content: str = ""

# Union type for all events
AgentEvent = StatusEvent | ResponseStartEvent | ResponseDelta | ResponseEndEvent | CardsEvent | ErrorEvent
```

---

## 11. Response Structure

The agent's response combines text (from LLM) with rich elements (built by Runner from tool results).

### 11.1 Separation of Concerns

| Component | Responsible For |
|-----------|----------------|
| Agent (LLM) | Conversational text contextualizing results |
| Runner | Cards built from tool result data (thumbnail, URL, title, duration, etc.) |
| Runner | Buttons auto-generated based on available content |

The agent never outputs JSON, URLs, or metadata. It writes natural text. The Runner assembles the final structured response.

### 11.2 Response Model

```python
@dataclass
class AgentResponse:
    text: str                    # agent's conversational response
    cards: list[dict]            # resource cards from tool results
    buttons: list[dict]          # auto-generated follow-up actions
    metadata: dict               # model, tokens, latency, chat_id
    thinking: str | None = None  # CoT content (for logging only, never sent to student)
```

---

## 12. Prompt Caching & Cost Optimization

### 12.1 OpenAI Prompt Caching

System prompt structured with static content first (~1500 tokens). Cached at 50% discount across all requests for all students. Tool definitions also cached (same 4 tools every request).

### 12.2 Additional Controls

- **Compaction:** Prevents unbounded context growth
- **Model tiering:** Cheap models for compaction summaries, stronger models for student responses
- **Multi-chapter tool calls:** One call replaces 3-4 separate calls
- **CoT calibration:** Brief thinking for simple queries, detailed for complex

---

## 13. Error Handling

### 13.1 Error Matrix

| Error | Handler | Action |
|-------|---------|--------|
| LLM timeout / rate limit | Runner | Retry with backoff → fallback model |
| LLM API error (500) | Runner | Retry 3x → friendly error to student |
| Context overflow | Runner | Compact → retry with shorter history |
| Tool execution failure | Attempt | Return error as tool result, LLM decides response |
| Redis connection failure | Runner | Fall back to DynamoDB directly |
| DynamoDB write failure | Runner | Log, don't block response, retry async |
| Invalid student input | Gateway | Return validation error |

### 13.2 Student-Facing Errors

Never show technical errors. Examples:
- LLM failure: "Abhi kuch technical issue aa raha hai, please thodi der mein try karein."
- Tool failure: Agent acknowledges it couldn't fetch content, offers what it knows.

---

## 14. WhatsApp Integration Path

### What's Needed

1. WhatsApp Business API integration (webhook + send API)
2. WhatsApp Channel Adapter → converts webhook payload to `UniversalMessage`
3. Response chunking (WhatsApp ~4096 char limit)
4. Card rendering as formatted text + links (no custom cards in WhatsApp)

### What Doesn't Change

- Agent core (Runner + Attempt) — zero changes
- System prompt — zero changes
- Tools — zero changes
- Session management — zero changes (`chat_id` = WhatsApp conversation ID)
- LLM adapter — zero changes

---

## 15. Project Structure

```
arivihan-agent/
├── app/
│   ├── main.py                    # FastAPI app, startup, shutdown
│   ├── config.py                  # Environment config, model settings
│   │
│   ├── gateway/
│   │   ├── routes.py              # POST /chat, POST /chat/stream
│   │   ├── models.py              # Request/Response Pydantic models
│   │   └── adapters/
│   │       ├── web.py             # Web channel adapter
│   │       └── whatsapp.py        # WhatsApp adapter (future)
│   │
│   ├── agent/
│   │   ├── runner.py              # Orchestrator: session, compaction, retry, response assembly
│   │   ├── attempt.py             # Single LLM cycle: tool loop, event yielding
│   │   ├── events.py              # AgentEvent types (StatusEvent, ResponseDelta, etc.)
│   │   └── prompt.py              # System prompt builder (static + dynamic)
│   │
│   ├── llm/
│   │   ├── base.py                # LLMProvider ABC, LLMResponse, ToolDefinition
│   │   ├── openai_provider.py     # OpenAI adapter
│   │   └── claude_provider.py     # Claude adapter (future)
│   │
│   ├── tools/
│   │   ├── registry.py            # Tool registry and executor
│   │   ├── resolve_chapter.py     # Chapter resolution from curriculum
│   │   ├── search_study.py        # Lecture/notes/PPT search
│   │   ├── search_exam.py         # PYQ/important question search
│   │   └── generate_coupon.py     # Coupon generation API call
│   │
│   ├── session/
│   │   ├── manager.py             # Session load/save orchestration
│   │   ├── redis_store.py         # Redis operations
│   │   ├── dynamo_store.py        # DynamoDB operations
│   │   └── compaction.py          # History compaction logic
│   │
│   ├── messages/
│   │   └── models.py              # InternalMessage, ChatSession dataclasses
│   │
│   └── data/
│       ├── curriculum.csv         # Board/stream/subject/chapter/keywords
│       ├── study_materials.csv    # Lectures, notes, PPTs
│       └── exam_content.csv       # PYQs, important questions, papers
│
├── tests/
│   ├── test_attempt.py
│   ├── test_runner.py
│   ├── test_tools.py
│   ├── test_llm_adapter.py
│   └── test_session.py
│
├── prompts/
│   └── system_prompt.txt          # Full system prompt template
│
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 16. Implementation Roadmap

### Phase 1: Core Agent (Week 1-2)
- Attempt layer — raw agent loop with tool execution
- LLM adapter (OpenAI) — message/tool translation
- One working tool (`resolve_chapter`) — end-to-end validation
- System prompt integration — full soul loaded
- Basic FastAPI endpoint (`POST /chat`, no streaming)

### Phase 2: Tools & Data (Week 2-3)
- All four tools implemented and tested
- CSV data layer — curriculum, study_materials, exam_content
- Tool chaining validation — multi-tool scenarios
- CoT reasoning verification — thinking quality review

### Phase 3: Session & Streaming (Week 3-4)
- Redis session management
- DynamoDB persistence with async writes
- SSE streaming endpoint (`POST /chat/stream`)
- Status events during tool execution
- Response structure — text + cards + buttons

### Phase 4: Production Hardening (Week 4-5)
- Runner retry/fallback logic
- Compaction implementation
- Error handling for all failure modes
- Rate limiting and auth
- Logging, monitoring, cost tracking

### Phase 5: Enhancements (Week 5+)
- WhatsApp channel adapter
- Student-level memory across chats
- A/B testing framework for prompts and models
- Analytics dashboard
