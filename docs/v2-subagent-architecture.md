# Arivihan Agent v2 — Subagent Architecture

> **Status:** Planning | **Last updated:** 2026-03-12

---

## 1. High-Level Architecture

Multi-agent system with a **Teacher Agent** as the orchestrator and three specialized subagents. The teacher owns all student-facing conversation, context resolution, and clarification. Subagents are pure English-only workers.

```
Student message + user_profile + user_stats + user_memory
                    ↓
┌─────────────────────────────────────────────────────┐
│  PARALLEL on arrival                                │
│  ├─ Session load (Redis → DynamoDB fallback)        │
│  ├─ User profile + stats + memory load              │
│  └─ Speculative chapter resolution (if keywords)    │
└────────────────────┬────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────┐
│  Teacher Agent (LLM call)                           │
│  Context: profile + stats + memory + session        │
│  Tools: resolve_chapter                             │
│                                                     │
│  Outputs (structured JSON):                         │
│  ├─ intent classification                           │
│  ├─ resolved chapter (if applicable)                │
│  ├─ subagent dispatch list (or direct response)     │
│  ├─ per-subagent input context (English)            │
│  ├─ per-subagent nudge message (student's language) │
│  └─ output language for text subagents              │
└────────────────────┬────────────────────────────────┘
                     ↓
          ┌──── General? ────┐
          │ YES              │ NO
          ↓                  ↓
   Return teacher's    ┌────┴────────────────┐
   direct response     │  PARALLEL subagents │
                       ├─ Doubt Agent        │
                       ├─ Content Agent      │
                       └─ Guidance Agent     │
                       └─────────┬───────────┘
                                 ↓
                    Programmatic assembly
                                 ↓
                    Stream to student
                                 ↓
                    Background: save session +
                    extract memories + update stats
```

---

## 2. Teacher Agent (Orchestrator)

The teacher is the single entry point. It talks to the student, owns the conversation, and dispatches work.

### Responsibilities
- Understands Hindi / English / Hinglish input
- Resolves chapter (has `resolve_chapter` tool)
- Checks context sufficiency — asks clarification if missing info
- Routes to subagents with clean, pre-resolved English context
- Handles general conversation directly (greetings, motivation, off-topic)
- Assembles subagent results programmatically (no final LLM call)
- Generates contextual status nudges in student's language

### Tools (teacher only)
- `resolve_chapter` — maps topic keywords to curriculum chapter

### Routing decisions

| Intent | Handler |
|---|---|
| Content request (notes, PYQs, lectures, tests) | Content Agent |
| Academic doubt (explain, samjhao, kya hota hai) | Doubt Agent |
| Guidance / study advice / app navigation | Guidance Agent |
| General / greeting / motivation / off-topic | Teacher responds directly |
| Mixed (explain + fetch content) | Multiple subagents in parallel |

### Structured output format
```json
{
    "type": "dispatch | direct_response",
    "direct_response": "...",
    "chapter_id": 12,
    "chapter_name": "Electrostatic Potential and Capacitance",
    "subagents": [
        {
            "agent": "doubt",
            "input": "explain dipole moment concept",
            "language": "hindi",
            "nudge": "Dipole ka concept samjha raha hoon..."
        },
        {
            "agent": "content",
            "input": "fetch PYQ papers",
            "content_types": ["pyq"],
            "chapter_id": 12,
            "nudge": "PYQ papers dhoondh raha hoon..."
        }
    ]
}
```

---

## 3. Subagents

### 3.1 Content Research Agent

**Purpose:** Fetches learning materials from CSV data sources.

**Tools:**
- `search_lectures` — video microlectures
- `search_topper_notes` — topper-written notes PDFs
- `search_ppt_notes` — PowerPoint notes PDFs
- `search_pyq_papers` — previous year question papers
- `search_important_questions` — important questions with filtering
- `search_tests` — chapterwise and full-length practice tests
- `search_ncert_solutions` — NCERT solutions (Maths only)

**Receives:** Pre-resolved chapter_id, content types needed, language for filtering.
**Returns:** Structured resource cards (no natural language needed).
**Model:** Fast/small model (just tool selection, no explanation needed).
**Parallel tool execution:** All tool calls within content agent run via `asyncio.gather()`.

### 3.2 Doubt Solver Agent

**Purpose:** Explains academic concepts clearly.

**Tools:** None — pure LLM reasoning.
**Receives:** Topic/concept to explain in English, output language.
**Returns:** Explanation text in student's language.
**Model:** Primary/strong model (needs deep reasoning for explanations).
**Key behavior:** Responds in student's language but reasons internally in English.

### 3.3 Guidance Agent

**Purpose:** Two types of guidance:
1. **Academic guidance** — study strategies, learning difficulties, exam preparation tips
2. **App navigation** — directing students to app features/screens

**Tools:**
- `get_app_screen_info` — returns deep link / activity info for app screens (TBD)

**Receives:** Guidance query in English, output language.
**Returns:** Advice text in student's language + optional navigation buttons.
**Model:** Primary model for academic guidance, fast model for app nav.

---

## 4. Subagent Contract

Every subagent returns the same structured format:

```python
@dataclass
class SubAgentResult:
    status: str              # "success" | "partial" | "need_clarification" | "error"
    text: str | None         # explanation text (doubt/guidance agents)
    cards: list[dict]        # resource cards (content agent)
    clarification: str|None  # question to ask student (shouldn't happen — teacher pre-checks)
    metadata: dict           # timing, model used, tokens, etc.
```

---

## 5. Language Architecture

**Single language boundary at the teacher level.**

| Layer | Language |
|---|---|
| Student input | Hindi / English / Hinglish |
| Teacher processing | Understands all, extracts intent in English |
| Subagent inputs | English |
| Subagent internal reasoning | English |
| Subagent text output | Student's language (passed as parameter) |
| Content agent output | Structured data (language-agnostic) |
| Final response to student | Student's language |

Benefits:
- LLMs faster and more accurate in English
- Fewer tokens (Hindi/Hinglish uses 2-3x more tokens in tokenizers)
- Tool/chapter matching more reliable in English
- Only one component needs multilingual capability

---

## 6. What the Teacher Owns (not subagents)

These responsibilities stay with the teacher to avoid wasted subagent calls:

| Responsibility | Why teacher owns it |
|---|---|
| `resolve_chapter` | Cross-cutting — both content and doubt agents need it. Resolve once, share. |
| Context sufficiency check | If chapter/topic missing, ask before dispatching. Saves wasted LLM calls. |
| Clarification questions | Teacher owns the conversation. Consistent UX, single point of dialogue. |
| General conversation | No tools needed, no specialized reasoning. Teacher handles directly. |

---

## 7. Runner Evolution

The current `runner.py` evolves into the teacher agent orchestrator:

```
Current:
  runner.py  →  run_attempt()  →  tool loop  →  response

New:
  runner.py (= teacher orchestrator)
    → parallel: session load + user data load
    → teacher LLM call (resolve_chapter + routing)
    → if general: return teacher's response directly
    → if subagent needed: dispatch in parallel via asyncio.gather()
    → programmatic assembly of SubAgentResults
    → stream to student
    → background: save session + extract memories + update stats

  attempt.py stays as reusable LLM call + tool loop primitive
    → teacher uses it (for resolve_chapter tool loop)
    → content agent uses it (for search tools)
    → doubt/guidance use simpler single LLM call (no tools or minimal tools)
```

---

## 8. Status Nudges / Thinking State

Contextual, language-aware status messages streamed to the student via SSE while subagents work.

**Generated by the teacher** as part of its routing output (no extra LLM call):

```
Student: "Dipole samjhao aur PYQ bhi do"

→ "Samajh raha hoon kya chahiye..."          (teacher routing)
→ "Dipole ka concept samjha raha hoon..."    (doubt agent starts)
→ "PYQ papers dhoondh raha hoon..."          (content agent starts)
→ [response starts streaming]
```

Emitted as `StatusEvent` via SSE. Frontend shows these as animated thinking indicators.

---

## 9. Failproof Design

### Tool failure handling

| Scenario | Behavior |
|---|---|
| Tool returns no results | Content agent returns structured "no results for X" → teacher tells student what IS available |
| Chapter misspelled | Fuzzy matching in resolve_chapter (Levenshtein / token overlap). If no match, suggest closest 3 chapters. |
| Topic not in curriculum | Teacher says "Yeh topic curriculum mein nahi mila. Chapter ka naam batao." |
| Content type unavailable for chapter | Content agent returns partial results. Teacher says "PYQs nahi hain, but notes aur lectures hain." |
| LLM API timeout/error | Timeout per subagent (~8s). If one fails, return what others produced. |
| All subagents fail | Cached template: "Abhi kuch technical issue aa raha hai, thodi der mein try karo." |
| Redis down | Falls back to DynamoDB (already implemented) |

### Clarification before wasted work
Teacher checks context sufficiency BEFORE dispatching. If chapter/topic missing and can't be inferred from session history → ask one clarifying question. Never more than one question at a time.

---

## 10. Speed Optimizations

| Optimization | Impact | Status |
|---|---|---|
| Parallel subagent dispatch (`asyncio.gather`) | -1-3s | Planned |
| Parallel tool calls within content agent | -500ms-1s | Planned |
| Speculative chapter pre-resolution on arrival | -200-300ms | Planned |
| Fast/small model for routing + content agent | -300-500ms | Planned |
| No final formatting LLM call (programmatic assembly) | -500ms-1s | Planned |
| Stream doubt solver immediately while content loads | Perceived latency near zero | Planned |
| In-memory CSV cache (load once at startup) | -100-200ms per tool call | Planned |
| Connection pooling (reuse HTTP connections to LLM) | -50-100ms per call | Planned |
| Shorter focused prompts per subagent (fewer input tokens) | Faster TTFT | Planned |
| Max tokens cap per subagent | Fewer output tokens | Planned |

**Target:** 5-8s current → 1.5-3s with all optimizations.

---

## 11. User Data System

### Layer 1: User Profile (static, from backend)
```
- name, class, board, subjects
- exam date / exam type
- subscription tier
- language preference
- app join date
```

### Layer 2: User Stats (computed, updated per session)
```
- chapters covered (per subject)
- study streak (consecutive days active)
- total study sessions
- subjects studied today
- weak areas (topics with repeated doubts)
- strong areas (topics covered quickly)
- last active timestamp
```

### Layer 3: User Memory (agent-learned facts)
```
- "Struggles with integration by parts"
- "Prefers examples over theory"
- "Preparing for JEE alongside boards"
- "Gets confused between velocity and acceleration"
```

All three layers injected into teacher's context on every request.

---

## 12. Model Strategy

| LLM Call | Model | Why |
|---|---|---|
| Teacher routing + chapter resolution | Fast model | Simple classification + tool call, speed critical |
| Doubt solver | Primary model | Deep reasoning for concept explanations |
| Content agent | Fast model | Just tool selection, no explanation |
| Guidance agent | Primary model | Nuanced advice needed |
| Memory extraction (background) | Fast model | Simple fact extraction |

---

## 13. "Damn" Features (Personalization)

With user profile + stats + memory in teacher's context:

- **Session continuity:** "Kal tumne Current Electricity padha tha. Aaj Magnetism pe chalein?"
- **Weak spot detection:** Repeated doubts on same concept → "Electric Field mein aur practice chahiye. Extra questions solve karte hain?"
- **Exam countdown:** "Board exam mein 45 din bache hain. Is chapter ka weightage 8 marks hai."
- **Smart follow-ups:** After explaining → "Samajh aa gaya? Ek quick question solve karke dekhte hain."
- **Study pattern awareness:** "Aaj 3 subjects padhe. Solid session!"
- **Connected concepts:** "Yeh formula Kinematics mein bhi dekha tha, yaad hai?"
- **Mood-aware tone:** Frustrated student → simpler language, more examples, more encouragement.

---

## 14. Smart Follow-ups

### Philosophy
Keep the student hooked by suggesting the logical next step after most responses. Not every response (not 100%) — only when there's a genuine next step that helps the student.

### When to include follow-ups

| Scenario | Follow-up? |
|---|---|
| After concept explanation | YES — suggest test/lecture/notes |
| After content delivery (notes/lectures/PYQs) | YES — suggest related content |
| After guidance | SOMETIMES — only if actionable |
| After greeting / general chat | NO |
| After clarification question | NO |
| After error / no results | NO |

Natural hit rate: ~70-80% of responses.

### Content mapping (what was served → what to suggest)

| Served | Follow-up suggests |
|---|---|
| Concept explanation | Test / lecture / notes |
| Lectures | Notes / important questions |
| Notes | Lectures / important questions / test |
| PYQ papers | More years / important questions |
| Important questions | PYQ papers / test |
| Tests | "Doubt ho toh pooch lo" |
| NCERT solutions | Important questions / PYQ |
| Explanation + content | Whatever content type wasn't included |

### Implementation

Generated by the teacher in its routing LLM call — **no extra LLM call needed**. The teacher knows the intent, the chapter, and what subagents are being dispatched. That's enough to craft a contextual follow-up.

Part of teacher's structured routing output:
```json
{
    "subagents": [...],
    "follow_up": "Samajh aa gaya? Test try karo ya lecture se aur clear ho jaayega."
}
```

**Text-only delivery:** Follow-up text appended at the end of the response. Student types their next request naturally — session context already has the chapter/topic, so the teacher handles continuity automatically.

No buttons for now. Keeps the UX conversational and avoids frontend complexity.

### Example flows

```
Student: "Dipole samajh nahi aa raha"
Agent: [dipole explanation]
       "Agar samajh aa gaya toh test try karo, ya lecture se aur clear ho jaayega."

Student: "Ray optics ke lectures do"
Agent: [lecture cards]
       "Notes bhi dekhoge? Revise karne mein help karenge."

Student: "Hi bhai"
Agent: "Hello! Aaj kya padh rahe ho?"
       (no follow-up — natural conversation)
```

---

## 15. Implementation Plan

### Phase 1: Subagent Foundation
- [ ] Define `SubAgentResult` dataclass
- [ ] Create subagent base class / interface
- [ ] Implement Doubt Agent (simplest — no tools)
- [ ] Implement Content Agent (move existing tools)
- [ ] Implement Guidance Agent (basic, no app nav tool yet)

### Phase 2: Teacher Agent
- [ ] Refactor `runner.py` into teacher orchestrator
- [ ] Teacher LLM call with structured output (routing + chapter resolution)
- [ ] Parallel subagent dispatch via `asyncio.gather()`
- [ ] Programmatic response assembly
- [ ] General conversation handling (no subagent)

### Phase 3: Language Boundary
- [ ] Teacher extracts English intent from Hindi/Hinglish
- [ ] Subagent prompts are English-only
- [ ] Text-generating subagents receive output language parameter

### Phase 4: Speed
- [ ] Parallel tool execution within content agent
- [ ] In-memory CSV caching at startup
- [ ] Connection pooling for LLM HTTP client
- [ ] Model tiering (fast model for routing + content, primary for doubt + guidance)
- [ ] Max tokens caps per subagent

### Phase 5: User Data & Personalization
- [ ] User profile loading
- [ ] User stats tracking and injection
- [ ] User memory extraction and storage
- [ ] Personalized teacher responses

### Phase 6: Status Nudges & Streaming
- [ ] Contextual nudge generation in teacher's routing output
- [ ] SSE streaming of nudges during subagent work
- [ ] Stream doubt solver text while content agent loads

### Phase 7: Resilience
- [ ] Subagent timeout handling
- [ ] Partial result assembly (if one subagent fails)
- [ ] Fuzzy chapter matching in resolve_chapter
- [ ] Clarification flow for insufficient context
- [ ] Fallback model on LLM failure
