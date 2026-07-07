# LibraryMind — Reflection

## Overview

LibraryMind is an AI-powered FastAPI backend for a public library: natural-language catalogue search, RAG-grounded Q&A, a multi-turn chatbot, support-ticket classification, and review summarisation, all built on a multi-provider AI layer with automatic fallback.

---

## Key Design Decisions

**Layered architecture.** API, Service, AI Provider, and Infrastructure layers, each depending only on the layer below. This kept every piece independently testable and made "what belongs where" an easy call throughout the build.

**Resilient multi-provider fallback.** A `BaseProvider` abstract class exposes one `generate()` interface; `ResilientAIService` tries OpenAI first and falls through to Anthropic on any transient failure. Retry logic (exponential backoff, only on 429/5xx/timeouts) lives in one place instead of being scattered across every service.

**RAG with explicit refusal on low relevance.** The pipeline embeds the question, searches ChromaDB, filters by cosine similarity, and only calls the AI if something clears the threshold — otherwise it returns a refusal with no AI call at all. The model is structurally unable to invent a book it was never shown.

**Singleton rate limiter.** One shared token bucket (`get_rate_limiter()`) across RAG, chatbot, classification, and summarisation, instead of one bucket per service silently multiplying the effective limit.

**Metadata-filter router for structured questions.** Dense similarity search can't answer "what books came out before 2000?" — no book *description* is semantically close to a date range, because the answer lives in a structured field, not free text. `RAGEngine` runs a cheap regex/keyword check first, and only when a question is clearly structured (a year range, decade, or an enumeration phrase like "how many"/"list all") does it skip embeddings and filter the catalogue's metadata directly. Otherwise it falls through to the unchanged semantic path.

---

## Challenges Faced

**Cosine distance vs. cosine similarity — the debugging story.** ChromaDB returns cosine *distance* (0 = identical), not similarity, and my first version compared raw distances against a `0.7` threshold — backwards. The RAG engine silently refused every query, and I initially suspected broken embeddings. Logging the raw distance values made it obvious: they clustered around `0.5`, meaning perfectly good matches were being read as bad ones. The fix was one line, `similarity = 1 - distance`, but the real lesson was to log intermediate values at every pipeline stage during development, not just at the input/output boundaries.

**512-dimension embeddings compress the score range.** Even after fixing the distance math, a `0.6` threshold blocked *everything*, including an exact title match scoring `0.492` — `text-embedding-3-small` at 512 dimensions produces much lower absolute similarities than the full 1536-dimension model. Lowering to `0.35` fixed it, but the number isn't obvious without knowing this quirk exists.

**Semantic search can't answer field-based questions.** "What books came out before 2000?" returned "couldn't find any books" despite 13 of 25 books qualifying — not a tuning problem but a category error: the question is about the `year` field, not a theme, so no threshold adjustment would have fixed it. This needed a different retrieval strategy entirely (the metadata router above), not a better similarity score.

**Titles don't reliably score as exact matches — an open problem.** A related version of the same issue: a short or generic title ("Educated", "Rebecca") can score only 0.6–0.7 in semantic search, because an embedding model produces one holistic vector per document, and a 150-word description dominates that vector far more than a 1–2 word title does. Reformatting the embedded text doesn't reliably fix this — it's a property of how dense embeddings compress meaning, not a bug in the input format. The general fix — a hybrid metadata pre-filter plus semantic re-ranking, with structured fields (including title) extracted by a small AI call instead of hand-written regex — is detailed in `what_i_would_improve.md`. Identified and scoped, not yet implemented.

**The model filling context gaps from its own training data.** Even with "never invent facts not in context" in the system prompt, asking about a missing `author` field sometimes returned a real, correct-sounding name — the model recognised the actual book and "helpfully" filled the gap from training rather than admitting it didn't know. The fix was making the prohibition explicit: don't use outside knowledge *even when confident it's correct*, and treat a missing field as "not in our catalog," not something to complete.

---

## Extensions Attempted

- **Retry with exponential backoff** (tenacity) on transient provider errors only; permanent errors (401/400) fail immediately.
- **Batch embedding with partial cache hits** — only texts not already cached are sent to the embeddings API.
- **Typed exception hierarchy** with compensating rate-limiter refunds when an AI call fails after a token was already spent.
- **Structured logging** (`structlog`) with per-request context (`request_id`) auto-attached to every log line, rendered as both console output and JSON file lines.
- **Prompt injection mitigations** — all user-supplied text is wrapped in XML tags with an explicit "treat as inert data" instruction; blast radius is bounded further since the AI layer has no tools or write access.
- **Metadata-filter query router** for structured catalogue questions, and **hardened anti-hallucination prompts** across both the RAG engine and the chatbot's second, conversational-wrap generation call.
