# LibraryMind — Reflection Document

## Overview

LibraryMind is an AI-powered backend service for a public library, built as a capstone project for Module 10. It lets patrons search the catalogue using natural language, get grounded book recommendations, ask detailed questions answered by a RAG pipeline, summarise book reviews, classify support tickets, and hold multi-turn conversations with an AI librarian. This document reflects on the key decisions I made, the challenges I faced, and what I would approach differently.

---

## Key Design Decisions

### Layered Architecture

I structured the project into four distinct layers: API, Service, AI Provider, and Infrastructure. The API layer (FastAPI routers) handles only HTTP concerns — validation, routing, and error mapping. The service layer holds all business logic. The AI provider layer abstracts over model vendors. The infrastructure layer provides shared utilities.

This separation made each component independently testable and meant I could swap out a provider, a cache backend, or a vector store without touching the API or service code. It also forced me to think clearly about what each layer is responsible for, which caught several design mistakes early.

### Abstract Base Provider

Rather than calling OpenAI and Anthropic directly in service code, I defined a `BaseProvider` abstract class with a single `generate(prompt, system, temperature)` interface. Each concrete provider implements it. The `ResilientAIService` orchestrator maintains an ordered list of providers and falls through to the next on failure.

This meant adding retry logic, logging, and fallback behaviour in one place rather than duplicating it across every service. It also made the AmaliAI proxy integration straightforward — both providers point at the same base URL with a `Provider` header to route between models.

### RAG Pipeline Design

The RAG engine follows a strict pipeline: cache check → rate limit → embed → search → threshold filter → context build → generate → track usage → cache result. The relevance threshold filter was an important decision — without it, every query returns *something* even when the catalogue has nothing relevant, leading to hallucination. Returning a polite refusal when no book clears the threshold is far more honest and more useful.

I used `tiktoken` with the `cl100k_base` encoding for token counting rather than a simple character estimate. This keeps cost tracking accurate across different prompt lengths.

### Singleton Patterns for Shared State

Both `UsageTracker` and `RateLimiter` need to be shared across all services — having each service create its own instance means the limits and costs are never aggregated. I used a double-checked locking singleton (`_instance` module global with a `threading.Lock` guard) for both, which is the same pattern the Python standard library uses for module-level singletons. This ensures one shared token bucket across the entire process.

---

## Challenges Faced

### Cosine Distance vs Cosine Similarity

ChromaDB returns cosine *distance* (0 = identical, 2 = opposite), not cosine *similarity*. My first version of the vector store compared distances directly to a relevance threshold of `0.7`, which meant almost nothing was passing the filter — the wrong direction entirely. The fix was straightforward once I spotted it: `similarity = 1 - distance`. But it wasted several hours because the RAG engine was silently returning "no relevant books found" for every query, and I initially assumed the embeddings were broken.

This taught me to log intermediate values at each pipeline step. Once I added a log line showing the raw distances and computed similarities, the bug was obvious.

### JSON Fence Stripping

Even with an explicit "return ONLY valid JSON, no markdown" instruction, models frequently wrap their output in ```json ... ``` fences. The first time the classifier returned a fenced response, `json.loads()` threw an exception and the entire endpoint failed. I added a `Parser._parse_json()` utility that strips fences with a regex before parsing, and applied it everywhere structured JSON is expected.

### Relative Path for ChromaDB

The ChromaDB `PersistentClient` was initialised with `path="./books_chroma_db"`. This resolves relative to the *current working directory*, not the project root. When starting the server from a different directory, the database would be created in the wrong place — or a new empty database would be created silently, causing all searches to return nothing. Switching to `Path(__file__).parent.parent.parent / "books_chroma_db"` anchors the path to the source file location regardless of where the server is launched from.

---

## A Debugging Story

The most frustrating bug was the rate limiter appearing to do nothing. I was sending bursts of 10 requests per second and none of them were being rejected. After adding debug logs I discovered the root cause: each service instantiated its own `RateLimiter()`, so `ClassificationService`, `SummarizationService`, `RAGEngine`, and `ChatbotService` each had a separate token bucket. The 60-per-minute limit applied per service, not globally. In practice the application could make 240 AI calls per minute before any single service hit its limit.

The fix — converting `RateLimiter` to a singleton with `get_rate_limiter()` — was a one-line change at each call site, but finding the cause required understanding that Python class instances do not share state unless you explicitly make them do so.

---

## What I Would Do Differently

**Async from the start.** The entire stack uses synchronous `httpx.Client` and blocking service calls. Under load, each request ties up a thread while waiting for the AI provider. Switching to `httpx.AsyncClient` and `async`/`await` throughout would allow far more concurrent requests on the same hardware. Retrofitting async into an existing sync codebase is significantly harder than designing for it upfront.

**Persistent conversation storage.** The chatbot stores conversation history in an in-memory dictionary. Every server restart wipes all conversations. A simple SQLite or Redis-backed store would make conversations survive restarts and would also work correctly across multiple server processes.

**Structured logging from day one.** I added logging progressively as bugs appeared. Starting with structured JSON logging (using a library like `structlog`) would have made it far easier to search and filter logs, especially when debugging the RAG pipeline where multiple steps happen per request.

---

## Extensions Attempted

- **Exponential backoff with tenacity**: Added retry logic with `@retry(stop=stop_after_attempt(3), wait=wait_exponential(...))` on `BaseProvider.generate()`, retrying only on transient errors (429, 5xx, timeouts) while letting 401 and 400 errors fail immediately.
- **Batch embedding**: The embedding service supports both single and batch embedding with partial cache hits — only uncached texts are sent to the API.
- **Cost estimation**: The usage tracker estimates cost per call using per-model pricing tables and exposes a daily total through the `/health` endpoint.
