# LibraryMind — AI-Powered Intelligent Library Assistant

A production-grade FastAPI backend that lets library patrons search the catalogue with natural language, ask grounded questions about books, chat with an AI librarian, classify support tickets, and summarise book reviews — all powered by a multi-provider AI layer with automatic fallback.

---

## Project Structure

```
LibraryMind/
├── app/
│   ├── api/            # FastAPI routers (one per domain)
│   ├── services/       # Business logic (RAG, chatbot, classify, summarise)
│   ├── providers/      # OpenAI + Claude with resilient fallback
│   └── infrastructure/ # Cache, rate limiter, usage tracker, vector store
├── data/
│   └── books.json      # 25-book catalogue (5+ genres)
├── scripts/
│   ├── seed.py         # Populate ChromaDB from books.json
│   └── smoke_test.py   # End-to-end validation script
├── tests/              # pytest unit tests (126 tests)
├── config.py
├── main.py
└── .env
```

---

## Setup Instructions

### 1. Prerequisites

- Python 3.10+
- Redis (optional — the app degrades gracefully without it)

### 2. Create virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy the example and fill in your values:

```bash
cp .env.example .env
```

| Variable | Required | Default | Description |
|---|---|---|---|
| `AMALIAI_API_KEY` | ✅ | — | AmaliAI API key |
| `AMALIAI_BASE_URL` | ✅ | — | AmaliAI base URL |
| `PRIMARY_PROVIDER` | | `openai` | `openai` or `anthropic` |
| `REDIS_HOST` | | `localhost` | Redis hostname |
| `REDIS_PORT` | | `6379` | Redis port |
| `REDIS_TTL` | | `3600` | Cache TTL in seconds |
| `RATE_LIMIT_PER_MINUTE` | | `60` | Max AI requests per minute |
| `RELEVANCE_THRESHOLD` | | `0.7` | Minimum similarity score for RAG |
| `RAG_TOP_K` | | `5` | Number of books to retrieve per query |
| `MAX_CONVERSATION_HISTORY` | | `10` | Max messages kept per conversation |

### 5. Seed the knowledge base

```bash
python scripts/seed.py
```

This embeds all 25 books from `data/books.json` and stores them in ChromaDB.

### 6. Start the server

```bash
uvicorn main:app --reload
```

API docs available at **http://localhost:8000/docs**

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health/` | Status, daily cost, request count |
| `POST` | `/books/` | Add a book to the knowledge base |
| `POST` | `/search/books` | Semantic search over the catalogue |
| `POST` | `/search/ask` | RAG-grounded Q&A with source citations |
| `POST` | `/chat/` | Multi-turn AI librarian conversation |
| `GET` | `/chat/{id}/history` | Retrieve conversation history |
| `DELETE` | `/chat/{id}` | Clear conversation history |
| `POST` | `/classify/ticket` | Classify a support ticket |
| `POST` | `/summarise/reviews` | Summarise a list of book reviews |

---

## Sample curl Commands

### Health check
```bash
curl http://localhost:8000/health/
```

### Semantic book search
```bash
curl -X POST http://localhost:8000/search/books \
  -H "Content-Type: application/json" \
  -d '{"query": "desert planet adventure", "top_k": 5}'
```

### RAG Q&A
```bash
curl -X POST http://localhost:8000/search/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What science fiction books do you have about space exploration?"}'
```

### Chat (turn 1)
```bash
curl -X POST http://localhost:8000/chat/ \
  -H "Content-Type: application/json" \
  -d '{"conversation_id": "user-123", "message": "Recommend a thriller novel"}'
```

### Chat (turn 2 — same conversation_id keeps memory)
```bash
curl -X POST http://localhost:8000/chat/ \
  -H "Content-Type: application/json" \
  -d '{"conversation_id": "user-123", "message": "Tell me more about that one"}'
```

### Classify a support ticket
```bash
curl -X POST http://localhost:8000/classify/ticket \
  -H "Content-Type: application/json" \
  -d '{"ticket": "My library card is not working at self-checkout and I am very frustrated!"}'
```

### Summarise book reviews
```bash
curl -X POST http://localhost:8000/summarise/reviews \
  -H "Content-Type: application/json" \
  -d '{"reviews": ["Absolutely brilliant writing.", "Slow start but worth it.", "The ending was disappointing."]}'
```

### Ingest a new book
```bash
curl -X POST http://localhost:8000/books/ \
  -H "Content-Type: application/json" \
  -d '{"title": "My New Book", "author": "Jane Doe", "year": "2024", "genre": "Fiction", "description": "A compelling story about..."}'
```

---

## Running Tests

### Unit tests (126 tests)
```bash
pytest tests/ -v
```

### Smoke test (requires running server)
```bash
python scripts/smoke_test.py
# Or with a custom URL:
python scripts/smoke_test.py http://localhost:8000
```

---

## Architecture

```
Client → FastAPI (API Layer)
           ↓
       Service Layer
       ├── RAGEngine          ← vector search + AI generation
       ├── ChatbotService     ← multi-turn memory + RAG
       ├── ClassificationService ← structured ticket analysis
       └── SummarizationService  ← review analysis
           ↓
       AI Provider Layer
       └── ResilientAIService ← OpenAI → Claude fallback
           ↓
       Infrastructure Layer
       ├── VectorStore (ChromaDB)
       ├── Cache (Redis, optional)
       ├── RateLimiter (token bucket)
       └── UsageTracker (cost + tokens)
```
