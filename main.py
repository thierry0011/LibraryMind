import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from structlog.contextvars import bind_contextvars, clear_contextvars

from app.api.books import router as books_router
from app.api.chat import router as chat_router
from app.api.classify import router as classify_router
from app.api.health import router as health_router
from app.api.search import router as search_router
from app.api.summarise import router as summarise_router
from logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="LibraryMind",
    version="0.1.0",
    description="""
## AI-powered intelligent library assistant

LibraryMind combines semantic search, retrieval-augmented generation (RAG), and
large language models to help library patrons discover books, get grounded answers,
and submit support requests — all backed by a ChromaDB vector store and a resilient
multi-provider AI layer (OpenAI + Anthropic with automatic fallback).

---

### Search & Q&A
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/search/books` | Semantic catalogue search — embed the query and return the top-K books ranked by cosine similarity. |
| `POST` | `/search/ask` | RAG-grounded Q&A — embed the question, retrieve relevant books above the similarity threshold, build a grounded context block, and generate an answer that cites only books in the catalogue. Returns the answer, source books, and a `cached` flag. |

---

### Multi-turn Chat
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/chat/` | Send a message in a conversation. Each turn runs a RAG lookup for context, then generates a conversational reply that references the catalogue and the prior history. Pass the same `conversation_id` across turns to maintain continuity. |
| `GET` | `/chat/{conversation_id}/history` | Retrieve the full message history for a conversation (user + assistant turns). Returns an empty list for unknown IDs. |
| `DELETE` | `/chat/{conversation_id}` | Clear all history for a conversation. Subsequent messages on the same ID start fresh. |

---

### Classification
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/classify/ticket` | Classify a library support ticket by **category** (account / borrowing / technical / complaint / suggestion / general), **priority** (low / medium / high / urgent), **sentiment** (positive / neutral / negative), **department** for routing, and a one-sentence **summary**. |

---

### Summarisation
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/summarise/reviews` | Analyse a collection of book reviews as a whole and return **overall sentiment**, estimated **average rating**, **key themes**, common **praise** and **criticism** points, and a reading **recommendation**. |

---

### Health
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check — returns service status, cumulative daily AI cost, and total request count since the last restart. |
""",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    """Bind a unique request_id to every log line emitted during this request."""
    clear_contextvars()
    bind_contextvars(
        request_id=str(uuid.uuid4())[:8],
        method=request.method,
        path=request.url.path,
    )
    return await call_next(request)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "Unhandled exception",
        exc_type=type(exc).__name__,
        exc_msg=str(exc),
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {type(exc).__name__}: {exc}"},
    )


app.include_router(health_router, prefix="/health", tags=["Health"])
app.include_router(books_router, prefix="/books", tags=["Knowledge Base"])
app.include_router(search_router, prefix="/search", tags=["Search"])
app.include_router(chat_router, prefix="/chat", tags=["Chat"])
app.include_router(classify_router, prefix="/classify", tags=["Classification"])
app.include_router(summarise_router, prefix="/summarise", tags=["Summarisation"])
