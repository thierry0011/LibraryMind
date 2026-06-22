from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.books import router as books_router
from app.api.chat import router as chat_router
from app.api.classify import router as classify_router
from app.api.health import router as health_router
from app.api.search import router as search_router
from app.api.summarise import router as summarise_router

app = FastAPI(
    title="LibraryMind",
    version="0.1.0",
    description="AI-powered intelligent library assistant",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix="/health", tags=["Health"])
app.include_router(books_router, prefix="/books", tags=["Knowledge Base"])
app.include_router(search_router, prefix="/search", tags=["Search"])
app.include_router(chat_router, prefix="/chat", tags=["Chat"])
app.include_router(classify_router, prefix="/classify", tags=["Classification"])
app.include_router(summarise_router, prefix="/summarise", tags=["Summarisation"])
