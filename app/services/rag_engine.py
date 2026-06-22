from app.infrastructure.cache import Cache
from app.infrastructure.vector_store import VectorStore
from app.infrastructure.usage_tracker import UsageTracker
from app.infrastructure.rate_limiter import RateLimiter
from app.services.embedding_service import EmbeddingsService
from logging import getLogger
from app.providers.resilient_ai_service import ResilientAIService
from config import settings
import tiktoken

logger = getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are LibraryMind, a knowledgeable and friendly library assistant. "
    "Your role is to help users discover and learn about books from our catalog. "
    "Answer questions ONLY using the book context provided — never invent titles, authors, "
    "plot summaries, or facts not explicitly stated in the context. "
    "If the context is insufficient to fully answer the question, acknowledge that clearly."
)


class RAGEngine:
    def __init__(self):
        self.vector_store = VectorStore()
        self.cache = Cache()
        self.usage_tracker = UsageTracker()
        self.rate_limiter = RateLimiter()
        self.provider = ResilientAIService()
        self.embedding_service = EmbeddingsService()
        self.relevance_threshold = settings.RELEVANCE_THRESHOLD
        self.rag_top_k = settings.RAG_TOP_K
        self._tokenizer = tiktoken.get_encoding("cl100k_base")

    def _build_context(self, books: list) -> str:
        entries = []
        for i, book in enumerate(books, 1):
            meta = book["metadata"]
            entries.append(
                f"[{i}] Title: {meta.get('title', 'Unknown')}\n"
                f"    Author: {meta.get('author', 'Unknown')}\n"
                f"    Year: {meta.get('year', 'N/A')}\n"
                f"    Genre: {meta.get('genre', 'N/A')}\n"
                f"    Description: {book['document']}"
            )
        return "\n\n".join(entries)

    def _build_prompt(self, question: str, context: str) -> str:
        return (
            f"Here are relevant books from our library catalog:\n\n"
            f"{context}\n\n"
            f"---\n\n"
            f"User question: {question}\n\n"
            f"Answer based solely on the books listed above."
        )

    def _count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text))

    def ask(self, question: str) -> dict:
        # 1. Check cache — return immediately on hit
        cache_key = self.cache.generate_key("rag", question)
        cached = self.cache.get(cache_key)
        if cached is not None:
            logger.info("Cache hit for RAG query.")
            return {**cached, "cached": True}

        # 2. Rate limit — raises if quota exceeded
        self.rate_limiter.acquire()

        # 3. Embed the question into a query vector
        query_vector = self.embedding_service.embed(question)

        # 4. Retrieve candidate books from the vector store
        candidates = self.vector_store.search_books(query_vector, top_k=self.rag_top_k)

        # 5. Discard weak matches below the relevance threshold
        relevant = [
            b for b in candidates if b["similarity"] >= self.relevance_threshold
        ]

        # 6. No relevant results → polite refusal, no hallucination
        if not relevant:
            logger.info("No books above relevance threshold for query.")
            return {
                "answer": (
                    "I couldn't find any books in our catalog that closely match your question. "
                    "Try rephrasing, or ask about a different topic."
                ),
                "sources": [],
                "cached": False,
            }

        # 7. Format the relevant books into a grounded context block
        context = self._build_context(relevant)
        prompt = self._build_prompt(question, context)

        # 8. Generate a grounded answer via the resilient AI service
        answer = self.provider.generate(prompt=prompt, system=_SYSTEM_PROMPT)

        # 9. Track token usage and cost
        prompt_tokens = self._count_tokens(_SYSTEM_PROMPT + prompt)
        completion_tokens = self._count_tokens(answer)
        model = (
            settings.MODEL
            if settings.PRIMARY_PROVIDER == "openai"
            else settings.MODEL_CLAUDE
        )
        self.usage_tracker.track(model, prompt_tokens, completion_tokens)
        logger.info(
            "Generated RAG answer. Prompt tokens: %d, completion tokens: %d",
            prompt_tokens,
            completion_tokens,
        )

        # 10. Cache the result for future identical queries
        sources = [
            {
                "title": b["metadata"].get("title"),
                "author": b["metadata"].get("author"),
                "similarity": round(b["similarity"], 4),
            }
            for b in relevant
        ]
        result = {"answer": answer, "sources": sources}
        self.cache.set(cache_key, result)

        # 11. Return answer, sources, and cache provenance flag
        return {**result, "cached": False}
