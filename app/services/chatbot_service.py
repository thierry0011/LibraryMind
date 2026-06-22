from app.infrastructure.rate_limiter import RateLimiter
from app.services.rag_engine import RAGEngine
from app.providers.resilient_ai_service import ResilientAIService
from config import settings

_SYSTEM_PROMPT = (
    "You are LibraryMind, a warm and knowledgeable library assistant. "
    "You help patrons discover books based on our catalogue. "
    "Remember the conversation history and refer back naturally. "
    "Never invent book titles, authors, or facts not in the provided context."
)


class ChatbotService:
    def __init__(self):
        self.rag_engine = RAGEngine()
        self.provider = ResilientAIService()
        self.rate_limiter = RateLimiter()
        self.conversation = {}
        self.max_history = settings.MAX_CONVERSATION_HISTORY

    def get_conversation_history(self, conversation_id: str) -> list:
        return self.conversation.get(conversation_id, [])

    def chat(self, conversation_id: str, message: str) -> dict:
        # Initialize conversation history if not present
        if conversation_id not in self.conversation:
            self.conversation[conversation_id] = []

        # Append user input to conversation history
        self.conversation[conversation_id].append({"role": "user", "content": message})

        # get RAG context (books relevant to current message)
        rag_result = self.rag_engine.ask(message)

        # build prompt that includes BOTH history AND rag context
        history_text = "\n".join(
            [
                f"{msg['role'].capitalize()}: {msg['content']}"
                for msg in self.conversation[conversation_id][
                    :-1
                ]  # exclude current message
            ]
        )
        prompt = f"""Relevant books from our catalogue:
                {rag_result['answer']}

                Conversation so far:
                {history_text}

                User: {message}"""

        # generate response with full context
        self.rate_limiter.acquire()
        reply = self.provider.generate(prompt=prompt, system=_SYSTEM_PROMPT)

        # Append model response to conversation history
        # self.conversation[conversation_id].append({"role": "assistant", "content": reply.get("answer", "")})
        self.conversation[conversation_id].append(
            {"role": "assistant", "content": reply}
        )

        # Limit conversation history to max_history
        if len(self.conversation[conversation_id]) > self.max_history:
            self.conversation[conversation_id] = self.conversation[conversation_id][
                -self.max_history :
            ]

        return {"reply": reply, "sources": rag_result.get("sources", [])}
