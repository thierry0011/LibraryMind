import os
from dotenv import load_dotenv

load_dotenv()


class settings:
    AMALIAI_API_KEY = os.getenv("AMALIAI_API_KEY", "")
    AMALIAI_BASE_URL = os.getenv("AMALIAI_BASE_URL", "")
    CHAT_URL = f"{AMALIAI_BASE_URL}/v1/chat/completions"
    EMBEDDINGS_URL = f"{AMALIAI_BASE_URL}/v1/embeddings"

    MODEL = "gpt-3.5-turbo"
    MODEL_CLAUDE = "claude-2"
    EMBEDDING_MODEL = "text-embedding-3-small"

    MAX_TOKENS = 4096
    TEMPERATURE = 0.7
    PRIMARY_PROVIDER = os.getenv("PRIMARY_PROVIDER", "openai")

    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    REDIS_TTL = int(os.getenv("REDIS_TTL", 3600))

    RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", 0.7))
    RAG_TOP_K = int(os.getenv("RAG_TOP_K", 5))

    RATE_LIMIT_PER_MINUTE = os.getenv("RATE_LIMIT_PER_MINUTE", 60)

    if not AMALIAI_API_KEY or not AMALIAI_BASE_URL:
        raise ValueError(
            "AMALIAI_API_KEY and AMALIAI_BASE_URL must be set in the environment variables."
        )
