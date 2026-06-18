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
    if not AMALIAI_API_KEY or not AMALIAI_BASE_URL:
        raise ValueError("AMALIAI_API_KEY and AMALIAI_BASE_URL must be set in the environment variables.")
    