import json

from app.infrastructure.cache import Cache
from app.infrastructure.rate_limiter import RateLimiter
from app.services.json_parse import Parser
from app.providers.resilient_ai_service import ResilientAIService
from logger import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are a library support ticket classifier. "
    "Analyse the provided ticket and return ONLY a valid JSON object with exactly these fields:\n\n"
    "{\n"
    '  "category": "account|borrowing|technical|complaint|suggestion|general",\n'
    '  "priority": "low|medium|high|urgent",\n'
    '  "sentiment": "positive|neutral|negative",\n'
    '  "department": "suggested department name for routing",\n'
    '  "summary": "one sentence summary of the issue"\n'
    "}\n\n"
    "Classification rules:\n"
    "- category: account=login/card/membership, borrowing=books/returns/renewals, "
    "technical=machines/systems/website, complaint=frustration/dissatisfaction, "
    "suggestion=improvements/ideas, general=anything else\n"
    "- priority: urgent=blocking/emergency, high=significant impact, "
    "medium=moderate issue, low=minor/positive\n"
    "- sentiment: detect emotional tone of the message\n"
    "- department: route to IT Support/Member Services/Circulation/Management\n\n"
    "Return ONLY the JSON object. No explanation, no markdown, no extra text."
)


class ClassificationService:
    def __init__(self):
        self.provider = ResilientAIService()
        self.cache = Cache()
        self.rate_limiter = RateLimiter()

    def classify(self, ticket: str) -> dict:
        logger.info(f"Classifying ticket: {ticket}")

        cache_key = self.cache.generate_key("classify", ticket)
        cached = self.cache.get(cache_key)
        if cached is not None:
            logger.info("Cache hit for classification.")
            return cached

        self.rate_limiter.acquire()

        response = self.provider.generate(prompt=ticket, system=_SYSTEM_PROMPT)
        try:
            result = Parser._parse_json(response)
        except json.JSONDecodeError as e:
            raise ValueError(f"AI returned invalid JSON: {e}\nRaw: {response}")

        self.cache.set(cache_key, result)
        return result
