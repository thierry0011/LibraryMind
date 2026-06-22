import json
from json_parse import Parser
from app.providers.resilient_ai_service import ResilientAIService
from logger import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are a literary analyst specialising in book review summarisation. "
    "You will receive a numbered list of book reviews. "
    "Analyse ALL reviews together as a whole — do not summarise them one by one. "
    "Identify patterns, common opinions, and overarching themes across the entire set.\n\n"
    "Return ONLY a valid JSON object with exactly these fields:\n\n"
    "{\n"
    '  "overall_sentiment": "positive|neutral|negative",\n'
    '  "average_rating": <number between 1.0 and 5.0>,\n'
    '  "key_themes": ["theme1", "theme2", "theme3"],\n'
    '  "praise": ["common praise point 1", "common praise point 2"],\n'
    '  "criticism": ["common criticism point 1", "common criticism point 2"],\n'
    '  "recommendation": "one sentence recommendation for potential readers"\n'
    "}\n\n"
    "Rules:\n"
    "- overall_sentiment: the dominant emotional tone across ALL reviews\n"
    "- average_rating: estimate based on language and enthusiasm, not exact scores\n"
    "- key_themes: recurring subjects or topics mentioned across multiple reviews\n"
    "- praise: specific positive aspects mentioned by multiple reviewers\n"
    "- criticism: specific negative aspects mentioned by multiple reviewers\n"
    "- recommendation: who would enjoy this book and why, in one sentence\n\n"
    "Return ONLY the JSON object. No explanation, no markdown, no extra text."
)


class SummarizationService:
    def __init__(self):
        self.provider = ResilientAIService()

    def summarize(self, reviews: list[str]):
        # response = self.provider.generate(prompt=reviews, system=_SYSTEM_PROMPT)
        numbered = "\n".join([f"Review {i+1}: {r}" for i, r in enumerate(reviews)])
        response = self.provider.generate(prompt=numbered, system=_SYSTEM_PROMPT)
        try:
            return Parser._parse_json(response)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON: {e}\nRaw: {response}")
            raise ValueError(f"Invalid JSON: {e}\nRaw: {response}")
