from config import settings
from .base import BaseProvider


class OpenAIProvider(BaseProvider):
    @property
    def model(self) -> str:
        return settings.MODEL

    @property
    def provider(self) -> str:
        return "openai"

    def messages(self, prompt: str, system: str) -> list:
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
