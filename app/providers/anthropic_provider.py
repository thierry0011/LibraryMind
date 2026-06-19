from app.providers.base import BaseProvider
from config import settings
# import httpx


class AnthropicProvider(BaseProvider):
    @property
    def provider(self) -> str:
        return "anthropic"

    @property
    def model(self) -> str:
        return settings.MODEL_CLAUDE

    def messages(self, prompt: str, system: str) -> list:
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
