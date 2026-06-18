from base import BaseProvider
from config import settings
# import httpx

class AnthropicProvider(BaseProvider):
    def __init__(self, api_key: str, base_url: str):
        super().__init__(api_key, base_url)

    def provider(self) -> str:
        return "anthropic"
        
    def model(self) -> str:
        return settings.MODEL_CLAUDE

    def build_payload(self, prompt: str, system: str, temperature: float, max_tokens: int) -> dict:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "max_tokens": max_tokens
        }