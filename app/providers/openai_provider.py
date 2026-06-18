from config import settings
from base import BaseProvider

class OpenAIProvider(BaseProvider):
    def __init__(self, api_key: str, base_url: str):
        super().__init__(api_key, base_url)
        
    def model(self) -> str:
        return settings.MODEL
    
    def provider(self) -> str:
        return "openai"
    
    def build_payload(self, prompt: str, system: str, temperature: float, max_tokens: int) -> dict:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }