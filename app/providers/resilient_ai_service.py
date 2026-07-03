from config import settings
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from logger import get_logger
from app.exceptions import AIProviderException

logger = get_logger(__name__)


class ResilientAIService:
    def __init__(self):
        self.providers = []
        if settings.PRIMARY_PROVIDER == "openai":
            self.providers = [OpenAIProvider(), AnthropicProvider()]
        else:
            self.providers = [AnthropicProvider(), OpenAIProvider()]

    def generate(
        self, prompt: str, system: str, temperature: float | None = None
    ) -> str:
        for provider in self.providers:
            try:
                if temperature is not None:
                    return provider.generate(prompt, system, temperature=temperature)
                return provider.generate(prompt, system)
            except Exception as e:
                logger.error(f"Error with {provider.provider}: {e}")
        raise AIProviderException("All providers failed to generate a response.")
