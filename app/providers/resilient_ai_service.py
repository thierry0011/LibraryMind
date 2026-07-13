from config import settings
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from app.infrastructure.circuit_breaker import get_circuit_breaker
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
            breaker = get_circuit_breaker(provider.provider)
            if not breaker.allow_request():
                logger.warning(
                    "Skipping provider — circuit breaker open",
                    provider=provider.provider,
                )
                continue

            try:
                if temperature is not None:
                    result = provider.generate(prompt, system, temperature=temperature)
                else:
                    result = provider.generate(prompt, system)
            except Exception as e:
                breaker.record_failure()
                logger.error(
                    "Provider failed", provider=provider.provider, error=str(e)
                )
                continue

            breaker.record_success()
            return result
        raise AIProviderException("All providers failed to generate a response.")
