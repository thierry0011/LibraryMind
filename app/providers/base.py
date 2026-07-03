import logging
import httpx
from abc import ABC, abstractmethod
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
    before_sleep_log,
)
from config import settings
from logger import get_logger
from app.exceptions import InvalidAIResponseException

logger = get_logger(__name__)


def _is_retryable(exc: BaseException) -> bool:
    """Only retry transient errors — rate limits, server faults, and network blips.

    httpx exceptions are checked directly so that our custom exception types
    (which are not httpx exceptions) are never retried — they represent
    permanent failures like malformed responses.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return isinstance(exc, (httpx.TimeoutException, httpx.ConnectError))


class BaseProvider(ABC):
    def __init__(self):
        self.api_key = settings.AMALIAI_API_KEY
        self.base_url = settings.AMALIAI_BASE_URL
        self.max_tokens = settings.MAX_TOKENS
        self.temperature = settings.TEMPERATURE

        self.client = httpx.Client(timeout=30.0)

    @property
    @abstractmethod
    def model(self) -> str:
        pass

    @property
    @abstractmethod
    def provider(self) -> str:
        pass

    @abstractmethod
    def messages(self, prompt: str, system: str) -> list:
        """
        Constructs the message payload for the API request.
        This method should be implemented by subclasses to format messages according to the provider's requirements.
        """
        pass

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(_is_retryable),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def generate(
        self, prompt: str, system: str, temperature: float | None = None
    ) -> str:
        headers = {
            "Provider": self.provider,
            "X-Api-Key": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": self.messages(prompt, system),
            "max_tokens": self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "stream": False,
        }

        try:
            # 1. Make the POST request
            response = self.client.post(
                url=self.base_url, headers=headers, json=payload
            )

            # 2. Raise an exception for 4xx or 5xx status codes
            response.raise_for_status()

            # 3. Parse the response body as JSON
            data = response.json()

            if self.provider == "openai":
                choices = data.get("choices", [])
                if not choices:
                    raise InvalidAIResponseException(
                        f"OpenAI response contained no choices: {data}"
                    )
                return data["choices"][0]["message"]["content"]

            elif self.provider == "anthropic":
                content = data.get("content", [])
                if not content:
                    raise InvalidAIResponseException(
                        f"Anthropic response contained no content blocks: {data}"
                    )
                return content[0]["text"]

            else:
                raise InvalidAIResponseException(f"Unknown provider: {self.provider}")

        except httpx.HTTPStatusError as exc:
            logger.warning(
                "HTTP %s from %s provider: %s",
                exc.response.status_code,
                self.provider,
                exc.response.text[:200],
            )
            raise

        except httpx.RequestError as exc:
            logger.warning("Network error from %s provider: %s", self.provider, exc)
            raise

        except InvalidAIResponseException:
            # Already typed correctly — log and let it propagate without wrapping
            logger.error("Malformed response from %s provider", self.provider)
            raise

        except (KeyError, IndexError, ValueError) as exc:
            logger.error(
                "Failed to parse response from %s provider: %s", self.provider, exc
            )
            raise InvalidAIResponseException(
                f"Unexpected response format from {self.provider}: {exc}"
            ) from exc

    def close(self):
        # Closes the internal HTTPX network connection pool
        self.client.close()
