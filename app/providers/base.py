import httpx
from abc import ABC, abstractmethod
from config import settings


class BaseProvider(ABC):
    def __init__(self):
        self.api_key = settings.AMALIAI_API_KEY
        self.base_url = settings.AMALIAI_BASE_URL
        self.max_tokens = settings.MAX_TOKENS
        self.temperature = settings.TEMPERATURE

        # Best practice: Initialize a reusable client with a default timeout
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

    def generate(self, prompt: str, system: str) -> str:
        headers = {
            "Provider": self.provider,
            "X-Api-Key": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": self.messages(prompt, system),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
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
                # OpenAI path
                choices = data.get("choices", [])
                if not choices:
                    raise ValueError("...")
                return data["choices"][0]["message"]["content"]

            elif self.provider == "anthropic":
                content = data.get("content", [])
                if not content:
                    raise ValueError("...")
                return content[0]["text"]

            else:
                raise ValueError(f"Unknown provider: {self.provider}")

        except httpx.HTTPStatusError as exc:
            # Handles bad HTTP status codes (e.g., 401 Unauthorized, 429 Rate Limit)
            print(
                f"HTTP Error {exc.response.status_code} while requesting AI response: {exc.response.text}"
            )
            raise

        except httpx.RequestError as exc:
            # Handles network-level errors (e.g., timeouts, connection failures)
            print(f"Network error occurred while connecting to AI provider: {exc}")
            raise

        except (KeyError, IndexError, ValueError) as exc:
            # Handles unexpected API response formats or parsing bugs
            print(f"Failed to parse API response structure: {exc}")
            raise

    def close(self):
        # Closes the internal HTTPX network connection pool
        self.client.close()
