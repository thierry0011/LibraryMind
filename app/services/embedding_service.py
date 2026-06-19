from config import settings
import httpx
from logger import get_logger
from app.infrastructure.cache import Cache


logger = get_logger(__name__)


class EmbeddingsService:
    def __init__(self, provider: str = "openai", model: str = settings.EMBEDDING_MODEL):
        self.api_key = settings.AMALIAI_API_KEY
        self.api_url = settings.EMBEDDINGS_URL
        self.provider = provider
        self.model = model
        self.cache = Cache()

        # Best practice: Initialize a reusable client with a default timeout
        self.client = httpx.Client(timeout=30.0)

    def embed(self, text: str) -> list:
        """
        Get embeddings for the given text using the specified model.

        Args:
            text (str): The input text for which to generate embeddings.

        Returns:
            list: A list of embedding vectors.
        """
        normalized = text.strip().lower()
        if self.cache.available:
            cache_key = self.cache.generate_key("embedding", normalized)
            cached_embedding = self.cache.get(cache_key)
            if cached_embedding is not None:
                logger.info("Returning cached embedding.")
                return cached_embedding
        payload = {"model": self.model, "input": text}

        headers = {
            "Provider": self.provider,
            "X-Api-Key": self.api_key,
            "Content-Type": "application/json",
        }

        response = self.client.post(url=self.api_url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()["data"]
        logger.info("Embeddings retrieved successfully.")
        cache_key = self.cache.generate_key("embedding", normalized)
        self.cache.set(cache_key, data[0]["embedding"])
        return data[0]["embedding"]

    def embed_batch(self, texts: list) -> list:
        normalized = [t.strip().lower() for t in texts]
        results = [None] * len(normalized)
        texts_to_fetch = []

        for i, text in enumerate(normalized):
            cache_key = self.cache.generate_key("embedding", text)
            cached = self.cache.get(cache_key) if self.cache.available else None
            if cached is not None:
                results[i] = cached
            else:
                texts_to_fetch.append((i, text))

        if not texts_to_fetch:
            logger.info("Returning cached batch embeddings.")
            return results

        payload = {
            "model": self.model,
            "input": [text for _, text in texts_to_fetch],
        }
        headers = {
            "Provider": self.provider,
            "X-Api-Key": self.api_key,
            "Content-Type": "application/json",
        }
        response = self.client.post(
            url=self.api_url,
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        api_data = response.json()["data"]
        logger.info("Batch embeddings retrieved successfully.")
        for (i, text), item in zip(texts_to_fetch, api_data):
            embedding = item["embedding"]
            results[i] = embedding
            if self.cache.available:
                self.cache.set(self.cache.generate_key("embedding", text), embedding)
        return results
