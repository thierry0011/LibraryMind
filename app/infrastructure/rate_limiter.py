import threading
import time

from config import settings


class RateLimiter:
    def __init__(self):
        self.max_tokens = int(settings.RATE_LIMIT_PER_MINUTE)
        self.tokens = self.max_tokens
        self.lock = threading.Lock()
        self.last_refill = time.monotonic()

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self.last_refill
        refill_amount = int(
            elapsed * (self.max_tokens / 60)
        )  # Refill tokens based on elapsed time
        if refill_amount > 0:
            self.tokens = min(self.max_tokens, self.tokens + refill_amount)
            self.last_refill = now

    def acquire(self):
        with self.lock:
            self._refill()
            if self.tokens > 0:
                self.tokens -= 1
                return True
            raise Exception("Rate limit exceeded. Please try again later.")
