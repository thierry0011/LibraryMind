from config import settings
import redis
import hashlib
import json


class Cache:
    def __init__(self):
        self.ttl = settings.REDIS_TTL
        try:
            self.client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                decode_responses=True,
            )
            self.available = True
            # Test the connection
            self.client.ping()
        except redis.ConnectionError:
            self.available = False

    def generate_key(self, prefix: str, *args):
        # Create a unique key based on the input arguments
        key_string = f"{prefix}:{'_'.join(map(str, args))}"
        return hashlib.sha256(key_string.encode()).hexdigest()

    def get(self, key):
        if not self.available:
            return None
        data = self.client.get(key)
        if data and isinstance(data, str):
            return json.loads(data)
        return None

    def set(self, key, value):
        if self.available:
            value = json.dumps(
                value
            )  # Convert the value to a JSON string before storing
            self.client.setex(key, self.ttl, value)
