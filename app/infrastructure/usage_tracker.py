from datetime import datetime
from logger import get_logger

logger = get_logger(__name__)

_instance: "UsageTracker | None" = None


def get_usage_tracker() -> "UsageTracker":
    global _instance
    if _instance is None:
        _instance = UsageTracker()
    return _instance


class UsageTracker:
    def __init__(self):
        self.records = []
        self.pricing = {
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            "claude-2": {"input": 0.0080, "output": 0.0240},
            "text-embedding-3-small": {"input": 0.00002, "output": 0.0},
        }

    def track(self, model, prompt_tokens, completion_tokens):
        """
        Track the usage of tokens for a given model.

        Args:
            model (str): The name of the model being used.
            prompt_tokens (int): The number of tokens used in the prompt.
            completion_tokens (int): The number of tokens used in the completion.
        """
        timestamp = datetime.utcnow().isoformat()
        total_tokens = prompt_tokens + completion_tokens

        input_cost = (prompt_tokens / 1000) * self.pricing.get(model, {}).get(
            "input", 0
        )
        output_cost = (completion_tokens / 1000) * self.pricing.get(model, {}).get(
            "output", 0
        )
        cost = input_cost + output_cost

        self.records.append(
            {
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost": cost,
                "timestamp": timestamp,
            }
        )
        logger.info(
            f"Model: {model}, Prompt Tokens: {prompt_tokens}, Completion Tokens: {completion_tokens}, Total Tokens: {total_tokens}",
            extra={"timestamp": timestamp},
        )

    def get_daily_cost(self):
        """
        Calculate the total cost for the current day based on the tracked usage.

        Returns:
            float: The total cost in USD for the current day.
        """
        today = datetime.utcnow().date()
        daily_cost = 0.0
        for record in self.records:
            record_date = datetime.fromisoformat(record["timestamp"]).date()
            if record_date == today:
                daily_cost += record["cost"]
        return daily_cost
