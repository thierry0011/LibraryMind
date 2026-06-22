import re
import json


class Parser:
    @staticmethod
    def _parse_json(text: str) -> dict:
        # strip ```json ... ``` fences if present
        text = re.sub(r"```json\s*|\s*```", "", text).strip()
        return json.loads(text)
