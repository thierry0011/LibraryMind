import os
import sys

# Set required env vars before any project module is imported,
# so config.py's class-body validation does not raise.
os.environ.setdefault("AMALIAI_API_KEY", "test-api-key-12345")
os.environ.setdefault("AMALIAI_BASE_URL", "https://test.amaliai.example.com")
os.environ.setdefault("PRIMARY_PROVIDER", "openai")

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_app_dir = os.path.join(_project_root, "app")

for _path in [_project_root, _app_dir]:
    if _path not in sys.path:
        sys.path.insert(0, _path)
