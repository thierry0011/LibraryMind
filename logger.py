import logging
import os
from logging.handlers import RotatingFileHandler

import structlog

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Processors that run on every log event before it reaches a renderer.
# merge_contextvars pulls in anything bound via bind_contextvars() —
# that is how request_id propagates automatically across every log line
# within a single request without being passed explicitly.
_SHARED_PROCESSORS: list = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_logger_name,
    structlog.stdlib.add_log_level,
    structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
    structlog.processors.StackInfoRenderer(),
    structlog.processors.ExceptionRenderer(),
]

structlog.configure(
    processors=_SHARED_PROCESSORS
    + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# Console: human-readable coloured output
_console_formatter = structlog.stdlib.ProcessorFormatter(
    processor=structlog.dev.ConsoleRenderer(),
    foreign_pre_chain=_SHARED_PROCESSORS,
)

# File: one JSON object per line — pipe to Datadog / Loki / jq with no parsing step
_json_formatter = structlog.stdlib.ProcessorFormatter(
    processor=structlog.processors.JSONRenderer(),
    foreign_pre_chain=_SHARED_PROCESSORS,
)

_root = logging.getLogger()
_root.setLevel(logging.DEBUG)

_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(_console_formatter)

_file_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, "app.log"),
    maxBytes=1_000_000,
    backupCount=3,
    encoding="utf-8",
)
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(_json_formatter)

_root.addHandler(_console_handler)
_root.addHandler(_file_handler)

# Third-party libraries that log excessively at DEBUG — cap them at WARNING
# so they don't drown out application logs.
for _noisy in ("httpx", "httpcore", "chromadb", "chromadb.config", "posthog"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.BoundLogger:
    return structlog.get_logger(name)
