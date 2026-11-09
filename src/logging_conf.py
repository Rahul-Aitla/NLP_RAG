"""P0.8 — stdlib logging with secret redaction. No patient content in logs."""
from __future__ import annotations

import logging
import re

_SECRET_PATTERNS = (
    re.compile(r"pcsk_[A-Za-z0-9_-]+"),
    re.compile(r"gsk_[A-Za-z0-9_-]+"),
    re.compile(r"sk-(proj-)?[A-Za-z0-9_-]+"),
    re.compile(r"(?i)(api[_-]?key\s*[:=]\s*)(['\"]?)[^'\"\s,}]+(['\"]?)"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~-]+"),
)

REDACTED = "[REDACTED]"


def redact(text: object) -> str:
    """Scrub API-key-like material from a log string."""
    out = str(text)
    for pat in _SECRET_PATTERNS:
        # Keep the key-name prefix for the api_key= form, drop the value.
        out = pat.sub(
            lambda m: (m.group(1) + REDACTED) if "api" in pat.pattern.lower() or "bearer" in pat.pattern.lower() else REDACTED,
            out,
        )
    return out


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg = redact(record.getMessage())
            record.args = ()
        except Exception:
            record.msg = REDACTED
            record.args = ()
        return True


def configure_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("hcbot")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s req=%(request_id)s "
                "route=%(route)s status=%(status)s %(message)s"
            )
        )
        logger.addHandler(handler)
    logger.addFilter(RedactingFilter())
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    return logger


def log_adapter(logger: logging.Logger, request_id: str, route: str):
    """LoggerAdapter carrying request_id/route so call sites stay one-liners."""
    return logging.LoggerAdapter(logger, {"request_id": request_id, "route": route, "status": "-"})
