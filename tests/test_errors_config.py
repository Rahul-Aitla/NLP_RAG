"""P0.7/P0.8/P0.9 — error envelope, redaction, config fail-fast."""
import logging

import pytest

from src.config import ConfigurationError, load_settings, require_provider_keys
from src.errors import error_response
from src.logging_conf import configure_logging, redact


def test_error_envelope_shape():
    import app as app_module

    with app_module.app.test_request_context():
        resp = error_response("X", "Y", 400, "req123")
        assert resp.status_code == 400
        body = resp.get_json()
        assert body == {"error": {"code": "X", "message": "Y", "request_id": "req123"}}
        assert resp.content_type.startswith("application/json")


@pytest.mark.parametrize(
    "secret",
    ["gsk_abc123XYZ", "pcsk_abc123XYZ", "sk-proj-abc123", "Bearer mytoken123"],
)
def test_redact_scrubs_secrets(secret):
    out = redact(f"call failed with key {secret} end")
    assert secret not in out
    assert "[REDACTED]" in out


def test_redact_keeps_benign_text():
    assert redact("clinic open 9am") == "clinic open 9am"


def test_logging_never_emits_raw_secret(caplog):
    logger = configure_logging("INFO")
    with caplog.at_level(logging.INFO, logger="hcbot"):
        logger.warning("provider said %(e)s", {"e": "key=gsk_superSECRET"})
    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert "gsk_superSECRET" not in joined


def test_production_without_secret_fails_fast(monkeypatch):
    monkeypatch.setenv("FLASK_ENV", "production")
    monkeypatch.delenv("FLASK_SECRET_KEY", raising=False)
    with pytest.raises(ConfigurationError):
        load_settings()


def test_missing_provider_keys_fail_clearly(monkeypatch):
    s = load_settings()
    import dataclasses

    s2 = dataclasses.replace(s, groq_api_key=None, pinecone_api_key=None)
    with pytest.raises(ConfigurationError) as ei:
        require_provider_keys(s2)
    assert "GROQ_API_KEY" in str(ei.value)
    assert "PINECONE_API_KEY" in str(ei.value)
