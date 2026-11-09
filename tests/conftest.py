"""P0.9 — shared fixtures. Env MUST be set before `import app` (limiter and
settings are bound at import time). Dummy keys are non-secret placeholders;
no real credentials are used or asserted anywhere in this suite.
"""
import os
import sys

os.environ["FLASK_ENV"] = "test"
os.environ["FLASK_SECRET_KEY"] = "test-secret-not-real"
os.environ["GROQ_API_KEY"] = "test-dummy-key"
os.environ["PINECONE_API_KEY"] = "test-dummy-key"
os.environ["OPENAI_API_KEY"] = "test-dummy-key"
# Low limits so rate-limit tests need only a handful of requests.
os.environ["RATE_CHAT_PER_MIN"] = "5 per minute"
os.environ["RATE_CHAT_THREAD_PER_MIN"] = "5 per minute"
os.environ["RATE_IMAGE_PER_MIN"] = "5 per minute"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

import app as app_module  # noqa: E402


@pytest.fixture()
def client(monkeypatch):
    """Flask test client with heavy runtime stubbed (no network, no creds).

    get_runtime is replaced so no test builds embeddings or calls Pinecone.
    Individual tests monkeypatch app_module.ask_agent for chat behavior.
    """
    monkeypatch.setattr(
        app_module, "get_runtime", lambda: {"agent": object(), "model": "test-model"}
    )
    monkeypatch.setattr(app_module, "ask_agent", lambda agent, msg, tid: "stub answer")
    app_module.app.config.update(TESTING=True)
    with app_module.app.test_client() as c:
        yield c


def _reset_storage():
    for attr in ("storage", "_storage"):
        store = getattr(app_module.limiter, attr, None)
        reset = getattr(store, "reset", None)
        if callable(reset):
            try:
                reset()
            except Exception:
                pass


@pytest.fixture(autouse=True)
def _reset_limiter():
    """Isolate rate-limit counters between tests (shared in-memory store)."""
    _reset_storage()
    yield
    _reset_storage()
