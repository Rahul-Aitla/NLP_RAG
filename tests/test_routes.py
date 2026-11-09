"""P0.7/P0.9 + P1.7/P1.20 — routes: validation, errors, headers, threads, limits."""
import io

import pytest

import app as app_module
from src.errors import ConfigurationError, ProviderError, RetrievalError


@pytest.fixture()
def tid(client):
    """Mint a server-owned thread for the test session."""
    r = client.post("/new-thread")
    assert r.status_code == 200, r.get_data(as_text=True)
    return r.get_json()["thread_id"]


def post_chat(client, msg, thread):
    return client.post("/get", data={"msg": msg, "thread_id": thread})


def test_healthz():
    with app_module.app.test_client() as c:
        r = c.get("/healthz")
        assert r.status_code == 200
        assert r.get_json()["status"] == "ok"


def test_security_headers_present():
    with app_module.app.test_client() as c:
        r = c.get("/healthz")
        assert r.headers.get("X-Content-Type-Options") == "nosniff"
        assert r.headers.get("X-Frame-Options") == "DENY"
        assert r.headers.get("Referrer-Policy") == "no-referrer"
        assert r.headers.get("X-Request-ID")


def test_new_thread_mints_server_id(client):
    r = client.post("/new-thread")
    assert r.status_code == 200
    body = r.get_json()
    assert body["thread_id"].startswith("th_")
    assert len(body["thread_id"]) > 20


def test_chat_missing_msg_400(client, tid):
    r = client.post("/get", data={"thread_id": tid})
    assert r.status_code == 400
    assert r.get_json()["error"]["code"] == "missing_msg"


def test_chat_empty_msg_400(client, tid):
    r = client.post("/get", data={"msg": "   ", "thread_id": tid})
    assert r.status_code == 400


def test_chat_oversized_msg_413(client, tid):
    r = post_chat(client, "x" * 5000, tid)
    assert r.status_code == 413
    assert r.get_json()["error"]["code"] == "message_too_long"


def test_chat_missing_thread_400(client):
    r = client.post("/get", data={"msg": "hi"})
    assert r.status_code == 400
    assert r.get_json()["error"]["code"] == "missing_thread"


def test_chat_bad_thread_400(client):
    r = client.post("/get", data={"msg": "hi", "thread_id": "evil;DROP"})
    assert r.status_code == 400


def test_chat_forged_thread_403(client):
    forged = "th_" + "A" * 24  # well-formed but never minted
    r = post_chat(client, "hi", forged)
    assert r.status_code == 403
    assert r.get_json()["error"]["code"] == "forbidden_thread"


def test_chat_cross_session_thread_403(client):
    mine = client.post("/new-thread").get_json()["thread_id"]
    with app_module.app.test_client() as other:
        r = other.post("/get", data={"msg": "hi", "thread_id": mine})
        assert r.status_code == 403


def test_chat_success_200_text(client, tid):
    r = post_chat(client, "What are the timings?", tid)
    assert r.status_code == 200
    assert r.data == b"stub answer"


def test_llm_failure_no_traceback_502(client, tid, monkeypatch):
    def boom(agent, msg, thread):
        raise ProviderError("groq 404 model_not_found")

    monkeypatch.setattr(app_module, "ask_agent", boom)
    r = post_chat(client, "hello", tid)
    assert r.status_code == 502
    body = r.get_json()
    assert body["error"]["code"] == "provider_unavailable"
    assert "Traceback" not in r.get_data(as_text=True)
    assert "model_not_found" not in r.get_data(as_text=True)


def test_retrieval_failure_503(client, tid, monkeypatch):
    monkeypatch.setattr(
        app_module, "ask_agent",
        lambda a, m, t: (_ for _ in ()).throw(RetrievalError("pinecone down")),
    )
    r = post_chat(client, "hello", tid)
    assert r.status_code == 503
    assert r.get_json()["error"]["code"] == "retrieval_unavailable"


def test_misconfigured_503_names_nothing_secret(client, tid, monkeypatch):
    monkeypatch.setattr(
        app_module, "get_runtime",
        lambda: (_ for _ in ()).throw(ConfigurationError("Missing: GROQ_API_KEY")),
    )
    r = post_chat(client, "hello", tid)
    assert r.status_code == 503
    assert "GROQ_API_KEY" in r.get_json()["error"]["message"]


def test_foreign_origin_403(client, tid):
    r = client.post("/get", data={"msg": "hi", "thread_id": tid},
                    headers={"Origin": "https://evil.example"})
    assert r.status_code == 403


def test_chat_rate_limited_429(client, tid):
    statuses = [post_chat(client, "hi", tid).status_code for _ in range(7)]
    assert 429 in statuses  # limit is 5/min in test env


def test_thread_exhaustion_429(client, monkeypatch):
    """30+ turn chat stays within budget: turns past the cap are refused."""
    import dataclasses

    import app as app_mod

    monkeypatch.setattr(
        app_mod, "settings",
        dataclasses.replace(app_mod.settings, thread_max_turns=2),
    )
    t = client.post("/new-thread").get_json()["thread_id"]
    assert post_chat(client, "one", t).status_code == 200
    assert post_chat(client, "two", t).status_code == 200
    r = post_chat(client, "three", t)
    assert r.status_code == 429
    assert r.get_json()["error"]["code"] == "thread_exhausted"


def _img_file(content: bytes, name="x.png"):
    return {"image": (io.BytesIO(content), name)}


def test_image_missing_400(client):
    r = client.post("/analyze-image", data={})
    assert r.status_code == 400


def test_image_invalid_bytes_400(client):
    r = client.post("/analyze-image", data=_img_file(b"not an image at all"))
    assert r.status_code == 400
    assert r.get_json()["error"]["code"] == "invalid_image"


def test_image_oversized_413(client):
    big = b"\x89PNG\r\n\x1a\n" + b"\x00" * (6 * 1024 * 1024)
    r = client.post("/analyze-image", data=_img_file(big))
    assert r.status_code == 413


def test_image_emergency_shortcircuit_no_provider_call(client, monkeypatch):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (50, 50), (10, 10, 10)).save(buf, format="PNG")

    def must_not_run(*a, **k):
        raise AssertionError("provider must not be called for emergencies")

    monkeypatch.setattr(app_module, "OpenAI", must_not_run)
    r = client.post(
        "/analyze-image",
        data={**_img_file(buf.getvalue()),
              "msg": "severe chest pain, what is this rash?"},
    )
    assert r.status_code == 200
    assert "emergency" in r.get_data(as_text=True).lower()


def test_image_success_appends_disclaimer(client, monkeypatch):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (60, 40), (10, 200, 10)).save(buf, format="PNG")

    class _Msg:
        content = "The image shows a generally red area."

    class _Choice:
        message = _Msg()

    class _Comp:
        choices = [_Choice()]

    class _Completions:
        @staticmethod
        def create(**kwargs):
            # P0.5: system prompt must not request diagnosis/treatment.
            sys = kwargs["messages"][0]["content"].lower()
            assert "suggest possible conditions" not in sys
            assert "recommend treatments" not in sys
            return _Comp()

    class _Chat:
        completions = _Completions()

    class _FakeOpenAI:
        def __init__(self, api_key=None):
            self.chat = _Chat()

    monkeypatch.setattr(app_module, "OpenAI", _FakeOpenAI)
    r = client.post("/analyze-image", data=_img_file(buf.getvalue()))
    assert r.status_code == 200
    text = r.get_data(as_text=True)
    assert "not a diagnosis" in text
    assert "healthcare professional" in text


def test_image_rate_limited_429(client, monkeypatch):
    from PIL import Image

    class _Msg:
        content = "A generally red area is visible."

    class _Choice:
        message = _Msg()

    class _Comp:
        choices = [_Choice()]

    class _Completions:
        @staticmethod
        def create(**kwargs):
            return _Comp()

    class _Chat:
        completions = _Completions()

    class _FakeOpenAI:
        def __init__(self, api_key=None):
            self.chat = _Chat()

    monkeypatch.setattr(app_module, "OpenAI", _FakeOpenAI)

    def payload():
        buf = io.BytesIO()
        Image.new("RGB", (20, 20), (1, 2, 3)).save(buf, format="PNG")
        return _img_file(buf.getvalue())

    statuses = [client.post("/analyze-image", data=payload()).status_code
                for _ in range(7)]
    assert 429 in statuses  # limit is 5/min in test env
