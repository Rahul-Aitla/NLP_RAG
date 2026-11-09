"""Healthcare Clinic RAG Assistant — Flask front end (P0 hardened).

P0 security properties:
- No debug mode / dev server in production (gunicorn only).
- FLASK_SECRET_KEY from env; production fails fast without it.
- Secure session cookies, security headers, same-origin POST check.
- Rate limits + body caps, all env-configurable (see src/config.py).
- Heavy singletons (embeddings / Pinecone / BM25 / LLM / graph) are built
  LAZILY in get_runtime() so importing this module never touches the network
  and misconfiguration surfaces as a clear error naming the variable.
- Consistent JSON error envelope with request_id; never tracebacks/secrets.
- Structured redacted logging; no patient content, pixels, or answers logged.
"""
from __future__ import annotations

import base64
import re
import time
from urllib.parse import urlparse

from dotenv import load_dotenv
from flask import Flask, g, render_template, request, session
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
)

from src.agent import EMERGENCY_MSG, ask_agent, build_medical_agent
from src.config import load_settings, require_provider_keys
from src.corpus import check_corpus, check_versions, load_manifest, scan_corpus
from src.errors import (
    ConfigurationError,
    ProviderError,
    RetrievalError,
    error_response,
    new_request_id,
)
from src.image_guard import ImageRejected, validate_image
from src.logging_conf import configure_logging, log_adapter, redact
from src.safety import IMAGE_DISCLAIMER, IMAGE_SYSTEM_PROMPT, looks_like_emergency
from src.threads import ThreadStore

load_dotenv()
settings = load_settings()
logger = configure_logging(settings.log_level)

app = Flask(__name__)
# P0.2: stable secret from env. Dev/test fallback is explicit and loud —
# never a fresh os.urandom() per restart (which invalidated all sessions).
if settings.flask_secret_key:
    app.secret_key = settings.flask_secret_key
else:
    import os as _os

    app.secret_key = _os.urandom(32)
    logger.warning("FLASK_SECRET_KEY not set; using ephemeral dev secret")

app.config["MAX_CONTENT_LENGTH"] = settings.max_request_bytes
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Strict"
# Secure cookies only in production (localhost dev over http would break).
app.config["SESSION_COOKIE_SECURE"] = settings.is_production
app.config["PERMANENT_SESSION_LIFETIME"] = 1800  # 30 min

limiter = Limiter(
    get_remote_address,
    app=app,
    storage_uri="memory://",  # P0 single-replica; P1: Redis for multi-worker
    strategy="fixed-window",
)

# Production boots fail fast on missing provider keys with a clear error
# naming the variable — never a mid-request provider traceback.
if settings.is_production:
    require_provider_keys(settings)

# --- Lazy runtime (P0: import must not touch network) --------------------------
_runtime = None
_corpus_state = "unknown"  # verified | unverified ... | mismatch: reason

# P1.7: server-owned threads. The client never invents a thread id.
threads = ThreadStore(ttl_s=settings.thread_ttl_s)


def get_runtime():
    """Build (once) and return (evidence retriever, agent, model)."""
    global _runtime, _corpus_state
    if _runtime is not None:
        return _runtime
    require_provider_keys(settings)
    # P1.13: refuse to silently serve a drifted corpus. Missing manifest
    # means a pre-manifest index: warn loudly, serve as unverified.
    from src.helper import (
        CHUNKING_VERSION,
        EMBEDDING_MODEL,
        download_hugging_face_embeddings,
    )
    from src.retriever import build_evidence_retriever

    manifest = load_manifest(settings.manifest_path)
    scan = scan_corpus("data/")
    ok, reason = check_corpus(manifest, scan)
    ok_v, reason_v = check_versions(manifest, EMBEDDING_MODEL, CHUNKING_VERSION)
    if manifest is None:
        _corpus_state = "unverified: no manifest"
        logger.warning("corpus %s", _corpus_state)
    elif not ok:
        _corpus_state = "mismatch: " + reason
        raise ConfigurationError(
            "Indexed corpus does not match data/ on disk (" + reason + "). "
            "Re-run ingestion: python store_index.py"
        )
    elif not ok_v:
        _corpus_state = "mismatch: " + reason_v
        raise ConfigurationError(
            "Index was built with incompatible versions (" + reason_v + "). "
            "Re-run ingestion: python store_index.py"
        )
    else:
        _corpus_state = "verified"

    embeddings = download_hugging_face_embeddings()
    retriever = build_evidence_retriever(
        index_name="health-care-chatbot",
        embeddings=embeddings,
        threshold=settings.rerank_threshold,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    from langchain_groq import ChatGroq

    chat_model = ChatGroq(model_name=settings.groq_model)
    agent = build_medical_agent(
        retriever,
        chat_model,
        gen_max_messages=settings.gen_max_messages,
        reform_max_turns=settings.reform_max_turns,
        reform_max_chars=settings.reform_max_chars,
    )
    _runtime = {"agent": agent, "model": settings.groq_model}
    return _runtime


# --- Request lifecycle helpers -------------------------------------------------
@app.before_request
def _begin_request():
    g.request_id = new_request_id()
    g.t0 = time.monotonic()


@app.after_request
def _secure_response(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["X-Request-ID"] = getattr(g, "request_id", "-")
    # HSTS only in production over TLS — never force it in local dev.
    if settings.is_production and request.is_secure:
        resp.headers["Strict-Transport-Security"] = (
            "max-age=63072000; includeSubDomains"
        )
    else:
        resp.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "style-src 'self' https://fonts.googleapis.com "
            "https://cdnjs.cloudflare.com; "
            "font-src 'self' https://fonts.gstatic.com "
            "https://cdnjs.cloudflare.com; "
            "img-src 'self' data:; script-src 'self' 'unsafe-inline'; "
            "connect-src 'self'; frame-ancestors 'none'"
        )
    return resp


def _origin_allowed() -> bool:
    """CSRF-lite: same-origin POST check. Absent Origin/Referer (curl, tests,
    same-origin form posts without Origin) is allowed; a PRESENT foreign
    origin is rejected."""
    origin = request.headers.get("Origin") or request.headers.get("Referer")
    if not origin:
        return True
    try:
        return urlparse(origin).netloc == request.host
    except Exception:
        return False


THREAD_RE = re.compile(r"^th_[A-Za-z0-9_-]{16,64}$")


def _session_id() -> str:
    """Stable browser-session id, minted server-side (P1.7)."""
    import secrets as _secrets

    sid = session.get("sid")
    if not sid:
        sid = "ss_" + _secrets.token_urlsafe(16)
        session["sid"] = sid
        session.permanent = True
    return sid


@app.errorhandler(429)
def _rate_limited(_e):
    return error_response(
        "rate_limited",
        "Too many requests. Please slow down and try again.",
        429,
        getattr(g, "request_id", new_request_id()),
    )


@app.errorhandler(413)
def _too_large(_e):
    return error_response(
        "request_too_large",
        "Request exceeds the maximum allowed size.",
        413,
        getattr(g, "request_id", new_request_id()),
    )


@app.route("/healthz")
def healthz():
    """Lightweight liveness probe. Never builds the heavy runtime."""
    return {
        "status": "ok",
        "env": settings.env,
        "retriever": "ready" if _runtime is not None else "not_initialized",
        "corpus": _corpus_state,
    }, 200


@app.route("/new-thread", methods=["POST"])
@limiter.limit(settings.rate_chat_per_min)
def new_thread():
    """Mint a server-owned thread bound to this browser session (P1.7)."""
    if not _origin_allowed():
        return error_response(
            "forbidden_origin", "Cross-origin requests are not allowed.",
            403, g.request_id,
        )
    threads.prune()
    tid = threads.create(_session_id())
    return {"thread_id": tid}, 200


@app.route("/")
def index():
    session.clear()
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
@limiter.limit(settings.rate_chat_per_min)
@limiter.limit(
    settings.rate_chat_thread_per_min,
    key_func=lambda: "thread:" + (request.form.get("thread_id") or "default"),
)
def chat():
    log = log_adapter(logger, g.request_id, "/get")
    if not _origin_allowed():
        return error_response(
            "forbidden_origin", "Cross-origin requests are not allowed.",
            403, g.request_id,
        )
    msg = request.form.get("msg", "")
    if not msg or not msg.strip():
        return error_response(
            "missing_msg", "Field 'msg' is required and must be non-empty.",
            400, g.request_id,
        )
    msg = msg.strip()
    if len(msg) > settings.chat_msg_max_chars:
        return error_response(
            "message_too_long",
            f"Message exceeds {settings.chat_msg_max_chars} characters.",
            413, g.request_id,
        )
    thread_id = request.form.get("thread_id") or ""
    if not thread_id:
        return error_response(
            "missing_thread",
            "A server-issued thread is required. POST /new-thread first.",
            400, g.request_id,
        )
    if not THREAD_RE.match(thread_id):
        return error_response(
            "invalid_thread_id",
            "thread_id must be a server-issued thread (ask POST /new-thread).",
            400, g.request_id,
        )
    # P1.7: ownership check — forged or cross-session ids are rejected here.
    if not threads.valid(_session_id(), thread_id):
        return error_response(
            "forbidden_thread",
            "This conversation is not available in the current session. "
            "Please start a new chat.",
            403, g.request_id,
        )
    # P1.8: bound memory before doing any model work.
    ok_turn, why = threads.record_turn(
        thread_id, len(msg),
        settings.thread_max_turns, settings.thread_max_chars,
    )
    if not ok_turn:
        return error_response("thread_exhausted", why, 429, g.request_id)
    try:
        runtime = get_runtime()
    except ConfigurationError as exc:
        log.warning("misconfigured: %s", redact(str(exc)))
        return error_response(
            "server_misconfigured", str(exc), 503, g.request_id
        )
    try:
        answer = ask_agent(runtime["agent"], msg, thread_id)
    except ProviderError:
        log.warning("provider_unavailable model=%s", runtime.get("model"))
        return error_response(
            "provider_unavailable",
            "The language service is temporarily unavailable. Try again shortly.",
            502, g.request_id,
        )
    except RetrievalError:
        log.warning("retrieval_unavailable")
        return error_response(
            "retrieval_unavailable",
            "Document search is temporarily unavailable. Try again shortly.",
            503, g.request_id,
        )
    except Exception:
        log.exception("unexpected_chat_error")
        return error_response(
            "internal_error",
            "Something went wrong. Reference this request ID when contacting staff.",
            500, g.request_id,
        )
    log.info(
        "chat ok model=%s msg_chars=%d", runtime.get("model"), len(msg)
    )
    resp = app.response_class(str(answer), status=200, mimetype="text/plain")
    return resp


@app.route("/analyze-image", methods=["POST"])
@limiter.limit(settings.rate_image_per_min)
def analyze_image():
    log = log_adapter(logger, g.request_id, "/analyze-image")
    if not _origin_allowed():
        return error_response(
            "forbidden_origin", "Cross-origin requests are not allowed.",
            403, g.request_id,
        )
    if not settings.openai_api_key:
        return error_response(
            "vision_unavailable",
            "Image analysis is not configured on this server.",
            503, g.request_id,
        )
    image_file = request.files.get("image")
    if image_file is None:
        return error_response(
            "missing_image", "Field 'image' is required.", 400, g.request_id
        )
    # Bound memory BEFORE reading: Content-Length gate + capped read.
    if (request.content_length or 0) > settings.max_request_bytes:
        return error_response(
            "request_too_large", "Upload exceeds the maximum allowed size.",
            413, g.request_id,
        )
    raw = image_file.read(settings.max_image_bytes + 1)
    try:
        validated = validate_image(
            raw,
            max_bytes=settings.max_image_bytes,
            max_dim=settings.img_max_dim,
            max_pixels=settings.img_max_pixels,
            provider_long_side=settings.img_provider_long_side,
        )
    except ImageRejected as rej:
        return error_response(rej.code, rej.message, rej.status, g.request_id)

    user_text = (request.form.get("msg", "") or "").strip()
    # Emergency short-circuit: fastest safe path, no provider call or cost.
    if user_text and looks_like_emergency(user_text):
        log.info("image emergency short-circuit")
        resp = app.response_class(
            EMERGENCY_MSG + IMAGE_DISCLAIMER, status=200, mimetype="text/plain"
        )
        return resp
    # P0.5: describe-only default. The old default asked for "possible
    # conditions and solutions" — that behavior is removed.
    prompt_text = (
        user_text
        if user_text
        else "Describe in general, non-diagnostic terms what is visible in this image."
    )
    image_b64 = base64.b64encode(validated.data).decode("ascii")

    def _call_vision():
        client = OpenAI(api_key=settings.openai_api_key)
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": IMAGE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{validated.mime};base64,{image_b64}"
                            },
                        },
                    ],
                },
            ],
            max_tokens=500,
            timeout=settings.vision_timeout_s,
        )
        return completion.choices[0].message.content or ""

    try:
        answer = _call_vision()
    except (APITimeoutError, APIConnectionError, InternalServerError):
        # One bounded retry on transient failures only; never on 4xx.
        try:
            answer = _call_vision()
        except Exception:
            log.warning("vision_transient_failure")
            return error_response(
                "vision_unavailable",
                "Image analysis is temporarily unavailable. Try again shortly.",
                502, g.request_id,
            )
    except Exception:
        log.warning("vision_failure")
        return error_response(
            "vision_unavailable",
            "Image analysis is temporarily unavailable. Try again shortly.",
            502, g.request_id,
        )

    # P0.5: deterministic disclaimer on every vision reply. The system prompt
    # forbids diagnosis/treatment; the disclaimer makes the non-diagnostic
    # nature explicit even if the model hedges poorly.
    answer = (answer.strip() + "\n\n" + IMAGE_DISCLAIMER.strip()).strip()
    # P0.8: log metadata only — never pixels, never the answer text.
    log.info(
        "vision ok bytes=%d format=%s dims=%dx%d prompt_chars=%d",
        len(validated.data), validated.mime, validated.width,
        validated.height, len(prompt_text),
    )
    # NOTE: full Q/A is intentionally NOT stored in the cookie session
    # (4KB cap + health-data minimization; the UI keeps its own history).
    return app.response_class(answer, status=200, mimetype="text/plain")


if __name__ == "__main__":
    if settings.is_production:
        raise SystemExit(
            "Refusing to run the Flask dev server in production. "
            "Use gunicorn: gunicorn app:app --bind 0.0.0.0:8080"
        )
    app.run(
        host=settings.host,
        port=settings.port,
        debug=(settings.env == "development"),
    )
