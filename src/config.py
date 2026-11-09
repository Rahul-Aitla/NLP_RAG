"""P0.2/P0.3 — environment-driven configuration with production fail-fast.

All P0 limits are env-configurable with documented defaults. Rationale for
each default lives next to the variable so operators can tune safely.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from src.errors import ConfigurationError


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    env: str                      # development | test | production
    host: str
    port: int
    flask_secret_key: str | None
    groq_model: str
    groq_api_key: str | None
    pinecone_api_key: str | None
    openai_api_key: str | None
    # --- Request limits (P0.3) ---
    # Global body cap 8MB: 5MB image + ~33% base64 inflation + form overhead.
    max_request_bytes: int
    # 2000 chars ~= 500 tokens; clinic questions are short, longer = abuse/cost.
    chat_msg_max_chars: int
    # Vision calls cost ~10x text turns -> tighter per-IP budget.
    rate_chat_per_min: str
    rate_chat_thread_per_min: str
    rate_image_per_min: str
    # --- Image limits (P0.4) ---
    # 5MB raw upload: phone photos downscale fine below this.
    max_image_bytes: int
    img_max_dim: int               # per-side px cap
    img_max_pixels: int            # decompression-bomb guard (4096^2)
    img_provider_long_side: int    # downscale before base64 -> bounds payload
    vision_timeout_s: int
    log_level: str
    # --- RAG evidence quality (P1.2) ---
    # Cross-encoder (ms-marco-MiniLM-L-6-v2) calibration on the clinic corpus:
    # clearly relevant pairs scored +5.0..+5.3, wrong-chunk same-doc -8.3,
    # unrelated queries -11.2..-11.5. 0.0 sits in the wide empty middle as a
    # conservative cutoff: above it the chunk is topically about the query.
    rerank_threshold: float
    # --- Chunking (P1.14) ---
    # 1000/150 keeps small fact tables (doctor roster ~400 chars) inside one
    # chunk; 500/20 was observed splitting table rows from their headers.
    chunk_size: int
    chunk_overlap: int
    # --- Generation + reformulation bounds (P1.6/P1.8) ---
    gen_max_messages: int            # recent messages sent to the LLM
    reform_max_turns: int            # conversation turns used for rewriting
    reform_max_chars: int            # char budget for the rewrite input
    # --- Thread memory bounds (P1.7/P1.8) ---
    thread_max_turns: int
    thread_max_chars: int
    thread_ttl_s: int                # idle expiry for server-side threads
    # --- Corpus manifest (P1.12/P1.13) ---
    manifest_path: str

    @property
    def is_production(self) -> bool:
        return self.env == "production"

    @property
    def is_test(self) -> bool:
        return self.env == "test"


def load_settings() -> Settings:
    env = os.environ.get("FLASK_ENV", "development").lower()
    secret = os.environ.get("FLASK_SECRET_KEY")
    if env == "production" and not secret:
        raise ConfigurationError(
            "FLASK_SECRET_KEY is required in production. "
            "Set it to a long random value (e.g. `python -c "
            "\"import secrets; print(secrets.token_hex(32))\"`)."
        )
    return Settings(
        env=env,
        host=os.environ.get("FLASK_HOST", "127.0.0.1"),
        port=_int("FLASK_PORT", 8080),
        flask_secret_key=secret,
        groq_model=os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"),
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        pinecone_api_key=os.environ.get("PINECONE_API_KEY"),
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        max_request_bytes=_int("MAX_REQUEST_BYTES", 8 * 1024 * 1024),
        chat_msg_max_chars=_int("CHAT_MSG_MAX_CHARS", 2000),
        rate_chat_per_min=os.environ.get("RATE_CHAT_PER_MIN", "20 per minute"),
        rate_chat_thread_per_min=os.environ.get(
            "RATE_CHAT_THREAD_PER_MIN", "30 per minute"
        ),
        rate_image_per_min=os.environ.get("RATE_IMAGE_PER_MIN", "6 per minute"),
        max_image_bytes=_int("MAX_IMAGE_BYTES", 5 * 1024 * 1024),
        img_max_dim=_int("IMG_MAX_DIM", 4096),
        img_max_pixels=_int("IMG_MAX_PIXELS", 16_777_216),
        img_provider_long_side=_int("IMG_PROVIDER_LONG_SIDE", 1600),
        vision_timeout_s=_int("VISION_TIMEOUT_S", 60),
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
        rerank_threshold=_float("RERANK_THRESHOLD", 0.0),
        chunk_size=_int("CHUNK_SIZE", 1000),
        chunk_overlap=_int("CHUNK_OVERLAP", 150),
        gen_max_messages=_int("GEN_MAX_MESSAGES", 8),
        reform_max_turns=_int("REFORM_MAX_TURNS", 3),
        reform_max_chars=_int("REFORM_MAX_CHARS", 1500),
        thread_max_turns=_int("THREAD_MAX_TURNS", 30),
        thread_max_chars=_int("THREAD_MAX_CHARS", 20000),
        thread_ttl_s=_int("THREAD_TTL_S", 7200),
        manifest_path=os.environ.get("INDEX_MANIFEST", "data/index_manifest.json"),
    )


def require_provider_keys(settings: Settings) -> None:
    """Fail clearly when keys needed by a code path are absent.

    Called at production boot (chat path) and at image-request time
    (vision key), so misconfiguration surfaces as a clear error naming
    the variable — never as a mid-request provider traceback.
    """
    missing = []
    if not settings.groq_api_key:
        missing.append("GROQ_API_KEY")
    if not settings.pinecone_api_key:
        missing.append("PINECONE_API_KEY")
    if not (settings.groq_model or "").strip():
        # P1.9: empty model id would 404 on every call; fail at config time.
        missing.append("GROQ_MODEL (must be a non-empty model id)")
    if missing:
        raise ConfigurationError(
            "Missing required provider keys: " + ", ".join(missing) + ". "
            "Set them in the environment (never bake them into images)."
        )
