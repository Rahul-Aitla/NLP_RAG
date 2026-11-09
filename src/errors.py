"""P0.7 — typed provider errors + consistent JSON error envelope."""
from __future__ import annotations

import uuid
from flask import jsonify


class ProviderError(Exception):
    """LLM / vision provider call failed (Groq/OpenAI outage, 404, timeout)."""


class RetrievalError(Exception):
    """Document retrieval failed (Pinecone/BM25/reranker error)."""


class ConfigurationError(Exception):
    """Server is missing required configuration (keys, secrets)."""


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def error_response(code: str, message: str, status: int, request_id: str):
    """Consistent API error shape. Never includes tracebacks or secrets."""
    payload = {"error": {"code": code, "message": message, "request_id": request_id}}
    resp = jsonify(payload)
    resp.status_code = status
    return resp
