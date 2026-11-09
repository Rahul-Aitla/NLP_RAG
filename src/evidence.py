"""P1.1/P1.3/P1.4/P1.15/P1.16 — structured retrieval evidence, citations, grounding.

Retrieval has three explicit states (P1.1):
  SUCCESS         — at least one above-threshold chunk; generation may proceed.
  NO_CONTEXT      — retriever worked but nothing relevant; safe "not found" reply.
  RETRIEVER_ERROR — retriever/Pinecone failed; NEVER claim absence, use a
                    temporary-service message instead.

Evidence carries source metadata end-to-end (P1.3) so citations (P1.4) are
built only from actually-retrieved chunks — page numbers are never invented.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

SUCCESS = "success"
NO_CONTEXT = "no_context"
RETRIEVER_ERROR = "retriever_error"


@dataclass
class Evidence:
    text: str
    score: float = 0.0
    document_id: str = "unknown"
    document_name: str = "clinic document"
    page: int | None = None          # None = genuinely unknown; never guessed
    chunk_id: str = ""
    doc_hash: str = ""
    ingestion_version: str = ""
    chunking_version: str = ""
    doc_version: str = ""            # P1.15: preserved when ingestion knows it
    effective_date: str = ""


def evidence_from_doc(doc, score: float) -> Evidence:
    """Build Evidence from a LangChain Document, tolerating legacy chunks
    that predate metadata enrichment (missing keys -> safe defaults)."""
    meta = getattr(doc, "metadata", None) or {}
    page = meta.get("page_number", meta.get("page"))
    try:
        page = int(page) if page is not None else None
    except (TypeError, ValueError):
        page = None
    return Evidence(
        text=getattr(doc, "page_content", "") or "",
        score=float(score),
        document_id=str(meta.get("document_id", "unknown")),
        document_name=str(meta.get("document_name", "clinic document")),
        page=page,
        chunk_id=str(meta.get("chunk_id", "")),
        doc_hash=str(meta.get("document_hash", "")),
        ingestion_version=str(meta.get("ingestion_version", "")),
        chunking_version=str(meta.get("chunking_version", "")),
        doc_version=str(meta.get("doc_version", "")),
        effective_date=str(meta.get("effective_date", "")),
    )


@dataclass
class RetrievalResult:
    status: str  # SUCCESS | NO_CONTEXT | RETRIEVER_ERROR
    evidence: list = field(default_factory=list)
    error: str = ""  # internal detail only; never shown to users


def format_cited_answer(answer: str, evidence: list, max_sources: int = 3) -> str:
    """Append a Sources block citing ONLY retrieved evidence (P1.4).

    Dedupes by (document, page); chunks with unknown page cite the document
    name alone rather than inventing a number.
    """
    seen: list = []
    for ev in evidence:
        key = (ev.document_name, ev.page)
        if key not in seen:
            seen.append(key)
        if len(seen) >= max_sources:
            break
    lines = []
    for name, page in seen:
        lines.append(f"- {name}, page {page}" if page is not None else f"- {name}")
    cited = answer.rstrip() + "\n\nSources:\n" + "\n".join(lines)
    if len({ev.document_id for ev in evidence}) > 1:
        # P1.15: deterministic multi-document notice — never silently pick one.
        cited += (
            "\n\nNote: this answer draws on more than one document. "
            "If they appear to differ, please confirm with clinic staff."
        )
    return cited


_NUMBER_RE = re.compile(r"\d+")


def _number_groups(text: str) -> set:
    """Normalized digit runs, e.g. '9:00 AM - 8:00 PM' -> {'9','00','8'}.

    Times written with or without minutes ('9 AM' vs '9:00 AM') share the
    same hour run, so paraphrase passes while invented numbers fail.
    """
    return set(_NUMBER_RE.findall(text or ""))


def grounding_ok(answer: str, evidence: list) -> bool:
    """P1.16 lightweight evidence validation (heuristic, not a guarantee).

    Passes when every numeric fact token in the answer (times, fees, phone
    fragments — the high-risk clinic facts) appears somewhere in the
    retrieved evidence. Non-numeric paraphrase is allowed; invented numbers
    are not.
    """
    if not answer or not evidence:
        return False
    corpus = " ".join(ev.text for ev in evidence)
    corpus_nums = _number_groups(corpus)
    return _number_groups(answer) <= corpus_nums
