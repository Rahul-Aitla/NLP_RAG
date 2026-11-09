"""P1.2/P1.11 — hybrid retrieval that keeps cross-encoder scores.

Pipeline (unchanged shape, new visibility):
  dense Pinecone (k) + in-memory BM25 (k) -> RRF ensemble
  -> cross-encoder scores EVERY candidate (scores were previously discarded)
  -> top_n by score -> configurable relevance threshold (P1.2)
  -> structured RetrievalResult (P1.1): SUCCESS | NO_CONTEXT | RETRIEVER_ERROR

P1.11: no @tool decoration anywhere here. Retrieval is orchestrated
directly by the LangGraph retrieve node via EvidenceRetriever.search();
the abstraction now says exactly what the system does.
"""
from __future__ import annotations

from src.errors import RetrievalError
from src.evidence import (
    NO_CONTEXT,
    RETRIEVER_ERROR,
    SUCCESS,
    RetrievalResult,
    evidence_from_doc,
)
from src.helper import (
    download_hugging_face_embeddings,
    enrich_docs,
    load_pdf_file,
    text_split,
)


def get_chunks(data_dir: str = "data/", chunk_size: int = 1000,
               chunk_overlap: int = 150):
    """Rebuild the exact chunk stream ingestion produces (for in-memory BM25).

    Uses the same enrich + split functions as store_index.py so the sparse
    side always matches the ingestion recipe (P1.13: same definition;
    P1.12 manifest versions detect drift against Pinecone).
    """
    from src.helper import CHUNKING_VERSION, INGESTION_VERSION

    extracted = load_pdf_file(data_dir)
    enriched = enrich_docs(extracted, INGESTION_VERSION)
    return text_split(enriched, chunk_size, chunk_overlap, CHUNKING_VERSION)


class EvidenceRetriever:
    """Thin orchestrator over an RRF ensemble + cross-encoder.

    search() NEVER raises for expected retrieval problems — it returns a
    RetrievalResult whose status the graph maps to user-facing messages.
    """

    def __init__(self, ensemble, cross_encoder, threshold: float = 0.0,
                 top_n: int = 4):
        self.ensemble = ensemble
        self.cross_encoder = cross_encoder
        self.threshold = threshold
        self.top_n = top_n

    def search(self, query: str) -> RetrievalResult:
        try:
            candidates = self.ensemble.invoke(query)
        except Exception as exc:
            return RetrievalResult(
                RETRIEVER_ERROR, [],
                error=f"ensemble retrieval failed: {type(exc).__name__}",
            )
        if not candidates:
            return RetrievalResult(NO_CONTEXT, [])
        try:
            pairs = [(query, getattr(d, "page_content", "") or "") for d in candidates]
            scores = list(self.cross_encoder.score(pairs))
        except Exception as exc:
            return RetrievalResult(
                RETRIEVER_ERROR, [],
                error=f"rerank scoring failed: {type(exc).__name__}",
            )
        ranked = sorted(zip(candidates, scores), key=lambda t: t[1], reverse=True)
        evidence = []
        for doc, score in ranked[: self.top_n]:
            try:
                if not isinstance(getattr(doc, "page_content", None), str):
                    continue  # malformed candidate: skip, don't trust it
                evidence.append(evidence_from_doc(doc, float(score)))
            except Exception:
                continue
        if not evidence:
            # Candidates existed but none were usable -> retriever untrusted.
            return RetrievalResult(
                RETRIEVER_ERROR, [], error="no parseable candidates"
            )
        passed = [ev for ev in evidence if ev.score >= self.threshold]
        if not passed:
            return RetrievalResult(NO_CONTEXT, [])
        return RetrievalResult(SUCCESS, passed)


def build_evidence_retriever(
    index_name: str,
    documents=None,
    data_dir: str = "data/",
    embeddings=None,
    k: int = 20,
    top_n: int = 4,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    threshold: float = 0.0,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> EvidenceRetriever:
    """Build the production retrieval bundle (Pinecone + BM25 + rerank)."""
    from langchain.retrievers import EnsembleRetriever

    from langchain_community.cross_encoders import HuggingFaceCrossEncoder
    from langchain_community.retrievers import BM25Retriever
    from langchain_pinecone import PineconeVectorStore

    if embeddings is None:
        embeddings = download_hugging_face_embeddings()
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name, embedding=embeddings
    )
    dense = vectorstore.as_retriever(search_kwargs={"k": k})
    if documents is None:
        documents = get_chunks(data_dir, chunk_size, chunk_overlap)
    bm25 = BM25Retriever.from_documents(documents)
    bm25.k = k
    ensemble = EnsembleRetriever(
        retrievers=[bm25, dense],
        weights=[sparse_weight, dense_weight],
    )
    cross_encoder = HuggingFaceCrossEncoder(model_name=reranker_model)
    return EvidenceRetriever(ensemble, cross_encoder, threshold, top_n)


def build_offline_bm25(documents, k: int = 20):
    """Test/eval helper: BM25-only candidate source, no network or creds."""
    from langchain_community.retrievers import BM25Retriever

    bm25 = BM25Retriever.from_documents(documents)
    bm25.k = k
    return bm25
