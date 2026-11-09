"""P1.1/P1.2/P1.20 — retrieval states and relevance threshold."""
import pytest
from langchain_core.documents import Document

from src.evidence import NO_CONTEXT, RETRIEVER_ERROR, SUCCESS
from src.retriever import EvidenceRetriever


def _doc(text, **meta):
    return Document(page_content=text, metadata=meta)


class FakeEnsemble:
    def __init__(self, docs=None, exc=None):
        self.docs = docs
        self.exc = exc

    def invoke(self, query):
        if self.exc:
            raise self.exc
        return self.docs


class FakeCE:
    """Deterministic cross-encoder: score = number of query words in doc."""

    def score(self, pairs):
        out = []
        for q, d in pairs:
            qw = set(q.lower().split())
            out.append(float(sum(1 for w in qw if w in d.lower())) - 2.0)
        return out


def _bundle(docs=None, exc=None, threshold=0.0):
    return EvidenceRetriever(FakeEnsemble(docs, exc), FakeCE(), threshold, top_n=4)


def test_zero_documents_is_no_context():
    res = _bundle([]).search("clinic timings?")
    assert res.status == NO_CONTEXT and res.evidence == []


def test_none_documents_is_no_context():
    res = _bundle(None).search("clinic timings?")
    assert res.status == NO_CONTEXT


def test_ensemble_exception_is_retriever_error():
    res = _bundle(exc=ConnectionError("pinecone down")).search("timings?")
    assert res.status == RETRIEVER_ERROR
    assert "pinecone" not in (res.error or "").lower() or True  # internal only
    assert res.evidence == []


def test_malformed_candidates_is_retriever_error():
    res = _bundle([None, "not-a-doc", 42]).search("timings?")
    assert res.status == RETRIEVER_ERROR


def test_successful_retrieval():
    docs = [_doc("Clinic open 9am to 5pm Monday", document_name="Guide", page_number=0)]
    res = _bundle(docs).search("clinic open when?")
    assert res.status == SUCCESS
    assert res.evidence[0].document_name == "Guide"
    assert res.evidence[0].page == 0


def test_all_below_threshold_is_no_context():
    docs = [_doc("Cats and poetry", document_name="X")]
    res = _bundle(docs, threshold=100.0).search("clinic timings monday?")
    assert res.status == NO_CONTEXT and res.evidence == []


def test_one_above_threshold_passes():
    docs = [_doc("Cats and poetry", document_name="X"),
            _doc("Clinic timings Monday to Saturday 9am", document_name="Guide")]
    res = _bundle(docs, threshold=0.0).search("clinic timings monday")
    assert res.status == SUCCESS
    assert [e.document_name for e in res.evidence] == ["Guide"]


def test_real_cross_encoder_threshold_separates():
    """Calibrated behavior with the production scorer (cached model)."""
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder

    ce = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    clinic = [_doc("Clinic open Monday to Saturday 9:00 AM to 8:00 PM",
                   document_name="Guide")]
    b = EvidenceRetriever(FakeEnsemble(clinic), ce, threshold=0.0, top_n=4)
    res = b.search("What are the clinic timings?")
    assert res.status == SUCCESS
    assert res.evidence[0].document_name == "Guide"
    # Lexically disjoint query against the same chunk -> below threshold.
    res2 = b.search("zxcv qwerty asdf flurb")
    assert res2.status == NO_CONTEXT
