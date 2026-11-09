"""P1.3/P1.4/P1.15/P1.16/P1.20 — metadata chain, citations, grounding."""
from langchain_core.documents import Document

from src.evidence import Evidence, evidence_from_doc, format_cited_answer, grounding_ok
from src.helper import enrich_docs, text_split


def _doc(text, **meta):
    return Document(page_content=text, metadata=meta)


def test_metadata_survives_page_to_evidence():
    pages = [_doc("Guide text here.", source="data/Clinic.pdf", page=3)]
    chunks = text_split(enrich_docs(pages), chunk_size=1000, chunk_overlap=150)
    assert chunks, "expected at least one chunk"
    ch = chunks[0]
    assert ch.metadata["page_number"] == 3
    assert ch.metadata["document_name"] == "Clinic.pdf"
    assert ch.metadata["chunk_id"].startswith("Clinic.pdf#p3#c")
    assert len(ch.metadata["chunk_hash"]) == 64
    ev = evidence_from_doc(ch, 5.0)
    assert ev.page == 3 and ev.document_name == "Clinic.pdf"
    assert ev.chunk_id == ch.metadata["chunk_id"]


def test_legacy_chunk_without_metadata_gets_safe_defaults():
    ev = evidence_from_doc(_doc("old text"), 1.0)
    assert ev.page is None and ev.document_name == "clinic document"


def test_valid_citation_single_source():
    ev = Evidence(text="Hours 9-5.", score=5.0, document_name="Guide", page=0)
    out = format_cited_answer("Open 9-5.", [ev])
    assert "Sources:" in out and "Guide, page 0" in out


def test_multiple_sources_all_cited():
    evs = [Evidence(text="a", score=5.0, document_name="Guide", page=0),
           Evidence(text="b", score=4.0, document_name="Fees", page=1)]
    out = format_cited_answer("Answer.", evs)
    assert "Guide, page 0" in out and "Fees, page 1" in out


def test_unknown_page_cites_document_without_inventing():
    ev = Evidence(text="a", score=5.0, document_name="Guide", page=None)
    out = format_cited_answer("Answer.", [ev])
    assert "- Guide" in out and "page" not in out.split("Sources:")[1]


def test_multi_document_note_present():
    evs = [Evidence(text="a", score=5.0, document_name="A", page=0, document_id="a"),
           Evidence(text="b", score=4.0, document_name="B", page=1, document_id="b")]
    assert "more than one document" in format_cited_answer("X.", evs)


def test_grounding_accepts_supported_numbers():
    ev = Evidence(text="Hours 9:00 AM to 8:00 PM, fee 500.", score=5.0)
    assert grounding_ok("Open 9 AM to 8 PM, fee 500.", [ev]) is True


def test_grounding_rejects_invented_numbers():
    ev = Evidence(text="Hours 9:00 AM to 8:00 PM.", score=5.0)
    assert grounding_ok("Open 9 AM to 8 PM, fee 999.", [ev]) is False


def test_grounding_rejects_empty_evidence():
    assert grounding_ok("Open 9-5.", []) is False
