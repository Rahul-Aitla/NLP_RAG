"""P1.12/P1.13/P1.20 — ingestion diff, manifest, corpus-mismatch detection."""
import json
import os

from src.corpus import (
    check_corpus,
    check_versions,
    diff_chunks,
    diff_corpus_docs,
    load_manifest,
    save_manifest,
)


def test_diff_idempotent_rerun_is_empty():
    existing = {"a#p0#c0": "h1", "a#p0#c1": "h2"}
    desired = {"a#p0#c0": "h1", "a#p0#c1": "h2"}
    assert diff_chunks(existing, desired) == {"to_add": [], "to_update": [], "to_delete": []}


def test_diff_new_changed_deleted():
    existing = {"keep": "h1", "chg": "old", "gone": "h9"}
    desired = {"keep": "h1", "chg": "new", "fresh": "h3"}
    d = diff_chunks(existing, desired)
    assert d == {"to_add": ["fresh"], "to_update": ["chg"], "to_delete": ["gone"]}


def test_diff_legacy_index_without_hashes_rebuilds():
    existing = {"random-uuid-1": "", "random-uuid-2": ""}
    desired = {"a#p0#c0": "h1"}
    d = diff_chunks(existing, desired)
    assert d["to_delete"] == ["random-uuid-1", "random-uuid-2"]
    assert d["to_add"] == ["a#p0#c0"]


def test_doc_summary_counts(tmp_path):
    scan = {"a.pdf": {"hash": "H1"}, "b.pdf": {"hash": "H2"}, "c.pdf": {"hash": "H3"}}
    man = {"a.pdf": {"hash": "H1"}, "b.pdf": {"hash": "OLD"}, "d.pdf": {"hash": "H9"}}
    assert diff_corpus_docs(man, scan) == {
        "new": ["c.pdf"], "changed": ["b.pdf"],
        "unchanged": ["a.pdf"], "deleted": ["d.pdf"]}


def test_manifest_roundtrip(tmp_path):
    p = str(tmp_path / "m.json")
    man = {"manifest_version": 1, "docs": {"a.pdf": {"hash": "H1"}}}
    save_manifest(p, man)
    assert load_manifest(p) == man
    assert load_manifest(str(tmp_path / "nope.json")) is None


def test_corpus_match_and_mismatch(tmp_path):
    f = tmp_path / "a.pdf"
    f.write_bytes(b"%PDF-1.4 fake")
    import hashlib

    h = hashlib.sha256(b"%PDF-1.4 fake").hexdigest()
    man = {"docs": {"a.pdf": {"hash": h}}}
    scan = {"a.pdf": {"name": "a.pdf", "hash": h}}
    ok, _ = check_corpus(man, scan)
    assert ok is True
    f.write_bytes(b"%PDF-1.4 CHANGED")
    scan2 = {"a.pdf": {"name": "a.pdf",
                       "hash": hashlib.sha256(b"%PDF-1.4 CHANGED").hexdigest()}}
    ok2, reason = check_corpus(man, scan2)
    assert ok2 is False and "a.pdf" in reason


def test_corpus_added_file_detected():
    man = {"docs": {"a.pdf": {"hash": "H1"}}}
    scan = {"a.pdf": {"hash": "H1"}, "b.pdf": {"hash": "H2"}}
    ok, reason = check_corpus(man, scan)
    assert ok is False and "b.pdf" in reason


def test_missing_manifest_is_unverified_not_fatal():
    ok, reason = check_corpus(None, {"a.pdf": {"hash": "H"}})
    assert ok is True and "unverified" in reason


def test_version_mismatch_detected():
    man = {"embedding_model": "model-A", "chunking_version": "v1"}
    ok, reason = check_versions(man, "model-B", "v1")
    assert ok is False and "embedding_model" in reason
    ok2, _ = check_versions(man, "model-A", "v1")
    assert ok2 is True
