"""P1.12/P1.13 — deterministic corpus identity: hashing, manifest, diff.

The manifest (data/index_manifest.json) is the source of truth for what is
indexed. Ingestion writes it; application startup compares the live data/
directory against it and refuses to silently serve a drifted corpus.
"""
from __future__ import annotations

import datetime as _dt
import glob
import json
import os

from src.helper import sha256_file


def scan_corpus(data_dir: str = "data/") -> dict:
    """Hash every PDF in data_dir -> {doc_id: {name, path, hash}}."""
    out = {}
    for path in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        doc_id = os.path.basename(path)
        try:
            out[doc_id] = {"name": doc_id, "path": path, "hash": sha256_file(path)}
        except OSError:
            continue
    return out


def load_manifest(path: str) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def save_manifest(path: str, manifest: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


def utcnow_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def check_corpus(manifest: dict | None, scan: dict) -> tuple:
    """Compare live corpus vs manifest.

    Returns (ok, reason). Missing manifest -> (True, 'unverified: ...') so
    pre-manifest indexes keep serving (warn loudly); any content/version
    mismatch with a present manifest -> (False, reason) and the app marks
    retrieval unavailable instead of serving drifted evidence.
    """
    if not manifest:
        return True, "unverified: no index manifest present (legacy index?)"
    man_docs = (manifest.get("docs") or {})
    live_ids, man_ids = set(scan), set(man_docs)
    if live_ids != man_ids:
        added = sorted(live_ids - man_ids)
        removed = sorted(man_ids - live_ids)
        return False, f"corpus mismatch: added={added} removed={removed}; re-run ingestion"
    for doc_id in live_ids:
        if scan[doc_id]["hash"] != man_docs[doc_id].get("hash"):
            return False, (
                f"corpus mismatch: '{doc_id}' changed on disk; "
                "re-run ingestion before serving"
            )
    return True, "verified"


def check_versions(manifest: dict | None, embedding_model: str,
                   chunking_version: str) -> tuple:
    """Refuse silently mixing incompatible embedding/chunking generations."""
    if not manifest:
        return True, "unverified: no index manifest present"
    problems = []
    if manifest.get("embedding_model") != embedding_model:
        problems.append(
            f"embedding_model index={manifest.get('embedding_model')} "
            f"code={embedding_model}"
        )
    if manifest.get("chunking_version") != chunking_version:
        problems.append(
            f"chunking_version index={manifest.get('chunking_version')} "
            f"code={chunking_version}"
        )
    if problems:
        return False, "version mismatch: " + "; ".join(problems) + "; re-run ingestion"
    return True, "verified"


def diff_chunks(existing: dict, desired: dict) -> dict:
    """Pure diff for idempotent sync (unit-testable, no Pinecone needed).

    existing: {chunk_id: chunk_hash} currently in the index.
    desired:  {chunk_id: chunk_hash} ingestion wants.
    Returns {to_add: [...], to_update: [...], to_delete: [...]} with sorted,
    deterministic lists.
    """
    existing = existing or {}
    desired = desired or {}
    to_add = sorted(i for i in desired if i not in existing)
    to_update = sorted(i for i in desired if i in existing and existing[i] != desired[i])
    to_delete = sorted(i for i in existing if i not in desired)
    return {"to_add": to_add, "to_update": to_update, "to_delete": to_delete}


def diff_corpus_docs(manifest_docs: dict, scan: dict) -> dict:
    """Document-level new/changed/unchanged/deleted summary for run output."""
    manifest_docs = manifest_docs or {}
    new = sorted(i for i in scan if i not in manifest_docs)
    deleted = sorted(i for i in manifest_docs if i not in scan)
    changed = sorted(
        i for i in scan
        if i in manifest_docs and scan[i]["hash"] != manifest_docs[i].get("hash")
    )
    unchanged = sorted(i for i in scan if i in manifest_docs and i not in changed)
    return {"new": new, "changed": changed, "unchanged": unchanged, "deleted": deleted}
