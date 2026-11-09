"""Deterministic idempotent ingestion into Pinecone (P1.12).

Same document + same configuration -> no duplicate vectors.
Modified document -> old chunks replaced. Deleted document -> vectors removed.
Re-running -> idempotent (second run prints all zeros).

Every chunk carries identity metadata (P1.3): document_id, document_name,
page_number, chunk_id, document_hash, chunk_hash, ingestion_version,
chunking_version, embedding_model, indexed_at.

Usage:
    python store_index.py [--data data/] [--index health-care-chatbot]
"""
from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

from src.corpus import (
    diff_chunks,
    diff_corpus_docs,
    load_manifest,
    save_manifest,
    scan_corpus,
    utcnow_iso,
)
from src.helper import (
    CHUNKING_VERSION,
    EMBEDDING_MODEL,
    INGESTION_VERSION,
    download_hugging_face_embeddings,
    enrich_docs,
    load_pdf_file,
    text_split,
)

INDEX_DIM = 384


def build_desired(data_dir: str, chunk_size: int, chunk_overlap: int):
    """Run the ingestion recipe -> {chunk_id: (chunk_hash, Document)}."""
    extracted = load_pdf_file(data_dir)
    enriched = enrich_docs(extracted, INGESTION_VERSION)
    chunks = text_split(enriched, chunk_size, chunk_overlap, CHUNKING_VERSION)
    desired = {}
    for ch in chunks:
        cid = ch.metadata["chunk_id"]
        desired[cid] = (ch.metadata["chunk_hash"], ch)
    return desired


def plan_sync(existing_hashes: dict, desired: dict, manifest_docs: dict,
              scan: dict) -> dict:
    """Pure sync plan (unit-testable): chunk diff + doc summary."""
    desired_hashes = {cid: h for cid, (h, _) in desired.items()}
    chunk_diff = diff_chunks(existing_hashes, desired_hashes)
    return {"chunks": chunk_diff, "docs": diff_corpus_docs(manifest_docs, scan)}


def _page_ids(page) -> list:
    """Normalize index.list() pagination shapes across client versions.

    pinecone-client v7 yields plain lists of id strings; older shapes yield
    dicts with a 'vectors' entry. Returns a flat list of id strings.
    """
    ids = []
    vecs = page.get("vectors") if isinstance(page, dict) else page
    for v in vecs or []:
        if isinstance(v, str):
            ids.append(v)
        elif isinstance(v, dict) and v.get("id"):
            ids.append(v["id"])
        elif getattr(v, "id", None):
            ids.append(v.id)
    return ids


def _fetched_vectors(fetched) -> dict:
    vecs = getattr(fetched, "vectors", None)
    if vecs is None and isinstance(fetched, dict):
        vecs = fetched.get("vectors") or {}
    return vecs or {}


def list_existing_hashes(index) -> dict:
    """Fetch {chunk_id: chunk_hash} for everything currently indexed."""
    ids = []
    for page in index.list():
        ids.extend(_page_ids(page))
    out = {}
    for i in range(0, len(ids), 100):
        batch = ids[i:i + 100]
        try:
            fetched = index.fetch(ids=batch)
        except Exception:
            continue
        for vid, rec in _fetched_vectors(fetched).items():
            meta = getattr(rec, "metadata", None)
            if meta is None and isinstance(rec, dict):
                meta = rec.get("metadata") or {}
            out[vid] = (meta or {}).get("chunk_hash", "")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Idempotent Pinecone ingestion (P1.12).")
    ap.add_argument("--data", default="data/")
    ap.add_argument("--index", default="health-care-chatbot")
    ap.add_argument("--manifest", default="data/index_manifest.json")
    ap.add_argument("--chunk-size", type=int, default=1000)
    ap.add_argument("--chunk-overlap", type=int, default=150)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    load_dotenv()
    pinecone_key = os.environ.get("PINECONE_API_KEY")
    if not pinecone_key:
        print("ERROR: PINECONE_API_KEY is required.", file=sys.stderr)
        return 2

    scan = scan_corpus(args.data)
    if not scan:
        print(f"ERROR: no PDFs found in {args.data}; refusing to wipe the index.",
              file=sys.stderr)
        return 2
    desired = build_desired(args.data, args.chunk_size, args.chunk_overlap)
    manifest = load_manifest(args.manifest) or {}
    manifest_docs = manifest.get("docs") or {}

    if (manifest.get("embedding_model") not in (None, EMBEDDING_MODEL)
            or manifest.get("chunking_version") not in (None, CHUNKING_VERSION)):
        print(
            f"NOTE: manifest versions {manifest.get('embedding_model')}/"
            f"{manifest.get('chunking_version')} differ from code "
            f"{EMBEDDING_MODEL}/{CHUNKING_VERSION}; full rebuild planned."
        )

    from pinecone import Pinecone, ServerlessSpec

    pc = Pinecone(api_key=pinecone_key)
    if not pc.has_index(args.index):
        pc.create_index(
            name=args.index, dimension=INDEX_DIM, metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    index = pc.Index(args.index)

    existing = list_existing_hashes(index)
    plan = plan_sync(existing, desired, manifest_docs, scan)
    cd, dd = plan["chunks"], plan["docs"]
    print(f"Documents scanned: {len(scan)}")
    print(f"New: {len(dd['new'])} Changed: {len(dd['changed'])} "
          f"Unchanged: {len(dd['unchanged'])} Deleted: {len(dd['deleted'])}")
    print(f"Chunks added: {len(cd['to_add'])} "
          f"Chunks replaced: {len(cd['to_update'])} "
          f"Chunks removed: {len(cd['to_delete'])}")
    if args.dry_run:
        print("dry-run: index untouched.")
        return 0

    if cd["to_delete"]:
        index.delete(ids=cd["to_delete"])
    upserts = cd["to_add"] + cd["to_update"]
    if upserts:
        from langchain_pinecone import PineconeVectorStore

        embeddings = download_hugging_face_embeddings()
        store = PineconeVectorStore(index=index, embedding=embeddings)
        docs, ids, metadatas = [], [], []
        for cid in upserts:
            _, doc = desired[cid]
            meta = dict(doc.metadata)
            meta.update({
                "embedding_model": EMBEDDING_MODEL,
                "indexed_at": utcnow_iso(),
                "doc_version": meta.get("doc_version", ""),
                "effective_date": meta.get("effective_date", ""),
            })
            docs.append(doc.page_content)
            ids.append(cid)
            metadatas.append(meta)
        store.add_texts(docs, ids=ids, metadatas=metadatas)

    new_manifest = {
        "manifest_version": 1,
        "embedding_model": EMBEDDING_MODEL,
        "chunking_version": CHUNKING_VERSION,
        "ingestion_version": INGESTION_VERSION,
        "indexed_at": utcnow_iso(),
        "index": args.index,
        "docs": {
            doc_id: {
                "name": info["name"],
                "hash": info["hash"],
                "chunk_ids": sorted(
                    cid for cid, (_, d) in desired.items()
                    if d.metadata.get("document_id") == doc_id
                ),
            }
            for doc_id, info in scan.items()
        },
    }
    save_manifest(args.manifest, new_manifest)
    print(f"Manifest written to {args.manifest}. Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
