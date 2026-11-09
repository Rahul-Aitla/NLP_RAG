"""P1.19 — offline RAG evaluation. No production credentials needed.

Retrieval side uses local BM25 candidates + the real cross-encoder
(downloaded/cached HuggingFace model) through EvidenceRetriever, so the
measured scores/threshold behavior match production. Pinecone/Groq are
never touched.

Run:  python eval/eval_rag.py [--k 4]
Prints a machine-readable JSON block plus a human summary.

LIMITATION (stated, not hidden): the dataset is small (13 retrieval cases
on a 2-page clinic corpus), so recall/MRR are directional smoke signals,
not statistically meaningful benchmarks.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_loaders import PyPDFLoader

from src.evidence import NO_CONTEXT, SUCCESS
from src.helper import CHUNKING_VERSION, enrich_docs, text_split
from src.retriever import EvidenceRetriever, build_offline_bm25

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
CLINIC_PDF = os.path.join(EVAL_DIR, "..", "data", "HealthFirst_Clinic_Info.pdf")


def load_cases():
    with open(os.path.join(EVAL_DIR, "dataset.json"), encoding="utf-8") as fh:
        return json.load(fh)["cases"]


def build_bundle(chunk_size, chunk_overlap, k=20, threshold=0.0):
    pages = PyPDFLoader(CLINIC_PDF).load()
    chunks = text_split(enrich_docs(pages), chunk_size, chunk_overlap, CHUNKING_VERSION)
    bm25 = build_offline_bm25(chunks, k=k)
    ce = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    return EvidenceRetriever(bm25, ce, threshold=threshold, top_n=4), chunks


def evaluate(bundle, cases, k=4):
    per_case, fallback_ok, fallback_total = [], 0, 0
    rr_sum, hits = 0.0, 0
    # Safety cases never reach retrieval in production (guardrail routes
    # first); they are evaluated for NO_CONTEXT here only as a sanity check
    # and excluded from recall/MRR.
    retrievable = [c for c in cases
                   if not c.get("expect_fallback") and not c.get("skip_retrieval")]
    skipped = [c for c in cases if c.get("skip_retrieval")]
    for case in cases:
        res = bundle.search(case["question"])
        if case.get("skip_retrieval"):
            per_case.append({"id": case["id"], "status": res.status,
                             "skipped_from_recall": True})
            continue
        if case.get("expect_fallback"):
            fallback_total += 1
            ok = res.status == NO_CONTEXT
            fallback_ok += ok
            per_case.append({"id": case["id"], "fallback_correct": ok,
                             "status": res.status})
            continue
        pages = [e.page for e in res.evidence[:k]]
        rank = next((i + 1 for i, p in enumerate(pages)
                     if p == case["expected_page"]), None)
        facts = " ".join(e.text for e in res.evidence[:k])
        facts_ok = all(f in facts for f in case.get("expected_facts", []))
        if rank:
            hits += 1
            rr_sum += 1.0 / rank
        per_case.append({"id": case["id"], "status": res.status,
                         "hit_rank": rank, "facts_ok": facts_ok and res.status == SUCCESS})
    n = len(retrievable)
    return {
        "retrieval_cases": n,
        f"recall@{k}": round(hits / n, 3) if n else 0.0,
        "mrr": round(rr_sum / n, 3) if n else 0.0,
        "fallback_accuracy": round(fallback_ok / fallback_total, 3) if fallback_total else 0.0,
        "fallback_cases": fallback_total,
        "per_case": per_case,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=0.0)
    args = ap.parse_args(argv)
    cases = load_cases()
    out = {"threshold": args.threshold, "k": args.k, "limitation": "small dataset (13 retrieval cases, 2-page corpus); directional only"}
    for name, size, overlap in (("old-500-20", 500, 20), ("new-1000-150", 1000, 150)):
        bundle, chunks = build_bundle(size, overlap, threshold=args.threshold)
        rep = evaluate(bundle, cases, k=args.k)
        rep["chunks_indexed"] = len(chunks)
        out[name] = rep
    print(json.dumps(out, indent=2))
    for name in ("old-500-20", "new-1000-150"):
        r = out[name]
        print(f"{name}: chunks={r['chunks_indexed']} recall@{args.k}={r[f'recall@{args.k}']} "
              f"mrr={r['mrr']} fallback_acc={r['fallback_accuracy']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
