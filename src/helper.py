"""PDF loading, metadata enrichment (P1.3), chunking (P1.14), embeddings."""
from __future__ import annotations

import hashlib
import os
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dimensions
INGESTION_VERSION = "1"
# P1.14: 1000/150 keeps small fact tables (e.g. the ~400-char doctor roster)
# inside a single chunk. The previous 500/20 was observed splitting doctor
# table rows away from their headers. Overlap preserves row context at
# boundaries. Change with care: bumping this requires reindexing (the
# manifest's chunking_version mismatch will tell operators exactly that).
CHUNKING_VERSION = "v2-1000-150"


# Extract data from the PDF files.
def load_pdf_file(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def enrich_docs(
    docs: List[Document], ingestion_version: str = INGESTION_VERSION
) -> List[Document]:
    """P1.3: attach stable source metadata to every page-level document.

    Preserved per page: document_id (stable, path-based), document_name,
    page_number (0-based, as PyPDFLoader reports), document_hash (sha256 of
    the source file so edits are detectable), ingestion_version.
    """
    enriched: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source", "") or ""
        doc_id = os.path.basename(src)
        try:
            doc_hash = sha256_file(src) if src and os.path.isfile(src) else ""
        except OSError:
            doc_hash = ""
        page = doc.metadata.get("page", doc.metadata.get("page_number"))
        meta = {
            "document_id": doc_id,
            "document_name": os.path.basename(src) or doc_id,
            "page_number": page,
            "document_hash": doc_hash,
            "ingestion_version": ingestion_version,
        }
        enriched.append(Document(page_content=doc.page_content, metadata=meta))
    return enriched


def text_split(
    extracted_data: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    chunking_version: str = CHUNKING_VERSION,
) -> List[Document]:
    """Split into chunks, stamping every chunk with identity metadata (P1.3).

    chunk_id is deterministic per (document, page, ordinal) so re-ingestion
    of unchanged content yields identical ids (P1.12 dedup); chunk_hash
    detects content edits.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(extracted_data)
    counters: dict = {}
    for ch in chunks:
        key = (ch.metadata.get("document_id"), ch.metadata.get("page_number"))
        n = counters.get(key, 0)
        counters[key] = n + 1
        ch.metadata["chunk_id"] = (
            f"{ch.metadata.get('document_id', '?')}#p{ch.metadata.get('page_number', '?')}#c{n}"
        )
        ch.metadata["chunk_hash"] = hashlib.sha256(
            ch.page_content.encode("utf-8")
        ).hexdigest()
        ch.metadata["chunking_version"] = chunking_version
    return chunks


# Download the embeddings from HuggingFace.
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return embeddings
