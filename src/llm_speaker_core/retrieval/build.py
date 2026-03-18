from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_speaker_core.ingest.normalize import (
    build_chunk_corpus,
    load_cloudflare_documents,
    load_manual_documents,
    write_chunks,
    write_documents,
)
from llm_speaker_core.retrieval.service import HybridRetrievalService


def build_hybrid_index(
    raw_records: Path,
    documents_out: Path,
    chunks_out: Path,
    manifest_out: Path,
    lexical_out: Path,
    dense_out: Path,
    embedding_model: str,
    reranker_model: str,
    manual_docs_dir: Path | None = None,
) -> dict:
    documents = load_cloudflare_documents(raw_records)
    if manual_docs_dir is not None:
        documents.extend(load_manual_documents(manual_docs_dir))
    chunks = build_chunk_corpus(documents)
    write_documents(documents_out, documents)
    write_chunks(chunks_out, chunks)
    service = HybridRetrievalService.build_from_chunks(
        chunks=chunks,
        lexical_path=lexical_out,
        dense_path=dense_out,
        manifest_path=manifest_out,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
    )
    return {
        "documents": len(documents),
        "chunks": len(chunks),
        "manifest": str(manifest_out),
        "retrieval_version": service.version,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the new hybrid RAG index from Cloudflare raw records")
    parser.add_argument("--raw-records", type=Path, default=Path("data/raw/cloudflare/latest/records.jsonl"))
    parser.add_argument("--documents-out", type=Path, default=Path("data/normalized/documents.jsonl"))
    parser.add_argument("--chunks-out", type=Path, default=Path("data/normalized/chunks.jsonl"))
    parser.add_argument("--manifest-out", type=Path, default=Path("data/index_manifest.json"))
    parser.add_argument("--lexical-out", type=Path, default=Path("data/indexes/bm25/index.json"))
    parser.add_argument("--dense-out", type=Path, default=Path("data/indexes/faiss/index.json"))
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--manual-docs-dir", type=Path, default=Path("data/raw/manual_docs"))
    args = parser.parse_args()
    report = build_hybrid_index(
        raw_records=args.raw_records,
        documents_out=args.documents_out,
        chunks_out=args.chunks_out,
        manifest_out=args.manifest_out,
        lexical_out=args.lexical_out,
        dense_out=args.dense_out,
        embedding_model=args.embedding_model,
        reranker_model=args.reranker_model,
        manual_docs_dir=args.manual_docs_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
