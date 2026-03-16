from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_speaker_core.retrieval.lexical import LexicalIndex
from llm_speaker_core.retrieval.schemas import ChunkRecord, IndexManifest
from llm_speaker_core.retrieval.service import HybridRetrievalService


def _chunk(
    chunk_id: str,
    text: str,
    source: str,
    section: str,
    *,
    is_archived: bool = False,
) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        doc_id=chunk_id.split(":")[0],
        source_url=source,
        canonical_url=source,
        source_type="web",
        title=section,
        section=section,
        published_at=None,
        language="ru",
        content_hash=chunk_id,
        chunk_index=0,
        token_count=len(text.split()),
        char_count=len(text),
        overlap_from_prev=0,
        ocr_used=False,
        quality_score=1.0,
        text=text,
        section_path=[section],
        metadata={},
        quality_flags=[],
        source=source,
        is_archived=is_archived,
        is_low_signal=False,
    )


def test_hybrid_search_prefers_relevant_official_source() -> None:
    chunks = [
        _chunk(
            "doc1:0",
            "Поступление в ГУАП: абитуриенты подают документы через официальный раздел для абитуриентов.",
            "https://guap.ru/abitur",
            "abitur",
        ),
        _chunk(
            "doc2:0",
            "Контакты университета и адреса корпусов опубликованы в разделе сведений.",
            "https://guap.ru/sveden/common",
            "sveden",
        ),
    ]
    lexical = LexicalIndex.build(chunks)
    manifest = IndexManifest(
        version="hybrid-rag-v2",
        corpus_checksum="abc",
        lexical_path="unused.json",
        dense_path=None,
        reranker_model="",
        embedding_model="",
        built_at="2026-03-15T00:00:00Z",
        doc_count=2,
        chunk_count=2,
        metadata={},
    )
    service = HybridRetrievalService(lexical=lexical, dense=None, reranker=None, manifest=manifest)

    hits = service.search("Как поступить в ГУАП?", top_k=2)

    assert hits
    assert hits[0]["source"] == "https://guap.ru/abitur"


def test_hybrid_loads_from_manifest_without_dense(tmp_path: Path) -> None:
    chunks = [
        _chunk(
            "doc1:0",
            "Стоимость обучения зависит от программы и формы обучения.",
            "https://guap.ru/eif/price",
            "eif",
            is_archived=False,
        )
    ]
    lexical = LexicalIndex.build(chunks)
    lexical_path = tmp_path / "bm25.json"
    lexical.save(lexical_path)
    manifest: dict[str, Any] = {
        "version": "hybrid-rag-v2",
        "corpus_checksum": "checksum",
        "lexical_path": lexical_path.name,
        "dense_path": None,
        "reranker_model": "",
        "embedding_model": "",
        "built_at": "2026-03-15T00:00:00Z",
        "doc_count": 1,
        "chunk_count": 1,
        "metadata": {},
    }
    manifest_path = tmp_path / "index_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    service = HybridRetrievalService.load(manifest_path)
    evidence = service.build_evidence_pack("Сколько стоит обучение в ГУАП?", top_k=1)

    assert evidence.hits
    assert evidence.hits[0].source == "https://guap.ru/eif/price"
    assert evidence.grounding_score > 0
