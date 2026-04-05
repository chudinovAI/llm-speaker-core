from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DocumentRecord:
    doc_id: str
    source_url: str
    canonical_url: str
    source_type: str
    title: str
    section: str
    published_at: str | None
    language: str
    crawl_job_id: str | None
    content_hash: str
    extraction_mode: str
    ocr_used: bool
    quality_score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    quality_flags: list[str] = field(default_factory=list)
    is_navigation: bool = False
    is_boilerplate_heavy: bool = False
    is_low_text: bool = False
    is_duplicate: bool = False
    is_archived: bool = False
    needs_ocr: bool = False
    is_foreign_language: bool = False
    is_low_signal: bool = False


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    source_url: str
    canonical_url: str
    source_type: str
    title: str
    section: str
    published_at: str | None
    language: str
    content_hash: str
    chunk_index: int
    token_count: int
    char_count: int
    overlap_from_prev: int
    ocr_used: bool
    quality_score: float
    text: str
    section_path: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    quality_flags: list[str] = field(default_factory=list)
    source: str = ""
    is_archived: bool = False
    is_low_signal: bool = False

    def hit_metadata(self) -> dict[str, object]:
        return {
            "section_path": list(self.section_path),
            "title": self.title,
            "section": self.section,
            "canonical_url": self.canonical_url,
            "source_url": self.source_url,
            "published_at": self.published_at,
            "page_type": self.metadata.get("page_type"),
            "source_facets": list(self.metadata.get("source_facets", [])),
            "quality_score": self.quality_score,
            "quality_flags": list(self.quality_flags),
            "is_archived": self.is_archived,
            "is_low_signal": self.is_low_signal,
        }


@dataclass
class RetrievalHit:
    chunk_id: str
    doc_id: str
    source: str
    text: str
    score: float
    retrieval_stage: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidencePack:
    query: str
    hits: list[RetrievalHit]
    token_budget: int
    grounding_score: float


@dataclass
class IndexManifest:
    version: str
    corpus_checksum: str
    lexical_path: str
    dense_path: str | None
    reranker_model: str
    embedding_model: str
    built_at: str
    doc_count: int
    chunk_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_path(cls, path: Path) -> "IndexManifest":
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(**payload)
