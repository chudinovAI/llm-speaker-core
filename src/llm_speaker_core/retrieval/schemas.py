from __future__ import annotations

from dataclasses import asdict, dataclass, field
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_path(cls, path: Path) -> "IndexManifest":
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(**payload)
