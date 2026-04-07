from __future__ import annotations

import re
from dataclasses import replace

from llm_speaker_core.retrieval.schemas import ChunkRecord, DocumentRecord

HEADING_RE = re.compile(r"^(#{1,6}\s+.+|[А-ЯA-Z][^.!?]{3,80}:)$")
LIST_RE = re.compile(r"^\s*([-*•]|\d+\.)\s+")


def _estimate_tokens(text: str) -> int:
    return max(1, len(re.findall(r"[A-Za-zА-Яа-яЁё0-9_]+", text)))


def split_blocks(text: str) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    current_heading = ""
    current_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if current_lines:
                blocks.append((current_heading, " ".join(current_lines).strip()))
                current_lines = []
            continue
        if HEADING_RE.match(line):
            if current_lines:
                blocks.append((current_heading, " ".join(current_lines).strip()))
                current_lines = []
            current_heading = line.lstrip("#").strip()
            continue
        if LIST_RE.match(line):
            current_lines.append(re.sub(r"^\s*([-*•]|\d+\.)\s+", "", line))
        else:
            current_lines.append(line)
    if current_lines:
        blocks.append((current_heading, " ".join(current_lines).strip()))
    return [(heading, block) for heading, block in blocks if block]


def chunk_document(
    doc: DocumentRecord,
    min_tokens: int | None = None,
    max_tokens: int | None = None,
    overlap_tokens: int | None = None,
) -> list[ChunkRecord]:
    if min_tokens is None or max_tokens is None or overlap_tokens is None:
        if doc.source_type == "web":
            min_tokens = 120
            max_tokens = 220
            overlap_tokens = 30
        else:
            min_tokens = 250
            max_tokens = 400
            overlap_tokens = 60
    blocks = split_blocks(doc.text)
    chunks: list[ChunkRecord] = []
    buffer: list[str] = []
    buffer_heading = doc.section
    buffer_tokens = 0
    prev_tail: list[str] = []
    chunk_index = 0

    for heading, block in blocks:
        block_tokens = _estimate_tokens(block)
        active_heading = heading or buffer_heading or doc.section
        if buffer and buffer_tokens + block_tokens > max_tokens:
            combined = " ".join([*prev_tail, *buffer]).strip()
            token_count = _estimate_tokens(combined)
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{doc.doc_id}:{chunk_index}",
                    doc_id=doc.doc_id,
                    source_url=doc.source_url,
                    canonical_url=doc.canonical_url,
                    source_type=doc.source_type,
                    title=doc.title,
                    section=buffer_heading or doc.section,
                    published_at=doc.published_at,
                    language=doc.language,
                    content_hash=doc.content_hash,
                    chunk_index=chunk_index,
                    token_count=token_count,
                    char_count=len(combined),
                    overlap_from_prev=min(overlap_tokens, len(" ".join(prev_tail).split())),
                    ocr_used=doc.ocr_used,
                    quality_score=doc.quality_score,
                    text=combined,
                    section_path=[doc.section, buffer_heading or doc.section],
                    metadata=dict(doc.metadata),
                    quality_flags=list(doc.quality_flags),
                    source=doc.canonical_url,
                    is_archived=doc.is_archived,
                    is_low_signal=doc.is_low_signal,
                )
            )
            chunk_index += 1
            words = combined.split()
            prev_tail = words[-overlap_tokens:] if overlap_tokens > 0 else []
            buffer = []
            buffer_tokens = 0

        buffer_heading = active_heading
        buffer.append(block)
        buffer_tokens += block_tokens

    if buffer:
        combined = " ".join([*prev_tail, *buffer]).strip()
        token_count = _estimate_tokens(combined)
        if token_count >= min_tokens or not chunks:
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{doc.doc_id}:{chunk_index}",
                    doc_id=doc.doc_id,
                    source_url=doc.source_url,
                    canonical_url=doc.canonical_url,
                    source_type=doc.source_type,
                    title=doc.title,
                    section=buffer_heading or doc.section,
                    published_at=doc.published_at,
                    language=doc.language,
                    content_hash=doc.content_hash,
                    chunk_index=chunk_index,
                    token_count=token_count,
                    char_count=len(combined),
                    overlap_from_prev=min(overlap_tokens, len(" ".join(prev_tail).split())),
                    ocr_used=doc.ocr_used,
                    quality_score=doc.quality_score,
                    text=combined,
                    section_path=[doc.section, buffer_heading or doc.section],
                    metadata=dict(doc.metadata),
                    quality_flags=list(doc.quality_flags),
                    source=doc.canonical_url,
                    is_archived=doc.is_archived,
                    is_low_signal=doc.is_low_signal,
                )
            )
        elif chunks:
            last = chunks[-1]
            merged = f"{last.text} {combined}".strip()
            chunks[-1] = replace(
                last,
                text=merged,
                token_count=_estimate_tokens(merged),
                char_count=len(merged),
            )
    return chunks
