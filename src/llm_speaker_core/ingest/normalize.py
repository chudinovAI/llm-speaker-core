from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlparse

from docx import Document as DocxDocument
from pypdf import PdfReader

from llm_speaker_core.ingest.chunking import chunk_document
from llm_speaker_core.ingest.ocr import ocr_pdf
from llm_speaker_core.ingest.extractors.cloudflare import (
    canonicalize_url,
    extract_document_links,
    extract_primary_text,
)
from llm_speaker_core.ingest.quality import assess_document_quality, is_near_duplicate
from llm_speaker_core.ingest.quality import normalized_fingerprint, similarity_signature
from llm_speaker_core.retrieval.schemas import ChunkRecord, DocumentRecord

ARCHIVE_YEAR_RE = re.compile(r"\b(20\d{2})\b")
MARKDOWN_H1_RE = re.compile(r"(?m)^#{1,2}\s+")
FOOTER_MARKERS = (
    "#### почтовый адрес:",
    "почтовый адрес:",
    "##### приемная комиссия",
    "## основные документы гуап",
    "## полезные ресурсы",
    "#### университет",
    "отдел информационно-стратегических коммуникаций и рекламы:",
    "отдел делопроизводства:",
    "сведения об образовательной организации",
    "разработка сайта",
    "вопросы по работе сайта:",
)


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _guess_section(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.strip("/")
    if not path:
        return "home"
    return path.split("/")[0]


def _extract_title(record: dict, fallback_url: str) -> str:
    title = str(
        record.get("title")
        or record.get("metadata", {}).get("title")
        or record.get("head", {}).get("title")
        or ""
    ).strip()
    if title:
        return title
    parsed = urlparse(fallback_url)
    tail = parsed.path.rstrip("/").split("/")[-1]
    return tail or parsed.netloc or "untitled"


def _trim_header_noise(text: str, extraction_mode: str) -> str:
    if extraction_mode != "markdown":
        return text.strip()
    match = MARKDOWN_H1_RE.search(text)
    if match is None:
        return text.strip()
    # Drop breadcrumb/logo/nav noise before the first H1 if it is near the top.
    if match.start() <= 2400:
        return text[match.start() :].strip()
    return text.strip()


def _trim_footer_noise(text: str) -> str:
    low = text.lower()
    cutoff = None
    min_offset = int(len(low) * 0.4)
    for marker in FOOTER_MARKERS:
        idx = low.find(marker, min_offset)
        if idx >= min_offset and (cutoff is None or idx < cutoff):
            cutoff = idx
    if cutoff is None:
        return text.strip()
    return text[:cutoff].strip()


def clean_extracted_text(text: str, extraction_mode: str) -> str:
    cleaned = _trim_header_noise(text, extraction_mode)
    cleaned = _trim_footer_noise(cleaned)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def document_from_cloudflare_record(record: dict, crawl_job_id: str | None = None) -> DocumentRecord | None:
    source_url = str(record.get("url") or record.get("sourceURL") or "").strip()
    if not source_url:
        return None
    canonical_url = canonicalize_url(source_url)
    text, extraction_mode = extract_primary_text(record)
    text = clean_extracted_text(text, extraction_mode)
    title = _extract_title(record, canonical_url)
    doc_id = f"web:{_content_hash(canonical_url)}"
    years = [int(y) for y in ARCHIVE_YEAR_RE.findall(text)]
    published_at = str(record.get("published_at") or record.get("metadata", {}).get("published_at") or "").strip() or None
    is_archived = False
    if published_at and published_at[:4].isdigit():
        is_archived = int(published_at[:4]) < 2020
    elif years:
        is_archived = max(years) < 2020

    doc = DocumentRecord(
        doc_id=doc_id,
        source_url=source_url,
        canonical_url=canonical_url,
        source_type="web",
        title=title,
        section=_guess_section(canonical_url),
        published_at=published_at,
        language="ru",
        crawl_job_id=crawl_job_id,
        content_hash=_content_hash(text or canonical_url),
        extraction_mode=extraction_mode,
        ocr_used=False,
        quality_score=0.0,
        text=text.strip(),
        metadata={
            "document_links": extract_document_links(record, allowed_domain="guap.ru"),
        },
        is_archived=is_archived,
    )
    return assess_document_quality(doc)


def load_cloudflare_documents(raw_records_path: Path, crawl_job_id: str | None = None) -> list[DocumentRecord]:
    documents: list[DocumentRecord] = []
    seen_fingerprints: set[str] = set()
    seen_signatures: set[str] = set()
    with raw_records_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            doc = document_from_cloudflare_record(record, crawl_job_id=crawl_job_id)
            if doc is None:
                continue
            if is_near_duplicate(doc.text, seen_fingerprints, seen_signatures):
                doc.is_duplicate = True
                doc.quality_score = min(doc.quality_score, 0.1)
            else:
                seen_fingerprints.add(normalized_fingerprint(doc.text))
                signature = similarity_signature(doc.text)
                if signature:
                    seen_signatures.add(signature)
            documents.append(doc)
    return documents


def parse_document_file(path: Path) -> tuple[str, str, bool]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        text = "\n".join((page.extract_text() or "") for page in reader.pages).strip()
        needs_ocr = len(text.split()) < 30
        ocr_used = False
        if needs_ocr:
            ocr_text = ocr_pdf(path)
            if ocr_text:
                text = ocr_text
                needs_ocr = False
                ocr_used = True
        return text, "pdf_ocr" if ocr_used else "pdf_text", needs_ocr
    if suffix == ".docx":
        doc = DocxDocument(str(path))
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip()).strip()
        return text, "docx_text", False
    return path.read_text(encoding="utf-8", errors="ignore"), "plain_text", False


def normalize_downloaded_document(path: Path, source_url: str) -> DocumentRecord:
    text, extraction_mode, needs_ocr = parse_document_file(path)
    text = clean_extracted_text(text, extraction_mode)
    canonical_url = canonicalize_url(source_url)
    doc = DocumentRecord(
        doc_id=f"doc:{_content_hash(str(path.resolve()))}",
        source_url=source_url,
        canonical_url=canonical_url,
        source_type="doc",
        title=path.name,
        section=_guess_section(canonical_url),
        published_at=None,
        language="ru",
        crawl_job_id=None,
        content_hash=_content_hash(text or path.name),
        extraction_mode=extraction_mode,
        ocr_used=extraction_mode == "pdf_ocr",
        quality_score=0.0,
        text=text.strip(),
        metadata={"file_name": path.name},
        needs_ocr=needs_ocr,
    )
    return assess_document_quality(doc)


def load_manual_documents(root: Path) -> list[DocumentRecord]:
    documents: list[DocumentRecord] = []
    if not root.exists():
        return documents
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".pdf", ".docx", ".txt"}:
            continue
        relative = path.relative_to(root).as_posix()
        source_url = f"manual://{relative}"
        documents.append(normalize_downloaded_document(path, source_url))
    return documents


def write_documents(path: Path, documents: list[DocumentRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for doc in documents:
            fh.write(json.dumps(asdict(doc), ensure_ascii=False) + "\n")


def write_chunks(path: Path, chunks: list[ChunkRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def build_chunk_corpus(documents: list[DocumentRecord]) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    for doc in documents:
        if doc.is_duplicate or doc.is_low_signal or doc.is_navigation or doc.is_low_text:
            continue
        chunks.extend(chunk_document(doc))
    return chunks
