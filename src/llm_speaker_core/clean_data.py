from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from llm_speaker_core.settings import SETTINGS

SOURCE_RE = re.compile(r"^---\s*Источник:\s*(https?://\S+)\s*---\s*$")
DOC_RE = re.compile(r"^---\s*Источник\s+Документ:\s*(.+?)\s*---\s*$")
DATE_RE = re.compile(r"\b(20\d{2}|19\d{2})\b")
NOISE_PATTERNS = (
    re.compile(r"^подробнее$", re.IGNORECASE),
    re.compile(r"^главная$", re.IGNORECASE),
    re.compile(r"^обучающимся$", re.IGNORECASE),
    re.compile(r"^абитуриентам$", re.IGNORECASE),
    re.compile(r"^сотрудникам$", re.IGNORECASE),
    re.compile(r"^выпускникам$", re.IGNORECASE),
    re.compile(r"^телефон", re.IGNORECASE),
    re.compile(r"^e-?mail", re.IGNORECASE),
    re.compile(r"^контакты$", re.IGNORECASE),
    re.compile(r"^расписание занятий$", re.IGNORECASE),
    re.compile(r"^приемная комиссия$", re.IGNORECASE),
    re.compile(r"^структура и органы управления$", re.IGNORECASE),
)


@dataclass
class Section:
    source_type: str
    source: str
    lines: list[str]


@dataclass
class Stats:
    sections_total: int = 0
    sections_web: int = 0
    sections_doc: int = 0
    chunks_raw: int = 0
    chunks_after_filters: int = 0
    duplicates_removed: int = 0
    too_short_removed: int = 0
    empty_removed: int = 0
    noise_lines_removed: int = 0


def _normalize_line(line: str) -> str:
    line = line.replace("\xa0", " ")
    line = re.sub(r"\s+", " ", line.strip())
    return line


def parse_raw_sections(raw_text: str) -> list[Section]:
    current: Section | None = None
    result: list[Section] = []

    for original_line in raw_text.splitlines():
        line = original_line.rstrip("\n")
        m_source = SOURCE_RE.match(line)
        m_doc = DOC_RE.match(line)

        if m_source:
            if current:
                result.append(current)
            current = Section(source_type="web", source=m_source.group(1), lines=[])
            continue

        if m_doc:
            if current:
                result.append(current)
            current = Section(source_type="doc", source=m_doc.group(1), lines=[])
            continue

        if current is None:
            continue

        current.lines.append(line)

    if current:
        result.append(current)

    return result


def _is_noise_line(line: str, frequent_lines: set[str]) -> bool:
    if not line:
        return True
    if line in frequent_lines:
        return True
    if len(line) < 2:
        return True
    if line in {"×", "›", "‹", ",", "(812)"}:
        return True
    if re.search(r"https?://|www\.", line, flags=re.IGNORECASE):
        return True
    if re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", line):
        return True
    if re.search(r"(\+7|8)\s*[\-(]?\d{3}", line):
        return True
    if re.fullmatch(r"[\d\W_]+", line):
        return True
    if len(line.split()) > 10 and line.isupper():
        return True
    for pattern in NOISE_PATTERNS:
        if pattern.match(line):
            return True
    return False


def _extract_title(lines: list[str]) -> str:
    for line in lines:
        if len(line) > 7 and not line.isdigit():
            return line[:120]
    return "untitled"


def _extract_date(lines: list[str]) -> str | None:
    joined = " ".join(lines[:8])
    m = DATE_RE.search(joined)
    return m.group(0) if m else None


def _chunk_lines(lines: list[str], chunk_size: int = 8) -> list[str]:
    paragraphs: list[str] = []
    buffer: list[str] = []

    for line in lines:
        if not line:
            if buffer:
                paragraphs.append(" ".join(buffer))
                buffer = []
            continue
        if len(line) > 350:
            if buffer:
                paragraphs.append(" ".join(buffer))
                buffer = []
            paragraphs.append(line)
            continue

        buffer.append(line)
        if len(buffer) >= chunk_size:
            paragraphs.append(" ".join(buffer))
            buffer = []

    if buffer:
        paragraphs.append(" ".join(buffer))

    return paragraphs


def clean_sections(sections: list[Section], min_words: int = 8) -> tuple[list[dict], Stats]:
    stats = Stats()
    stats.sections_total = len(sections)
    stats.sections_web = sum(1 for s in sections if s.source_type == "web")
    stats.sections_doc = sum(1 for s in sections if s.source_type == "doc")

    normalized_lines: list[str] = []
    for section in sections:
        for line in section.lines:
            normalized = _normalize_line(line)
            if normalized:
                normalized_lines.append(normalized)

    line_counter = Counter(normalized_lines)
    frequent_lines = {line for line, count in line_counter.items() if count >= 30}

    unique_hashes: set[str] = set()
    cleaned: list[dict] = []

    for section in sections:
        normalized_section_lines = [_normalize_line(line) for line in section.lines]
        before_count = len([line for line in normalized_section_lines if line])
        filtered_lines = [
            line
            for line in normalized_section_lines
            if not _is_noise_line(line, frequent_lines)
        ]
        stats.noise_lines_removed += max(before_count - len(filtered_lines), 0)

        chunks = _chunk_lines(filtered_lines)
        stats.chunks_raw += len(chunks)

        title = _extract_title(filtered_lines)
        date = _extract_date(filtered_lines)

        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                stats.empty_removed += 1
                continue

            word_count = len(chunk.split())
            if word_count < min_words:
                stats.too_short_removed += 1
                continue

            dedup_key = hashlib.sha256(chunk.lower().encode("utf-8")).hexdigest()
            if dedup_key in unique_hashes:
                stats.duplicates_removed += 1
                continue
            unique_hashes.add(dedup_key)

            cleaned.append(
                {
                    "id": f"{section.source_type}:{dedup_key[:16]}",
                    "source_type": section.source_type,
                    "source": section.source,
                    "title": title,
                    "date": date,
                    "text": chunk,
                    "word_count": word_count,
                }
            )

    stats.chunks_after_filters = len(cleaned)
    return cleaned, stats


def run_cleaning(
    raw_corpus_path: Path,
    cleaned_corpus_path: Path,
    report_path: Path,
    min_words: int,
) -> tuple[int, dict]:
    if not raw_corpus_path.exists():
        raise FileNotFoundError(f"Raw corpus not found: {raw_corpus_path}")

    raw_text = raw_corpus_path.read_text(encoding="utf-8", errors="ignore")
    sections = parse_raw_sections(raw_text)
    cleaned_docs, stats = clean_sections(sections, min_words=min_words)

    cleaned_corpus_path.parent.mkdir(parents=True, exist_ok=True)
    with cleaned_corpus_path.open("w", encoding="utf-8") as f:
        for row in cleaned_docs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "raw_corpus_path": str(raw_corpus_path),
        "cleaned_corpus_path": str(cleaned_corpus_path),
        "sections_total": stats.sections_total,
        "sections_web": stats.sections_web,
        "sections_doc": stats.sections_doc,
        "chunks_raw": stats.chunks_raw,
        "chunks_after_filters": stats.chunks_after_filters,
        "duplicates_removed": stats.duplicates_removed,
        "too_short_removed": stats.too_short_removed,
        "empty_removed": stats.empty_removed,
        "noise_lines_removed": stats.noise_lines_removed,
        "retention_ratio": round(
            stats.chunks_after_filters / stats.chunks_raw, 4
        )
        if stats.chunks_raw
        else 0.0,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return len(cleaned_docs), report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean raw RAG corpus into JSONL.")
    parser.add_argument("--raw", type=Path, default=SETTINGS.raw_corpus_path)
    parser.add_argument("--out", type=Path, default=SETTINGS.cleaned_corpus_path)
    parser.add_argument("--report", type=Path, default=SETTINGS.quality_report_path)
    parser.add_argument("--min-words", type=int, default=8)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    rows, report = run_cleaning(args.raw, args.out, args.report, args.min_words)
    print(f"Clean corpus rows: {rows}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
