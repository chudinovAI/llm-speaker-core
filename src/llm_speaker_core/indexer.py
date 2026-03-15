from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

from llm_speaker_core.clean_data import run_cleaning
from llm_speaker_core.rag import tokenize
from llm_speaker_core.settings import SETTINGS

LOW_SIGNAL_DOC_SOURCE_RE = re.compile(
    r"(?:^|_)(sbor\d+(?:_\d+)?|sputnik|program\d+|20\d{2}_\d+)\.pdf$"
)
HIGH_SIGNAL_DOC_SOURCE_HINTS = (
    "pravila",
    "priem",
    "abitur",
    "dogovor",
    "stoim",
    "oplata",
    "contact",
    "adress",
)
TUITION_PHRASES = (
    "стоимость обучения",
    "оплата обучения",
    "платное обучение",
    "форма договора",
    "образовательный кредит",
)
ARCHIVE_YEAR_CUTOFF = 2020
ARCHIVE_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
ARCHIVE_INTENTS = {"tuition", "admission", "contacts"}


def _prefix_count(tokens: set[str], stems: tuple[str, ...]) -> int:
    return sum(1 for token in tokens if any(token.startswith(stem) for stem in stems))


def _has_path_hint(path: str, hints: tuple[str, ...]) -> bool:
    return any(h in path for h in hints)


def load_cleaned_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Cleaned corpus not found: {path}")

    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _extract_year_candidates(row: dict) -> list[int]:
    candidates: list[int] = []
    date_value = str(row.get("date") or "").strip()
    if date_value.isdigit():
        candidates.append(int(date_value))

    text = str(row.get("text", ""))
    title = str(row.get("title", ""))
    source = str(row.get("source", ""))
    for match in ARCHIVE_YEAR_RE.findall(f"{title} {text} {source}"):
        year = int(match)
        if 1990 <= year <= 2100:
            candidates.append(year)
    return candidates


def _is_archived_row(row: dict, cutoff_year: int = ARCHIVE_YEAR_CUTOFF) -> bool:
    years = _extract_year_candidates(row)
    if not years:
        return False
    return max(years) < cutoff_year


def build_index(cleaned_rows: list[dict], index_path: Path) -> dict:
    docs: list[dict] = []
    doc_tfs: list[dict[str, int]] = []
    doc_lens: list[int] = []
    df_counter: Counter[str] = Counter()

    for row in cleaned_rows:
        text = row.get("text", "")
        tokens = tokenize(text)
        tf = Counter(tokens)

        docs.append(
            {
                "id": row.get("id"),
                "source": row.get("source"),
                "source_type": row.get("source_type"),
                "title": row.get("title"),
                "date": row.get("date"),
                "is_archived": bool(row.get("is_archived", False)),
                "text": text,
            }
        )

        doc_tfs.append(dict(tf))
        doc_lens.append(sum(tf.values()))

        for token in tf.keys():
            df_counter[token] += 1

    payload = {
        "docs": docs,
        "doc_tfs": doc_tfs,
        "doc_lens": doc_lens,
        "doc_freqs": dict(df_counter),
    }

    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return {
        "index_path": str(index_path),
        "doc_count": len(docs),
        "vocab_size": len(df_counter),
    }


def _row_intent_tags(row: dict) -> set[str]:
    source_type = str(row.get("source_type", "")).lower()
    source = str(row.get("source", "")).lower()
    path = urlparse(source).path.lower() if source.startswith("http") else source
    text = str(row.get("text", "")).lower()
    title = str(row.get("title", "")).lower()
    combined = f"{title} {text[:2600]} {path}"
    tokens = set(tokenize(combined))
    is_low_signal_doc = (
        source_type == "doc"
        and LOW_SIGNAL_DOC_SOURCE_RE.search(path)
        and not any(h in path for h in HIGH_SIGNAL_DOC_SOURCE_HINTS)
    )

    tags: set[str] = set()
    admission_score = 0
    if _has_path_hint(path, ("/priem", "/abitur", "/admission")):
        admission_score += 2
    admission_score += min(
        _prefix_count(
            tokens, ("поступ", "абитур", "приемн", "приёмн", "балл", "документ")
        ),
        2,
    )
    if _prefix_count(tokens, ("комисси", "егэ", "вступит", "направлен")) >= 2:
        admission_score += 1
    if is_low_signal_doc and admission_score < 3:
        admission_score = 0
    if admission_score >= 2:
        tags.add("admission")

    contacts_score = 0
    if _has_path_hint(path, ("/contacts", "/contact", "/priem", "/sveden/common")):
        contacts_score += 2
    contacts_score += min(
        _prefix_count(
            tokens,
            ("контакт", "телефон", "email", "почт", "приемн", "приёмн", "комисси"),
        ),
        2,
    )
    if _prefix_count(tokens, ("горяч", "линия", "адрес")) >= 2:
        contacts_score += 1
    if is_low_signal_doc and contacts_score < 3:
        contacts_score = 0
    if contacts_score >= 2:
        tags.add("contacts")

    location_score = 0
    if _has_path_hint(path, ("/contacts", "/sveden/common", "/map", "/address")):
        location_score += 2
    location_score += min(
        _prefix_count(tokens, ("адрес", "наход", "корпус", "кампус")), 2
    )
    if location_score >= 2:
        tags.add("location")

    student_life_score = 0
    if _has_path_hint(path, ("/studlife", "/students", "/clubs", "/sport")):
        student_life_score += 2
    student_life_score += min(
        _prefix_count(
            tokens, ("студен", "круж", "клуб", "театр", "спорт", "волонтер", "актив")
        ),
        2,
    )
    if student_life_score >= 2:
        tags.add("student_life")

    tuition_score = 0
    if _has_path_hint(
        path, ("/eif/pay", "/eif/inf_dog", "/pay", "/tuition", "/priem", "/abitur")
    ):
        tuition_score += 2
    tuition_score += min(
        _prefix_count(
            tokens,
            ("стоим", "оплат", "платн", "договор", "контракт", "внебюджет", "кредит"),
        ),
        2,
    )
    if _prefix_count(tokens, ("обучен", "бакалавр", "магистр", "специалит")) >= 2:
        tuition_score += 1
    if any(phrase in text for phrase in TUITION_PHRASES):
        tuition_score += 2
    if is_low_signal_doc and tuition_score < 4:
        tuition_score = 0
    if tuition_score >= 2:
        tags.add("tuition")

    if _prefix_count(tokens, ("положени", "приказ", "регламент", "правил")) >= 1:
        tags.add("policies")
    if _prefix_count(tokens, ("новост", "мероприят", "конференц", "анонс")) >= 1:
        tags.add("news")
    if not tags:
        tags.add("general")
    return tags


def build_multi_index(cleaned_rows: list[dict], index_path: Path) -> dict:
    groups: dict[str, list[dict]] = {
        "general": cleaned_rows,
        "admission": [],
        "contacts": [],
        "location": [],
        "student_life": [],
        "tuition": [],
        "policies": [],
        "news": [],
    }

    for row in cleaned_rows:
        row_with_flags = dict(row)
        row_with_flags["is_archived"] = _is_archived_row(row_with_flags)
        tags = _row_intent_tags(row)
        for tag in tags:
            if tag == "general":
                continue
            if tag in ARCHIVE_INTENTS and row_with_flags["is_archived"]:
                continue
            groups.setdefault(tag, []).append(row_with_flags)

    index_reports: dict[str, dict] = {}
    index_map: dict[str, str] = {}
    stem = index_path.stem
    suffix = index_path.suffix or ".json"
    out_dir = index_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, rows in groups.items():
        sub_path = out_dir / f"{stem}.{name}{suffix}"
        report = build_index(rows, sub_path)
        index_reports[name] = report
        index_map[name] = sub_path.name

    manifest = {
        "format": "multi_v1",
        "indexes": index_map,
    }
    index_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "index_path": str(index_path),
        "format": "multi_v1",
        "indexes": {
            k: {"doc_count": v["doc_count"], "path": str(out_dir / index_map[k])}
            for k, v in index_reports.items()
        },
    }


def run_nightly_reindex(
    raw_path: Path,
    cleaned_path: Path,
    report_path: Path,
    index_path: Path,
    min_words: int = 8,
) -> dict:
    rows, cleaning_report = run_cleaning(
        raw_corpus_path=raw_path,
        cleaned_corpus_path=cleaned_path,
        report_path=report_path,
        min_words=min_words,
    )
    cleaned_rows = load_cleaned_rows(cleaned_path)
    index_report = build_multi_index(cleaned_rows, index_path)
    return {
        "cleaned_rows": rows,
        "cleaning_report": cleaning_report,
        "index_report": index_report,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build lexical RAG index.")
    parser.add_argument("--input", type=Path, default=SETTINGS.cleaned_corpus_path)
    parser.add_argument("--output", type=Path, default=SETTINGS.index_path)
    parser.add_argument(
        "--nightly-reindex",
        action="store_true",
        help="Run cleaning + indexing in one command.",
    )
    parser.add_argument("--raw", type=Path, default=SETTINGS.raw_corpus_path)
    parser.add_argument("--report", type=Path, default=SETTINGS.quality_report_path)
    parser.add_argument("--min-words", type=int, default=8)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.nightly_reindex:
        report = run_nightly_reindex(
            raw_path=args.raw,
            cleaned_path=args.input,
            report_path=args.report,
            index_path=args.output,
            min_words=args.min_words,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    rows = load_cleaned_rows(args.input)
    report = build_multi_index(rows, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
