from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from llm_speaker_core.clean_data import run_cleaning
from llm_speaker_core.rag import tokenize
from llm_speaker_core.settings import SETTINGS


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
    index_report = build_index(cleaned_rows, index_path)
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
    report = build_index(rows, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
