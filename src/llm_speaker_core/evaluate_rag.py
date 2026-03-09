from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_speaker_core.rag import LexicalRAG
from llm_speaker_core.settings import SETTINGS

DEFAULT_QUERIES = [
    "Как поступить в ГУАП?",
    "Какие есть студенческие активности в ГУАП?",
    "Где находится ГУАП?",
    "Как связаться с приемной комиссией ГУАП?",
]


def run_eval(index_path: Path, out_path: Path, top_k: int = 4) -> dict:
    rag = LexicalRAG(index_path)
    rag.load()

    results = []
    total_hits = 0

    for query in DEFAULT_QUERIES:
        hits = rag.search(query, top_k=top_k)
        total_hits += len(hits)
        results.append(
            {
                "query": query,
                "hits": [
                    {
                        "id": hit["id"],
                        "source": hit["source"],
                        "score": hit["score"],
                    }
                    for hit in hits
                ],
            }
        )

    report = {
        "queries": len(DEFAULT_QUERIES),
        "total_hits": total_hits,
        "avg_hits": round(total_hits / max(len(DEFAULT_QUERIES), 1), 2),
        "results": results,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate lexical RAG retrieval")
    parser.add_argument("--index", type=Path, default=SETTINGS.index_path)
    parser.add_argument("--out", type=Path, default=Path("data/rag_eval_report.json"))
    parser.add_argument("--top-k", type=int, default=SETTINGS.rag_top_k)
    args = parser.parse_args()

    report = run_eval(args.index, args.out, top_k=args.top_k)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
