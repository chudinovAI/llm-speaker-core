from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from llm_speaker_core.ingest.extractors.cloudflare import canonicalize_url
from llm_speaker_core.retrieval.service import HybridRetrievalService
from llm_speaker_core.retrieval.schemas import EvidencePack


def run_eval(manifest_path: Path, gold_path: Path, out_path: Path, top_k: int = 5) -> dict:
    retriever = HybridRetrievalService.load(manifest_path)
    rows = [json.loads(line) for line in gold_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    results = []
    total_recall = 0.0
    total_top3 = 0.0
    for row in rows:
        evidence = retriever.build_evidence_pack(row["query"], top_k=top_k)
        expected = {canonicalize_url(src) for src in row.get("expected_sources", []) if canonicalize_url(src)}
        found = [canonicalize_url(hit.source) for hit in evidence.hits if canonicalize_url(hit.source)]
        recall = 1.0 if expected and any(src in found for src in expected) else 0.0
        top3 = 1.0 if expected and any(src in found[:3] for src in expected) else 0.0
        total_recall += recall
        total_top3 += top3
        results.append(
            {
                "query": row["query"],
                "expected_sources": list(expected),
                "found_sources": found,
                "grounding_score": evidence.grounding_score,
                "recall_at_k": recall,
                "top3_hit": top3,
            }
        )
    payload = {
        "queries": len(rows),
        "recall_at_k": round(total_recall / max(len(rows), 1), 4),
        "top3_hit_rate": round(total_top3 / max(len(rows), 1), 4),
        "results": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate hybrid RAG on a gold query set")
    parser.add_argument("--manifest", type=Path, default=Path("data/index_manifest.json"))
    parser.add_argument("--gold", type=Path, default=Path("data/eval/gold_queries.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/eval/hybrid_eval_report.json"))
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(run_eval(args.manifest, args.gold, args.out, top_k=args.top_k), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
