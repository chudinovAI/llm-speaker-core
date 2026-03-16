from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from llm_speaker_core.retrieval.dense import DenseIndex
from llm_speaker_core.retrieval.lexical import LexicalIndex
from llm_speaker_core.retrieval.rerank import CrossEncoderReranker
from llm_speaker_core.retrieval.schemas import ChunkRecord, EvidencePack, IndexManifest, RetrievalHit
from llm_speaker_core.utils.text import INTENT_HINTS, QUERY_EXPANSIONS, tokenize


class HybridRetrievalService:
    version = "hybrid-rag-v2"

    def __init__(
        self,
        lexical: LexicalIndex,
        dense: DenseIndex | None,
        reranker: CrossEncoderReranker | None,
        manifest: IndexManifest,
    ) -> None:
        self.lexical = lexical
        self.dense = dense
        self.reranker = reranker
        self.manifest = manifest

    @classmethod
    def load(cls, manifest_path: Path) -> "HybridRetrievalService":
        manifest = IndexManifest.from_path(manifest_path)
        lexical = LexicalIndex.load(manifest_path.parent / manifest.lexical_path)
        dense = None
        if manifest.dense_path:
            dense_path = manifest_path.parent / manifest.dense_path
            if dense_path.exists():
                dense = DenseIndex.load(dense_path)
        reranker = CrossEncoderReranker(manifest.reranker_model) if manifest.reranker_model else None
        return cls(lexical=lexical, dense=dense, reranker=reranker, manifest=manifest)

    @staticmethod
    def detect_intents(text: str) -> list[str]:
        low = text.lower()
        intents: list[str] = []
        for intent, hints in INTENT_HINTS.items():
            if any(h in low for h in hints):
                intents.append(intent)
        return intents or ["general"]

    def _query_expansions(self, query: str) -> list[str]:
        tokens = tokenize(query)
        expanded = list(tokens)
        for token in tokens:
            for stem, extra in QUERY_EXPANSIONS.items():
                if token.startswith(stem):
                    expanded.extend(extra)
        return expanded

    def _freshness_penalty(self, hit: RetrievalHit, intents: list[str]) -> float:
        archived = bool(hit.metadata.get("is_archived", False))
        if not archived:
            return 0.0
        if any(intent in {"tuition", "admission", "contacts"} for intent in intents):
            return 0.3
        return 0.08

    def _source_bonus(self, hit: RetrievalHit, intents: list[str]) -> float:
        source = hit.source.lower()
        bonus = 0.0
        if "guap.ru" in source:
            bonus += 0.08
        if any(intent == "admission" for intent in intents) and any(k in source for k in ("/abitur", "/priem")):
            bonus += 0.15
        if any(intent == "tuition" for intent in intents) and any(k in source for k in ("/eif/", "/paid_edu")):
            bonus += 0.15
        if any(intent == "contacts" for intent in intents) and "/sveden/common" in source:
            bonus += 0.12
        return bonus

    def _merge_hits(self, query: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        intents = self.detect_intents(query)
        by_chunk: dict[str, RetrievalHit] = {}
        for hit in hits:
            adjusted = hit.score - self._freshness_penalty(hit, intents) + self._source_bonus(hit, intents)
            merged = RetrievalHit(
                chunk_id=hit.chunk_id,
                doc_id=hit.doc_id,
                source=hit.source,
                text=hit.text,
                score=round(adjusted, 6),
                retrieval_stage=hit.retrieval_stage,
                metadata=dict(hit.metadata),
            )
            current = by_chunk.get(hit.chunk_id)
            if current is None or merged.score > current.score:
                by_chunk[hit.chunk_id] = merged
        merged_hits = list(by_chunk.values())
        merged_hits.sort(key=lambda item: item.score, reverse=True)
        return merged_hits

    def search_hits(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        lexical_hits = self.lexical.search(query, top_k=max(top_k * 3, 12))
        dense_hits = self.dense.search(query, top_k=max(top_k * 3, 12)) if self.dense else []
        merged = self._merge_hits(query, [*lexical_hits, *dense_hits])
        if self.reranker is not None:
            merged = self.reranker.rerank(query, merged, top_k=max(top_k * 2, 8))
        return merged[:top_k]

    def build_evidence_pack(self, query: str, top_k: int = 5, token_budget: int = 900) -> EvidencePack:
        hits = self.search_hits(query, top_k=top_k)
        chosen: list[RetrievalHit] = []
        budget = 0
        for hit in hits:
            est = max(len(hit.text.split()), 1)
            if chosen and budget + est > token_budget:
                break
            chosen.append(hit)
            budget += est
        grounding_score = round(sum(max(hit.score, 0.0) for hit in chosen) / max(len(chosen), 1), 4) if chosen else 0.0
        return EvidencePack(query=query, hits=chosen, token_budget=token_budget, grounding_score=grounding_score)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        return [
            {
                "id": hit.chunk_id,
                "doc_id": hit.doc_id,
                "source": hit.source,
                "text": hit.text,
                "score": hit.score,
                **dict(hit.metadata),
            }
            for hit in self.search_hits(query, top_k=top_k)
        ]

    @classmethod
    def build_from_chunks(
        cls,
        chunks: list[ChunkRecord],
        lexical_path: Path,
        dense_path: Path,
        manifest_path: Path,
        embedding_model: str,
        reranker_model: str,
    ) -> "HybridRetrievalService":
        lexical = LexicalIndex.build(chunks)
        lexical.save(lexical_path)
        dense = DenseIndex.build(chunks, embedding_model)
        dense.save(dense_path)
        manifest = IndexManifest(
            version=cls.version,
            corpus_checksum=str(abs(hash("".join(chunk.content_hash for chunk in chunks))) % (10**16)),
            lexical_path=str(lexical_path.relative_to(manifest_path.parent)),
            dense_path=str(dense_path.relative_to(manifest_path.parent)),
            reranker_model=reranker_model,
            embedding_model=embedding_model,
            built_at=__import__("datetime").datetime.utcnow().isoformat() + "Z",
            doc_count=len({chunk.doc_id for chunk in chunks}),
            chunk_count=len(chunks),
            metadata={"storage": "faiss+jsonl", "dense_optional": False},
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(asdict(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
        return cls(
            lexical=lexical,
            dense=dense,
            reranker=CrossEncoderReranker(reranker_model),
            manifest=manifest,
        )
