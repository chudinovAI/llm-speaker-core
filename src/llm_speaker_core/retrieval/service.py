from __future__ import annotations

import json
import re
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse

from llm_speaker_core.retrieval.dense import DenseIndex
from llm_speaker_core.retrieval.lexical import LexicalIndex
from llm_speaker_core.retrieval.rerank import CrossEncoderReranker
from llm_speaker_core.retrieval.schemas import ChunkRecord, EvidencePack, IndexManifest, RetrievalHit
from llm_speaker_core.taxonomy import FACET_RULES, INTENT_PROFILES, PATH_TOKEN_ALIASES
from llm_speaker_core.utils.text import detect_facets, detect_intents, expand_query, tokenize


@dataclass(frozen=True)
class ScoreBreakdown:
    semantic_score: float
    policy_score: float
    freshness_penalty: float
    source_penalty: float
    source_bonus: float
    facet_bonus: float
    structural_bonus: float


class HybridRetrievalService:
    version = "hybrid-rag-v3"

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
    def load(
        cls,
        manifest_path: Path,
        *,
        enable_dense: bool = True,
        enable_reranker: bool = True,
    ) -> "HybridRetrievalService":
        manifest = IndexManifest.from_path(manifest_path)
        lexical = LexicalIndex.load(manifest_path.parent / manifest.lexical_path)
        dense = None
        if enable_dense and manifest.dense_path:
            dense_path = manifest_path.parent / manifest.dense_path
            if dense_path.exists():
                dense = DenseIndex.load(dense_path)
        reranker = (
            CrossEncoderReranker(manifest.reranker_model)
            if enable_reranker and manifest.reranker_model
            else None
        )
        return cls(lexical=lexical, dense=dense, reranker=reranker, manifest=manifest)

    @staticmethod
    def detect_intents(text: str) -> list[str]:
        return detect_intents(text)

    def _query_expansions(self, query: str) -> list[str]:
        return expand_query(tokenize(query))

    def _query_facets(self, query: str) -> list[str]:
        return detect_facets(query)

    def _structural_bonus(self, query: str, hit: RetrievalHit) -> float:
        query_tokens = set(self._query_expansions(query))
        if not query_tokens:
            bonus = 0.0
        else:
            structural_text = " ".join(
                [
                    urlparse(hit.source).path.replace("/", " "),
                    str(hit.metadata.get("title", "")),
                    str(hit.metadata.get("section", "")),
                    " ".join(str(part) for part in hit.metadata.get("section_path", [])),
                ]
            )
            structural_tokens = set(tokenize(structural_text))
            for token in list(structural_tokens):
                structural_tokens.update(PATH_TOKEN_ALIASES.get(token, ()))
            bonus = 0.0
            if structural_tokens:
                overlap = len(query_tokens & structural_tokens) / len(query_tokens)
                bonus += overlap * 0.45

        query_numbers = set(re.findall(r"\b\d+\b", query))
        if query_numbers:
            source = hit.source.lower()
            title = str(hit.metadata.get("title", "")).lower()
            for number in query_numbers:
                if f"/{number}" in source:
                    bonus += 0.28
                if f"№{number}" in title or f" {number}" in title:
                    bonus += 0.18
        return bonus

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
        for intent in intents:
            profile = INTENT_PROFILES.get(intent)
            if not profile or not profile.preferred_source_hints:
                continue
            if any(hint in source for hint in profile.preferred_source_hints):
                if intent == "admission":
                    bonus += 0.26
                elif intent == "tuition":
                    bonus += 0.22
                elif intent in {"contacts", "organization"}:
                    bonus += 0.24
                elif intent == "location":
                    bonus += 0.22
                elif intent == "student_life":
                    bonus += 0.24
                elif intent == "official_info":
                    bonus += 0.28
                else:
                    bonus += 0.18
        return bonus

    def _facet_bonus(self, hit: RetrievalHit, facets: list[str]) -> float:
        if not facets:
            return 0.0
        source = hit.source.lower()
        page_type = str(hit.metadata.get("page_type", ""))
        source_facets = {str(value) for value in hit.metadata.get("source_facets", [])}
        bonus = 0.0
        penalty = 0.0
        for facet in facets:
            rules = FACET_RULES.get(facet)
            if not rules:
                continue
            for hint, value in rules["path_bonus"].items():
                if hint in source:
                    bonus += value
            for hint, value in rules["path_penalty"].items():
                if hint in source:
                    penalty += value
            bonus += rules["page_type_bonus"].get(page_type, 0.0)
            penalty += rules["page_type_penalty"].get(page_type, 0.0)
            for source_facet, value in rules["source_facet_bonus"].items():
                if source_facet in source_facets:
                    bonus += value
            for source_facet, value in rules["source_facet_penalty"].items():
                if source_facet in source_facets:
                    penalty += value
        return bonus - penalty

    def _source_penalty(self, hit: RetrievalHit, intents: list[str]) -> float:
        source = hit.source.lower()
        flags = {str(flag) for flag in hit.metadata.get("quality_flags", [])}
        page_type = str(hit.metadata.get("page_type", ""))
        penalty = 0.0
        if hit.metadata.get("is_low_signal", False):
            penalty += 0.45
        if "is_boilerplate_heavy" in flags:
            penalty += 0.18
        if "is_low_priority_source" in flags:
            penalty += 0.16
        if any(marker in source for marker in ("/faq", "/sitemap")):
            penalty += 0.34
        if "/greeting" in source:
            penalty += 0.16
        if page_type == "faq":
            penalty += 0.12
        if any(intent in {"admission", "contacts", "official_info"} for intent in intents) and "/faq" in source:
            penalty += 0.12
        for intent in intents:
            profile = INTENT_PROFILES.get(intent)
            preferred = profile.preferred_source_hints if profile else ()
            if preferred and not any(hint in source for hint in preferred):
                penalty += 0.18
        quality_score = float(hit.metadata.get("quality_score", 1.0))
        penalty += max(0.0, 0.65 - quality_score) * 0.4
        return penalty

    def _normalize_stage_scores(self, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        stage_max: dict[str, float] = {}
        for hit in hits:
            stage_max[hit.retrieval_stage] = max(stage_max.get(hit.retrieval_stage, 0.0), hit.score)
        normalized: list[RetrievalHit] = []
        for hit in hits:
            denom = stage_max.get(hit.retrieval_stage, 1.0) or 1.0
            normalized.append(
                RetrievalHit(
                    chunk_id=hit.chunk_id,
                    doc_id=hit.doc_id,
                    source=hit.source,
                    text=hit.text,
                    score=round(hit.score / denom, 6),
                    retrieval_stage=hit.retrieval_stage,
                    metadata=dict(hit.metadata),
                )
            )
        return normalized

    def _score_hit(self, query: str, hit: RetrievalHit, intents: list[str], facets: list[str]) -> ScoreBreakdown:
        freshness_penalty = self._freshness_penalty(hit, intents)
        source_penalty = self._source_penalty(hit, intents)
        source_bonus = self._source_bonus(hit, intents)
        facet_bonus = self._facet_bonus(hit, facets)
        structural_bonus = self._structural_bonus(query, hit)
        policy_score = round(
            -freshness_penalty - source_penalty + source_bonus + facet_bonus + structural_bonus,
            6,
        )
        return ScoreBreakdown(
            semantic_score=round(hit.score, 6),
            policy_score=policy_score,
            freshness_penalty=round(freshness_penalty, 6),
            source_penalty=round(source_penalty, 6),
            source_bonus=round(source_bonus, 6),
            facet_bonus=round(facet_bonus, 6),
            structural_bonus=round(structural_bonus, 6),
        )

    def _merge_hits(self, query: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        intents = self.detect_intents(query)
        facets = self._query_facets(query)
        hits = self._normalize_stage_scores(hits)
        by_chunk: dict[str, RetrievalHit] = {}
        for hit in hits:
            breakdown = self._score_hit(query, hit, intents, facets)
            adjusted = round(breakdown.semantic_score + breakdown.policy_score, 6)
            metadata = dict(hit.metadata)
            metadata["semantic_score"] = breakdown.semantic_score
            metadata["policy_score"] = breakdown.policy_score
            metadata["score_components"] = {
                "freshness_penalty": breakdown.freshness_penalty,
                "source_penalty": breakdown.source_penalty,
                "source_bonus": breakdown.source_bonus,
                "facet_bonus": breakdown.facet_bonus,
                "structural_bonus": breakdown.structural_bonus,
            }
            merged = RetrievalHit(
                chunk_id=hit.chunk_id,
                doc_id=hit.doc_id,
                source=hit.source,
                text=hit.text,
                score=adjusted,
                retrieval_stage=hit.retrieval_stage,
                metadata=metadata,
            )
            current = by_chunk.get(hit.chunk_id)
            if current is None or merged.score > current.score:
                by_chunk[hit.chunk_id] = merged
        merged_hits = list(by_chunk.values())
        merged_hits.sort(key=lambda item: item.score, reverse=True)
        return merged_hits

    def _diversify_hits(self, hits: list[RetrievalHit], top_k: int) -> list[RetrievalHit]:
        diversified: list[RetrievalHit] = []
        seen_sources: dict[str, int] = {}
        seen_chunks: set[str] = set()
        for hit in hits:
            if hit.chunk_id in seen_chunks:
                continue
            if seen_sources.get(hit.source, 0) >= 2:
                continue
            diversified.append(hit)
            seen_sources[hit.source] = seen_sources.get(hit.source, 0) + 1
            seen_chunks.add(hit.chunk_id)
            if len(diversified) >= top_k:
                return diversified
        for hit in hits:
            if hit.chunk_id in seen_chunks:
                continue
            diversified.append(hit)
            seen_chunks.add(hit.chunk_id)
            if len(diversified) >= top_k:
                break
        return diversified

    def search_hits(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        candidate_top_k = max(top_k * 6, 24)
        lexical_hits = self.lexical.search(query, top_k=candidate_top_k)
        expanded_query = " ".join(self._query_expansions(query)) or query
        dense_hits = (
            self.dense.search(expanded_query, top_k=candidate_top_k)
            if self.dense and self.dense.available
            else []
        )
        merged = self._merge_hits(query, [*lexical_hits, *dense_hits])
        if self.reranker is not None and self.reranker.is_available:
            merged = self.reranker.rerank(expanded_query, merged, top_k=max(top_k * 6, 24))
        merged = self._diversify_hits(merged, top_k=max(top_k * 2, 8))
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
        reranker = CrossEncoderReranker(reranker_model)
        corpus_checksum = hashlib.sha256(
            "".join(chunk.content_hash for chunk in chunks).encode("utf-8")
        ).hexdigest()[:16]
        manifest = IndexManifest(
            version=cls.version,
            corpus_checksum=corpus_checksum,
            lexical_path=str(lexical_path.relative_to(manifest_path.parent)),
            dense_path=str(dense_path.relative_to(manifest_path.parent)),
            reranker_model=reranker_model,
            embedding_model=embedding_model,
            built_at=__import__("datetime").datetime.utcnow().isoformat() + "Z",
            doc_count=len({chunk.doc_id for chunk in chunks}),
            chunk_count=len(chunks),
            metadata={
                "storage": "faiss+jsonl",
                "dense_available": dense.available,
                "reranker_available": reranker.is_available,
            },
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(asdict(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
        return cls(
            lexical=lexical,
            dense=dense,
            reranker=reranker,
            manifest=manifest,
        )
