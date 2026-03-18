from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlparse

from llm_speaker_core.retrieval.dense import DenseIndex
from llm_speaker_core.retrieval.lexical import LexicalIndex
from llm_speaker_core.retrieval.rerank import CrossEncoderReranker
from llm_speaker_core.retrieval.schemas import ChunkRecord, EvidencePack, IndexManifest, RetrievalHit
from llm_speaker_core.utils.text import INTENT_HINTS, expand_query, tokenize

PATH_TOKEN_ALIASES = {
    "bach": ("бакалавриат", "специалитет", "поступление", "направления"),
    "mag": ("магистратура", "магистр", "программы"),
    "dates": ("сроки", "даты", "прием"),
    "contacts": ("контакты", "телефон", "адрес", "приемная", "комиссия"),
    "rules": ("правила", "прием", "поступление"),
    "pay": ("оплата", "стоимость", "обучение"),
    "pay_edu": ("оплата", "платные", "услуги", "обучение"),
    "paid_edu": ("оплата", "платные", "услуги", "обучение"),
    "price": ("стоимость", "цена", "обучение"),
    "common": ("сведения", "контакты", "адрес", "режим"),
    "objects": ("общежитие", "корпуса", "объекты", "адрес"),
    "budget": ("бюджет", "бюджетные", "места"),
    "grants": ("стипендия", "поддержка"),
    "studlife": ("студенты", "активности", "объединения", "клубы"),
    "vrmp": ("воспитательная", "молодежная", "политика"),
    "tochka": ("точка", "кипения"),
    "faq": ("вопросы", "ответы"),
}
PREFERRED_SOURCE_HINTS = {
    "admission": ("priem.guap.ru", "/abitur"),
    "contacts": ("priem.guap.ru/contacts", "/contacts", "/contact", "/sveden/common", "/sveden/managers"),
    "location": ("/sveden/common", "/sveden/objects", "/address", "/contacts"),
    "student_life": ("/studlife", "/vrmp"),
    "official_info": ("/sveden", "/document"),
    "tuition": ("/eif/", "/sveden/pay_edu", "/sveden/paid_edu", "/price", "/pay", "/pol_usl"),
}


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
        return expand_query(tokenize(query))

    def _structural_bonus(self, query: str, hit: RetrievalHit) -> float:
        query_tokens = set(self._query_expansions(query))
        if not query_tokens:
            return 0.0
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
        if not structural_tokens:
            return 0.0
        overlap = len(query_tokens & structural_tokens) / len(query_tokens)
        return overlap * 0.45

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
        if any(intent == "admission" for intent in intents):
            if "priem.guap.ru" in source:
                bonus += 0.26
            if any(k in source for k in ("/abitur", "/bach", "/mag", "/dates", "/contacts")):
                bonus += 0.16
        if any(intent == "tuition" for intent in intents):
            if any(k in source for k in ("/eif/", "/paid_edu", "/price", "/pay", "/pol_usl")):
                bonus += 0.22
        if any(intent == "contacts" for intent in intents):
            if any(k in source for k in ("/contacts", "/contact", "/sveden/common", "priem.guap.ru/contacts")):
                bonus += 0.24
        if any(intent == "location" for intent in intents):
            if any(k in source for k in ("/sveden/common", "/sveden/objects", "/address")):
                bonus += 0.22
        if any(intent == "student_life" for intent in intents):
            if any(k in source for k in ("/studlife", "/vrmp")):
                bonus += 0.24
        if any(intent == "official_info" for intent in intents):
            if any(k in source for k in ("/sveden", "/document", "/common")):
                bonus += 0.28
        return bonus

    def _source_penalty(self, hit: RetrievalHit, intents: list[str]) -> float:
        source = hit.source.lower()
        flags = {str(flag) for flag in hit.metadata.get("quality_flags", [])}
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
        if any(intent in {"admission", "contacts", "official_info"} for intent in intents) and "/faq" in source:
            penalty += 0.12
        for intent in intents:
            preferred = PREFERRED_SOURCE_HINTS.get(intent)
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

    def _merge_hits(self, query: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        intents = self.detect_intents(query)
        hits = self._normalize_stage_scores(hits)
        by_chunk: dict[str, RetrievalHit] = {}
        for hit in hits:
            adjusted = (
                hit.score
                - self._freshness_penalty(hit, intents)
                - self._source_penalty(hit, intents)
                + self._source_bonus(hit, intents)
                + self._structural_bonus(query, hit)
            )
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

    def _diversify_hits(self, hits: list[RetrievalHit], top_k: int) -> list[RetrievalHit]:
        diversified: list[RetrievalHit] = []
        seen_sources: dict[str, int] = {}
        seen_chunks: set[str] = set()
        for hit in hits:
            if hit.chunk_id in seen_chunks:
                continue
            if seen_sources.get(hit.source, 0) >= 1:
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
        lexical_hits = self.lexical.search(query, top_k=max(top_k * 3, 12))
        expanded_query = " ".join(self._query_expansions(query)) or query
        dense_hits = (
            self.dense.search(expanded_query, top_k=max(top_k * 3, 12))
            if self.dense and self.dense.available
            else []
        )
        merged = self._merge_hits(query, [*lexical_hits, *dense_hits])
        if self.reranker is not None and self.reranker.is_available:
            merged = self.reranker.rerank(expanded_query, merged, top_k=max(top_k * 4, 12))
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
