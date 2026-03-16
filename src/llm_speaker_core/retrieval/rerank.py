from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

from llm_speaker_core.retrieval.schemas import RetrievalHit
from llm_speaker_core.utils.text import tokenize

try:
    from sentence_transformers import CrossEncoder
except Exception:  # noqa: BLE001
    CrossEncoder = None  # type: ignore[assignment,misc]


@dataclass
class CrossEncoderReranker:
    model_name: str

    def __post_init__(self) -> None:
        self._model = None
        if CrossEncoder is not None:
            try:
                self._model = CrossEncoder(self.model_name, local_files_only=True)
            except Exception:  # noqa: BLE001
                self._model = None

    def rerank(self, query: str, hits: list[RetrievalHit], top_k: int) -> list[RetrievalHit]:
        if not hits:
            return []
        if self._model is not None:
            pairs = [[query, hit.text] for hit in hits]
            raw_scores = self._model.predict(pairs)
            if hasattr(raw_scores, "tolist"):
                scores = cast(list[float], raw_scores.tolist())
            else:
                scores = [float(score) for score in raw_scores]
        else:
            q_tokens = set(tokenize(query))
            scores = []
            for hit in hits:
                h_tokens = set(tokenize(hit.text))
                overlap = len(q_tokens & h_tokens) / max(len(q_tokens), 1)
                scores.append(overlap + math.tanh(hit.score))
        rescored: list[RetrievalHit] = []
        for hit, score in zip(hits, scores, strict=False):
            rescored.append(
                RetrievalHit(
                    chunk_id=hit.chunk_id,
                    doc_id=hit.doc_id,
                    source=hit.source,
                    text=hit.text,
                    score=round(float(score), 6),
                    retrieval_stage="rerank",
                    metadata=dict(hit.metadata),
                )
            )
        rescored.sort(key=lambda item: item.score, reverse=True)
        return rescored[:top_k]
