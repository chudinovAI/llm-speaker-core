from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

from llm_speaker_core.retrieval.schemas import RetrievalHit
from llm_speaker_core.utils.text import tokenize

try:
    from FlagEmbedding import FlagReranker
except Exception:  # noqa: BLE001
    FlagReranker = None  # type: ignore[assignment,misc]

try:
    from huggingface_hub import snapshot_download
except Exception:  # noqa: BLE001
    snapshot_download = None  # type: ignore[assignment,misc]

try:
    import torch
except Exception:  # noqa: BLE001
    torch = None  # type: ignore[assignment,misc]


@dataclass
class CrossEncoderReranker:
    model_name: str

    def __post_init__(self) -> None:
        self._model = None
        if FlagReranker is not None:
            try:
                local_path = self._resolve_local_model_path(self.model_name)
                use_fp16 = bool(torch is not None and torch.cuda.is_available())
                self._model = FlagReranker(
                    local_path,
                    use_fp16=use_fp16,
                    trust_remote_code=False,
                )
            except Exception:  # noqa: BLE001
                self._model = None
        self.is_available = self._model is not None

    @staticmethod
    def _resolve_local_model_path(model_name: str) -> str:
        if snapshot_download is None:
            return model_name
        try:
            return str(snapshot_download(repo_id=model_name, local_files_only=True))
        except Exception:  # noqa: BLE001
            return model_name

    def rerank(self, query: str, hits: list[RetrievalHit], top_k: int) -> list[RetrievalHit]:
        if not hits:
            return []
        if self._model is not None:
            pairs = [[query, hit.text] for hit in hits]
            raw_scores = self._model.compute_score(pairs)
            if isinstance(raw_scores, list):
                scores = [float(score) for score in raw_scores]
            else:
                scores = [float(cast(float, raw_scores))]
        else:
            q_tokens = set(tokenize(query))
            scores = []
            for hit in hits:
                h_tokens = set(tokenize(hit.text))
                overlap = len(q_tokens & h_tokens) / max(len(q_tokens), 1)
                scores.append(overlap + math.tanh(hit.score))
        rescored: list[RetrievalHit] = []
        for hit, score in zip(hits, scores, strict=False):
            combined_score = float(score)
            if self._model is not None:
                combined_score += 0.35 * max(hit.score, 0.0)
            rescored.append(
                RetrievalHit(
                    chunk_id=hit.chunk_id,
                    doc_id=hit.doc_id,
                    source=hit.source,
                    text=hit.text,
                    score=round(combined_score, 6),
                    retrieval_stage="rerank",
                    metadata=dict(hit.metadata),
                )
            )
        rescored.sort(key=lambda item: item.score, reverse=True)
        return rescored[:top_k]
