from __future__ import annotations

import numpy as np

from llm_speaker_core.retrieval.rerank import CrossEncoderReranker
from llm_speaker_core.retrieval.schemas import RetrievalHit


class FakeArrayReranker:
    def compute_score(self, pairs: list[list[str]]) -> np.ndarray:
        assert len(pairs) == 2
        return np.asarray([0.25, 0.75], dtype=np.float32)


def test_rerank_accepts_numpy_array_scores() -> None:
    reranker = CrossEncoderReranker(model_name="unused")
    reranker._model = FakeArrayReranker()
    reranker.is_available = True

    hits = [
        RetrievalHit(
            chunk_id="doc1:0",
            doc_id="doc1",
            source="https://guap.ru/a",
            text="Первый текст",
            score=0.2,
            retrieval_stage="lexical",
            metadata={},
        ),
        RetrievalHit(
            chunk_id="doc2:0",
            doc_id="doc2",
            source="https://guap.ru/b",
            text="Второй текст",
            score=0.3,
            retrieval_stage="dense",
            metadata={},
        ),
    ]

    rescored = reranker.rerank("тестовый запрос", hits, top_k=2)

    assert len(rescored) == 2
    assert rescored[0].chunk_id == "doc2:0"
