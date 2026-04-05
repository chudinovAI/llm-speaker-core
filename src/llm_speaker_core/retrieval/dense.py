from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from llm_speaker_core.retrieval.schemas import ChunkRecord, RetrievalHit

logger = logging.getLogger(__name__)

try:
    import faiss  # type: ignore[import-not-found]
except Exception:  # noqa: BLE001
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # noqa: BLE001
    SentenceTransformer = None  # type: ignore[assignment,misc]

QUERY_PROMPT = "Instruct: Дан вопрос, необходимо найти абзац текста с ответом\nQuery: "


class DenseEncoder:
    def __init__(self, model_name: str, dimension: int = 2048) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self.batch_size = 4
        self._model = None

        if SentenceTransformer is not None:
            try:
                os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
                self._model = SentenceTransformer(
                    model_name,
                    trust_remote_code=True,
                )
                self._model.max_seq_length = 4096
                test_vec = self._model.encode(["test"])
                self.dimension = int(test_vec.shape[1])
                logger.info("DenseEncoder loaded %s (dim=%d)", model_name, self.dimension)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Failed to load dense model %s, dense retrieval disabled",
                    model_name, exc_info=True,
                )
                self._model = None

        self.available = self._model is not None

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        if self._model is None:
            raise RuntimeError("Dense model not loaded")
        total = len(texts)
        print(f"[dense] encoding {total} texts...", flush=True)
        result = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return np.asarray(result, dtype=np.float32)

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        if not queries:
            return np.zeros((0, self.dimension), dtype=np.float32)
        if self._model is None:
            raise RuntimeError("Dense model not loaded")
        result = self._model.encode(
            queries,
            prompt=QUERY_PROMPT,
            batch_size=self.batch_size,
            normalize_embeddings=True,
        )
        return np.asarray(result, dtype=np.float32)


@dataclass
class DenseIndex:
    chunks: list[ChunkRecord]
    vectors: np.ndarray
    model_name: str
    available: bool = True

    @classmethod
    def build(cls, chunks: list[ChunkRecord], model_name: str) -> "DenseIndex":
        encoder = DenseEncoder(model_name)
        if not encoder.available:
            logger.warning("Dense encoder not available, building empty index")
            return cls(
                chunks=chunks,
                vectors=np.zeros((len(chunks), encoder.dimension), dtype=np.float32),
                model_name=model_name,
                available=False,
            )
        vectors = encoder.encode_documents([chunk.text for chunk in chunks])
        return cls(chunks=chunks, vectors=vectors, model_name=model_name, available=True)

    def search(self, query: str, top_k: int = 8) -> list[RetrievalHit]:
        if not self.chunks or not self.available:
            return []
        encoder = DenseEncoder(self.model_name, dimension=int(self.vectors.shape[1]))
        if not encoder.available:
            return []
        q = encoder.encode_queries([query])
        if q.size == 0:
            return []
        if faiss is not None and self.vectors.size > 0:
            index = faiss.IndexFlatIP(self.vectors.shape[1])
            index.add(self.vectors)
            scores, indices = index.search(q, top_k)
            idxs = indices[0].tolist()
            vals = scores[0].tolist()
        else:
            sims = self.vectors @ q[0]
            order = np.argsort(-sims)[:top_k]
            idxs = order.tolist()
            vals = [float(sims[idx]) for idx in order]
        hits: list[RetrievalHit] = []
        for idx, score in zip(idxs, vals, strict=False):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            hits.append(
                RetrievalHit(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    source=chunk.source,
                    text=chunk.text,
                    score=round(float(score), 6),
                    retrieval_stage="dense",
                    metadata=chunk.hit_metadata(),
                )
            )
        return hits

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path.with_suffix(".vectors.npy"), self.vectors)
        payload = {
            "model_name": self.model_name,
            "available": self.available,
            "chunks": [chunk.__dict__ for chunk in self.chunks],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        if faiss is not None and self.vectors.size > 0 and self.available:
            index = faiss.IndexFlatIP(self.vectors.shape[1])
            index.add(self.vectors)
            faiss.write_index(index, str(path.with_suffix(".faiss")))

    @classmethod
    def load(cls, path: Path) -> "DenseIndex":
        payload = json.loads(path.read_text(encoding="utf-8"))
        chunks = [ChunkRecord(**row) for row in payload["chunks"]]
        vectors = np.load(path.with_suffix(".vectors.npy"))
        return cls(
            chunks=chunks,
            vectors=vectors.astype(np.float32),
            model_name=str(payload["model_name"]),
            available=bool(payload.get("available", True)),
        )
