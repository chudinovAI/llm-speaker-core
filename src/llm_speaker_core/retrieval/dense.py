from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from llm_speaker_core.retrieval.schemas import ChunkRecord, RetrievalHit

try:
    import faiss  # type: ignore[import-not-found]
except Exception:  # noqa: BLE001
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # noqa: BLE001
    SentenceTransformer = None  # type: ignore[assignment,misc]


class DenseEncoder:
    def __init__(self, model_name: str, dimension: int = 768) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self._model: Any | None = None
        if SentenceTransformer is not None:
            try:
                self._model = SentenceTransformer(model_name, local_files_only=True)
                test_vec = self._model.encode(["test"], normalize_embeddings=True)
                self.dimension = int(test_vec.shape[1])
            except Exception:  # noqa: BLE001
                self._model = None

    def _fallback_encode_text(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8", errors="ignore")).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dimension
            vec[bucket] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        if self._model is not None:
            return cast(np.ndarray, self._model.encode(texts, normalize_embeddings=True)).astype(np.float32)
        return np.vstack([self._fallback_encode_text(text) for text in texts]).astype(np.float32)


@dataclass
class DenseIndex:
    chunks: list[ChunkRecord]
    vectors: np.ndarray
    model_name: str

    @classmethod
    def build(cls, chunks: list[ChunkRecord], model_name: str) -> "DenseIndex":
        encoder = DenseEncoder(model_name)
        vectors = encoder.encode([chunk.text for chunk in chunks])
        return cls(chunks=chunks, vectors=vectors, model_name=model_name)

    def search(self, query: str, top_k: int = 8) -> list[RetrievalHit]:
        if not self.chunks:
            return []
        encoder = DenseEncoder(self.model_name, dimension=int(self.vectors.shape[1]))
        q = encoder.encode([query])
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
                    metadata={"section_path": list(chunk.section_path)},
                )
            )
        return hits

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path.with_suffix(".vectors.npy"), self.vectors)
        payload = {
            "model_name": self.model_name,
            "chunks": [chunk.__dict__ for chunk in self.chunks],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        if faiss is not None and self.vectors.size > 0:
            index = faiss.IndexFlatIP(self.vectors.shape[1])
            index.add(self.vectors)
            faiss.write_index(index, str(path.with_suffix(".faiss")))

    @classmethod
    def load(cls, path: Path) -> "DenseIndex":
        payload = json.loads(path.read_text(encoding="utf-8"))
        chunks = [ChunkRecord(**row) for row in payload["chunks"]]
        vectors = np.load(path.with_suffix(".vectors.npy"))
        return cls(chunks=chunks, vectors=vectors.astype(np.float32), model_name=str(payload["model_name"]))
