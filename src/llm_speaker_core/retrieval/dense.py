from __future__ import annotations

import hashlib
import json
import os
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
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
except Exception:  # noqa: BLE001
    torch = None  # type: ignore[assignment,misc]
    F = None  # type: ignore[assignment,misc]
    AutoModel = None  # type: ignore[assignment,misc]
    AutoTokenizer = None  # type: ignore[assignment,misc]

try:
    from huggingface_hub import snapshot_download
except Exception:  # noqa: BLE001
    snapshot_download = None  # type: ignore[assignment,misc]

class DenseEncoder:
    def __init__(self, model_name: str, dimension: int = 768) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self.batch_size = 4
        self.max_length = 512
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        if AutoTokenizer is not None and AutoModel is not None and torch is not None:
            try:
                os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
                os.environ.setdefault("OMP_NUM_THREADS", "1")
                torch.set_num_threads(1)
                local_path = self._resolve_local_model_path(model_name)
                self._tokenizer = AutoTokenizer.from_pretrained(
                    local_path,
                    local_files_only=True,
                    trust_remote_code=True,
                )
                self._model = AutoModel.from_pretrained(
                    local_path,
                    local_files_only=True,
                    trust_remote_code=True,
                )
                self._model.eval()
                test_vec = self.encode(["test"])
                self.dimension = int(test_vec.shape[1])
            except Exception:  # noqa: BLE001
                self._tokenizer = None
                self._model = None
        self.available = self._model is not None

    @staticmethod
    def _resolve_local_model_path(model_name: str) -> str:
        if snapshot_download is None:
            return model_name
        try:
            return str(snapshot_download(repo_id=model_name, local_files_only=True))
        except Exception:  # noqa: BLE001
            return model_name

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
        if self._model is not None and self._tokenizer is not None and torch is not None and F is not None:
            batches: list[np.ndarray] = []
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                encoded = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                with torch.no_grad():
                    outputs = self._model(**encoded)
                hidden = outputs.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                normalized = F.normalize(pooled, p=2, dim=1)
                batches.append(normalized.cpu().numpy().astype(np.float32))
            return np.vstack(batches).astype(np.float32)
        return np.vstack([self._fallback_encode_text(text) for text in texts]).astype(np.float32)


@dataclass
class DenseIndex:
    chunks: list[ChunkRecord]
    vectors: np.ndarray
    model_name: str
    available: bool = True

    @staticmethod
    def _chunk_metadata(chunk: ChunkRecord) -> dict[str, object]:
        return {
            "section_path": list(chunk.section_path),
            "title": chunk.title,
            "section": chunk.section,
            "canonical_url": chunk.canonical_url,
            "source_url": chunk.source_url,
            "published_at": chunk.published_at,
            "quality_score": chunk.quality_score,
            "quality_flags": list(chunk.quality_flags),
            "is_archived": chunk.is_archived,
            "is_low_signal": chunk.is_low_signal,
        }

    @classmethod
    def build(cls, chunks: list[ChunkRecord], model_name: str) -> "DenseIndex":
        encoder = DenseEncoder(model_name)
        vectors = encoder.encode([chunk.text for chunk in chunks])
        return cls(chunks=chunks, vectors=vectors, model_name=model_name, available=encoder.available)

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
                    metadata=self._chunk_metadata(chunk),
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
        if faiss is not None and self.vectors.size > 0:
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
