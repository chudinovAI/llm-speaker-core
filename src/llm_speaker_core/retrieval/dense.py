from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

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

try:
    import torch
except Exception:  # noqa: BLE001
    torch = None  # type: ignore[assignment,misc]

QUERY_PROMPT = "Instruct: Дан вопрос, необходимо найти абзац текста с ответом\nQuery: "
MAX_SEQ_LENGTH = 1024
_MODEL_CACHE: dict[str, object] = {}
_DIM_CACHE: dict[str, int] = {}
_MODEL_CACHE_LOCK = Lock()


class DenseEncoder:
    def __init__(self, model_name: str, dimension: int = 2048) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self.batch_size = 8
        self._model = None
        self.device = "cpu"

        if SentenceTransformer is not None:
            try:
                with _MODEL_CACHE_LOCK:
                    cached_model = _MODEL_CACHE.get(model_name)
                    cached_dim = _DIM_CACHE.get(model_name)
                    if cached_model is None:
                        cached_model, cached_dim, device = self._load_model_with_fallback(model_name)
                        _MODEL_CACHE[model_name] = cached_model
                        _DIM_CACHE[model_name] = cached_dim
                        logger.info("DenseEncoder loaded %s (dim=%d, device=%s)", model_name, cached_dim, device)
                    else:
                        device = "gpu_or_auto"
                self._model = cached_model
                if cached_dim is not None:
                    self.dimension = cached_dim
                self.device = device
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Failed to load dense model %s, dense retrieval disabled",
                    model_name, exc_info=True,
                )
                self._model = None

        self.available = self._model is not None
        if self.device == "cpu":
            self.batch_size = 4

    @staticmethod
    def _instantiate_model(model_name: str, *, device: str | None = None) -> object:
        kwargs: dict[str, object] = {
            "trust_remote_code": True,
        }
        if torch is not None and torch.cuda.is_available():
            kwargs["model_kwargs"] = {"torch_dtype": torch.bfloat16}
        if device is not None:
            kwargs["device"] = device
        return SentenceTransformer(model_name, **kwargs)

    def _load_model_with_fallback(self, model_name: str) -> tuple[object, int, str]:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        try:
            model = self._instantiate_model(model_name)
            model.max_seq_length = MAX_SEQ_LENGTH
            test_vec = model.encode(["test"])
            return model, int(test_vec.shape[1]), "auto"
        except Exception as exc:  # noqa: BLE001
            if "out of memory" not in str(exc).lower():
                raise
            logger.warning("Dense model %s hit OOM on default device, retrying on CPU", model_name, exc_info=True)
            if torch is not None and hasattr(torch, "cuda"):
                try:
                    torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001
                    pass
            model = self._instantiate_model(model_name, device="cpu")
            model.max_seq_length = MAX_SEQ_LENGTH
            test_vec = model.encode(["test"])
            return model, int(test_vec.shape[1]), "cpu"

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
    _faiss_index: object | None = None
    _encoder: DenseEncoder | None = None

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
        index = cls(chunks=chunks, vectors=vectors, model_name=model_name, available=True)
        index._encoder = encoder
        index._faiss_index = index._build_faiss_index()
        return index

    def _get_encoder(self) -> DenseEncoder:
        if self._encoder is None:
            self._encoder = DenseEncoder(self.model_name, dimension=int(self.vectors.shape[1]))
        return self._encoder

    def _build_faiss_index(self) -> object | None:
        if faiss is None or self.vectors.size == 0 or not self.available:
            return None
        index = faiss.IndexFlatIP(self.vectors.shape[1])
        index.add(self.vectors)
        return index

    def _get_faiss_index(self) -> object | None:
        if self._faiss_index is None:
            self._faiss_index = self._build_faiss_index()
        return self._faiss_index

    def search(self, query: str, top_k: int = 8) -> list[RetrievalHit]:
        if not self.chunks or not self.available:
            return []
        encoder = self._get_encoder()
        if not encoder.available:
            return []
        q = encoder.encode_queries([query])
        if q.size == 0:
            return []
        index = self._get_faiss_index()
        if index is not None:
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
        index = self._get_faiss_index()
        if index is not None:
            faiss.write_index(index, str(path.with_suffix(".faiss")))

    @classmethod
    def load(cls, path: Path) -> "DenseIndex":
        payload = json.loads(path.read_text(encoding="utf-8"))
        chunks = [ChunkRecord(**row) for row in payload["chunks"]]
        vectors = np.load(path.with_suffix(".vectors.npy"))
        index = cls(
            chunks=chunks,
            vectors=vectors.astype(np.float32),
            model_name=str(payload["model_name"]),
            available=bool(payload.get("available", True)),
        )
        faiss_path = path.with_suffix(".faiss")
        if faiss is not None and faiss_path.exists():
            try:
                index._faiss_index = faiss.read_index(str(faiss_path))
            except Exception:  # noqa: BLE001
                logger.warning("Failed to load FAISS index from %s", faiss_path, exc_info=True)
        return index
