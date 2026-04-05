from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from llm_speaker_core.retrieval.schemas import ChunkRecord, RetrievalHit
from llm_speaker_core.utils.text import expand_query, tokenize


@dataclass
class LexicalIndex:
    chunks: list[ChunkRecord]
    doc_freqs: dict[str, int]
    doc_tfs: list[dict[str, int]]
    doc_lens: list[int]
    avg_doc_len: float

    @classmethod
    def build(cls, chunks: list[ChunkRecord]) -> "LexicalIndex":
        doc_freqs: Counter[str] = Counter()
        doc_tfs: list[dict[str, int]] = []
        doc_lens: list[int] = []
        for chunk in chunks:
            tokens = tokenize(chunk.text)
            tf = Counter(tokens)
            doc_tfs.append(dict(tf))
            doc_lens.append(len(tokens))
            doc_freqs.update(tf.keys())
        avg_doc_len = sum(doc_lens) / max(len(doc_lens), 1)
        return cls(
            chunks=chunks,
            doc_freqs=dict(doc_freqs),
            doc_tfs=doc_tfs,
            doc_lens=doc_lens,
            avg_doc_len=avg_doc_len,
        )

    def search(self, query: str, top_k: int = 8) -> list[RetrievalHit]:
        query_tokens = expand_query(tokenize(query))
        if not query_tokens:
            return []
        scores: list[tuple[float, int]] = []
        total_docs = len(self.chunks)
        k1 = 1.5
        b = 0.75
        for idx, tf in enumerate(self.doc_tfs):
            score = 0.0
            doc_len = self.doc_lens[idx] or 1
            text_tokens = set(tf.keys())
            for token in query_tokens:
                freq = tf.get(token, 0)
                if freq == 0:
                    continue
                df = self.doc_freqs.get(token, 0)
                idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
                denom = freq + k1 * (1 - b + b * doc_len / max(self.avg_doc_len, 1))
                score += idf * freq * (k1 + 1) / denom
            if score == 0.0:
                continue
            overlap = len(set(query_tokens) & text_tokens) / max(len(set(query_tokens)), 1)
            if self.chunks[idx].source_type == "web":
                score += 0.1
            score += overlap
            scores.append((score, idx))
        scores.sort(key=lambda item: item[0], reverse=True)
        hits: list[RetrievalHit] = []
        for score, idx in scores[:top_k]:
            chunk = self.chunks[idx]
            hits.append(
                RetrievalHit(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    source=chunk.source,
                    text=chunk.text,
                    score=round(score, 6),
                    retrieval_stage="lexical",
                    metadata=chunk.hit_metadata(),
                )
            )
        return hits

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "chunks": [asdict(chunk) for chunk in self.chunks],
            "doc_freqs": self.doc_freqs,
            "doc_tfs": self.doc_tfs,
            "doc_lens": self.doc_lens,
            "avg_doc_len": self.avg_doc_len,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "LexicalIndex":
        payload = json.loads(path.read_text(encoding="utf-8"))
        chunks = [ChunkRecord(**row) for row in payload["chunks"]]
        return cls(
            chunks=chunks,
            doc_freqs={k: int(v) for k, v in payload["doc_freqs"].items()},
            doc_tfs=[{k: int(v) for k, v in row.items()} for row in payload["doc_tfs"]],
            doc_lens=[int(v) for v in payload["doc_lens"]],
            avg_doc_len=float(payload["avg_doc_len"]),
        )
