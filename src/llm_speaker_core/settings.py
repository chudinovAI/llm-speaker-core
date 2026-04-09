from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppSettings:
    cloudflare_raw_dir: Path = Path("data/raw/cloudflare/latest")
    cloudflare_records_path: Path = Path("data/raw/cloudflare/latest/records.jsonl")
    normalized_documents_path: Path = Path("data/normalized/documents.jsonl")
    normalized_chunks_path: Path = Path("data/normalized/chunks.jsonl")
    hybrid_manifest_path: Path = Path("data/index_manifest.json")
    hybrid_lexical_index_path: Path = Path("data/indexes/bm25/index.json")
    hybrid_dense_index_path: Path = Path("data/indexes/faiss/index.json")
    retrieval_backend: str = "hybrid"
    embedding_model: str = "ai-sage/Giga-Embeddings-instruct"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    rag_top_k: int = 5
    language: str = "ru"
    max_display_words: int = 120
    max_speaker_chars: int = 160
    max_memory_turns: int = 5
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3.5:2b"
    llm_timeout_s: float = 180.0
    fallback_text: str = (
        "Извините, не получилось сформировать ответ. Повторите вопрос, пожалуйста."
    )


SETTINGS = AppSettings()
