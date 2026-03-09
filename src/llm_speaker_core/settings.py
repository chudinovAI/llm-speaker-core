from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppSettings:
    raw_corpus_path: Path = Path("suai_facts.txt")
    cleaned_corpus_path: Path = Path("data/cleaned_corpus.jsonl")
    quality_report_path: Path = Path("data/cleaning_report.json")
    index_path: Path = Path("data/rag_index.json")
    rag_top_k: int = 3
    language: str = "ru"
    max_display_words: int = 70
    max_speaker_words: int = 14
    speaker_mode: str = "llm"  # local | llm
    max_memory_turns: int = 5
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3.5:2b"
    llm_timeout_s: float = 30.0
    fallback_text: str = "Извините, не получилось сформировать ответ. Повторите вопрос, пожалуйста."


SETTINGS = AppSettings()
