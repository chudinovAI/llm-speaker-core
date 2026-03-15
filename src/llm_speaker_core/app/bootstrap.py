from __future__ import annotations

import logging

from llm_speaker_core.indexer import run_nightly_reindex
from llm_speaker_core.llm_runtime.client import OllamaClient
from llm_speaker_core.rag import LexicalRAG
from llm_speaker_core.service import LLMService
from llm_speaker_core.settings import SETTINGS

logger = logging.getLogger(__name__)


def build_service() -> LLMService:
    if not SETTINGS.index_path.exists() and SETTINGS.raw_corpus_path.exists():
        logger.info("RAG index is absent. Running nightly reindex once...")
        run_nightly_reindex(
            raw_path=SETTINGS.raw_corpus_path,
            cleaned_path=SETTINGS.cleaned_corpus_path,
            report_path=SETTINGS.quality_report_path,
            index_path=SETTINGS.index_path,
        )

    rag = LexicalRAG(SETTINGS.index_path)
    llm = OllamaClient(
        base_url=SETTINGS.ollama_base_url,
        model=SETTINGS.ollama_model,
        timeout_s=SETTINGS.llm_timeout_s,
    )
    return LLMService(rag=rag, llm=llm)
