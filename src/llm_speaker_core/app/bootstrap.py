from __future__ import annotations

import logging

from llm_speaker_core.llm_runtime.client import OllamaClient
from llm_speaker_core.retrieval.service import HybridRetrievalService
from llm_speaker_core.service import LLMService, RetrievalProtocol
from llm_speaker_core.settings import SETTINGS

logger = logging.getLogger(__name__)


def build_service() -> LLMService:
    rag: RetrievalProtocol | None = None
    backend = SETTINGS.retrieval_backend.lower()

    if backend in {"auto", "hybrid"} and SETTINGS.hybrid_manifest_path.exists():
        logger.info("Loading new hybrid retrieval backend from %s", SETTINGS.hybrid_manifest_path)
        rag = HybridRetrievalService.load(SETTINGS.hybrid_manifest_path)

    if rag is None:
        raise RuntimeError(
            "Hybrid retrieval manifest is missing. Build the new index at data/index_manifest.json."
        )
    llm = OllamaClient(
        base_url=SETTINGS.ollama_base_url,
        model=SETTINGS.ollama_model,
        timeout_s=SETTINGS.llm_timeout_s,
    )
    return LLMService(rag=rag, llm=llm)
