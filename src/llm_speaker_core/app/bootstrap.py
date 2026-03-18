from __future__ import annotations

import logging

from llm_speaker_core.llm_runtime.client import OllamaClient
from llm_speaker_core.retrieval.service import HybridRetrievalService
from llm_speaker_core.service import LLMService, RetrievalProtocol
from llm_speaker_core.settings import SETTINGS

logger = logging.getLogger(__name__)


def build_service(retrieval_runtime_mode: str = "full") -> LLMService:
    rag: RetrievalProtocol | None = None
    backend = SETTINGS.retrieval_backend.lower()
    runtime_mode = retrieval_runtime_mode.lower()
    enable_dense = runtime_mode == "full"
    enable_reranker = runtime_mode == "full"

    if backend in {"auto", "hybrid"} and SETTINGS.hybrid_manifest_path.exists():
        logger.info(
            "Loading hybrid retrieval backend from %s (mode=%s, dense=%s, reranker=%s)",
            SETTINGS.hybrid_manifest_path,
            runtime_mode,
            enable_dense,
            enable_reranker,
        )
        rag = HybridRetrievalService.load(
            SETTINGS.hybrid_manifest_path,
            enable_dense=enable_dense,
            enable_reranker=enable_reranker,
        )

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
