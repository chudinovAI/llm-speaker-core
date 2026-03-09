from __future__ import annotations

import logging

import uvicorn
from fastapi import FastAPI, HTTPException

from llm_speaker_core.api_models import HealthResponse, QueryMeta, QueryRequest, QueryResponse
from llm_speaker_core.indexer import run_nightly_reindex
from llm_speaker_core.llm import OllamaClient
from llm_speaker_core.rag import LexicalRAG
from llm_speaker_core.service import LLMService
from llm_speaker_core.settings import SETTINGS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
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


app = FastAPI(title="LLM Speaker Core", version="0.1.0")
service: LLMService | None = None


@app.on_event("startup")
def on_startup() -> None:
    global service
    service = build_service()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    index_loaded = SETTINGS.index_path.exists()
    return HealthResponse(status="ok", model=SETTINGS.ollama_model, index_loaded=index_loaded)


@app.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest) -> QueryResponse:
    if service is None:
        raise HTTPException(status_code=503, detail="Service is not ready")

    try:
        result = service.handle_query(
            text=payload.text,
            session_id=payload.session_id,
            history=payload.history,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Query handling failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return QueryResponse(
        display_text=result.display_text,
        speaker_text=result.speaker_text,
        meta=QueryMeta(
            latency_ms=result.latency_ms,
            used_rag=result.used_rag,
            fallback_used=result.fallback_used,
            limits_applied=result.limits_applied,
            rag_hits=result.rag_hits,
            rag_sources=result.rag_sources,
        ),
    )


@app.get("/metrics")
def metrics() -> dict:
    if service is None:
        raise HTTPException(status_code=503, detail="Service is not ready")
    return service.metrics.snapshot()


def main() -> None:
    uvicorn.run("llm_speaker_core.api:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
