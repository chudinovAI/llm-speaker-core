from __future__ import annotations

import logging
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException

from llm_speaker_core.api_models import (
    QueryMeta,
    QueryRequest,
    QueryResponse,
)
from llm_speaker_core.app.bootstrap import build_service
from llm_speaker_core.app.health import build_health_response

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="LLM Speaker Core", version="0.1.0")
service: Any | None = None


@app.on_event("startup")
def on_startup() -> None:
    global service
    service = build_service()


@app.get("/health")
def health():
    return build_health_response()


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
            intent=result.intent,
            evidence_coverage=result.evidence_coverage,
            answer_mode=result.answer_mode,
        ),
    )


@app.get("/metrics")
def metrics() -> dict[str, int]:
    if service is None:
        raise HTTPException(status_code=503, detail="Service is not ready")
    return service.metrics.snapshot()


def main() -> None:
    uvicorn.run("llm_speaker_core.app.api:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()

