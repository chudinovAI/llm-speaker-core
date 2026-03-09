from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    text: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    history: list[Any] | None = None


class QueryMeta(BaseModel):
    latency_ms: int
    used_rag: bool
    fallback_used: bool
    limits_applied: bool
    rag_hits: int
    rag_sources: list[str] = []


class QueryResponse(BaseModel):
    display_text: str
    speaker_text: str
    meta: QueryMeta


class HealthResponse(BaseModel):
    status: str
    model: str
    index_loaded: bool
