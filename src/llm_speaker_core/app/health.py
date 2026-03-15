from __future__ import annotations

from llm_speaker_core.api_models import HealthResponse
from llm_speaker_core.settings import SETTINGS


def build_health_response() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model=SETTINGS.ollama_model,
        index_loaded=SETTINGS.index_path.exists(),
    )

