from fastapi.testclient import TestClient

import llm_speaker_core.api as api_module
from llm_speaker_core.service import GenerationResult


class StubService:
    class StubMetrics:
        @staticmethod
        def snapshot() -> dict:
            return {"requests": 1, "fallbacks": 0, "empty_speaker": 0, "rag_hits": 1}

    def __init__(self) -> None:
        self.metrics = self.StubMetrics()

    def handle_query(self, text: str, session_id: str, history=None) -> GenerationResult:
        return GenerationResult(
            display_text="**ГУАП**: это краткий ответ для дисплея.",
            speaker_text="ГУАП: краткий ответ.",
            used_rag=True,
            fallback_used=False,
            limits_applied=False,
            rag_hits=2,
            rag_sources=["https://guap.ru/sveden/common"],
            latency_ms=123,
        )


def test_query_contract() -> None:
    api_module.service = StubService()
    client = TestClient(api_module.app)

    response = client.post(
        "/query",
        json={
            "text": "Что такое ГУАП?",
            "session_id": "session-1",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert "display_text" in body
    assert "speaker_text" in body
    assert "meta" in body
    assert body["meta"]["used_rag"] is True


def test_metrics_endpoint() -> None:
    api_module.service = StubService()
    client = TestClient(api_module.app)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.json()["requests"] == 1
