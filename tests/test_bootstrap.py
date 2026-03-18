from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from llm_speaker_core.app import bootstrap


class DummyRetrieval:
    version = "test"

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return []

    def detect_intents(self, text: str) -> list[str]:
        return ["general"]


class DummyLLM:
    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 180) -> str:
        return "ok"


def test_build_service_fast_mode_disables_dense_and_reranker(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_load(manifest_path, *, enable_dense: bool, enable_reranker: bool):  # type: ignore[no-untyped-def]
        captured["enable_dense"] = enable_dense
        captured["enable_reranker"] = enable_reranker
        return DummyRetrieval()

    monkeypatch.setattr(
        bootstrap,
        "SETTINGS",
        SimpleNamespace(
            retrieval_backend="hybrid",
            hybrid_manifest_path=Path("data/index_manifest.json"),
            ollama_base_url="http://127.0.0.1:11434",
            ollama_model="qwen3.5:2b",
            llm_timeout_s=30.0,
        ),
    )
    monkeypatch.setattr(bootstrap.HybridRetrievalService, "load", fake_load)
    monkeypatch.setattr(bootstrap, "OllamaClient", lambda **kwargs: DummyLLM())
    monkeypatch.setattr(Path, "exists", lambda self: True)

    service = bootstrap.build_service(retrieval_runtime_mode="fast")

    assert captured == {"enable_dense": False, "enable_reranker": False}
    assert service.rag.version == "test"


def test_build_service_full_mode_enables_dense_and_reranker(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_load(manifest_path, *, enable_dense: bool, enable_reranker: bool):  # type: ignore[no-untyped-def]
        captured["enable_dense"] = enable_dense
        captured["enable_reranker"] = enable_reranker
        return DummyRetrieval()

    monkeypatch.setattr(
        bootstrap,
        "SETTINGS",
        SimpleNamespace(
            retrieval_backend="hybrid",
            hybrid_manifest_path=Path("data/index_manifest.json"),
            ollama_base_url="http://127.0.0.1:11434",
            ollama_model="qwen3.5:2b",
            llm_timeout_s=30.0,
        ),
    )
    monkeypatch.setattr(bootstrap.HybridRetrievalService, "load", fake_load)
    monkeypatch.setattr(bootstrap, "OllamaClient", lambda **kwargs: DummyLLM())
    monkeypatch.setattr(Path, "exists", lambda self: True)

    service = bootstrap.build_service(retrieval_runtime_mode="full")

    assert captured == {"enable_dense": True, "enable_reranker": True}
    assert service.rag.version == "test"
