import json
from pathlib import Path

from llm_speaker_core.rag import LexicalRAG
from llm_speaker_core.service import LLMService


class FakeLLM:
    def __init__(self, speaker_should_fail: bool = False) -> None:
        self.speaker_should_fail = speaker_should_fail
        self.calls = 0

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 180) -> str:
        self.calls += 1
        if "Сожми текст" in system_prompt:
            if self.speaker_should_fail:
                raise RuntimeError("speaker down")
            return "Кратко: это ключевой ответ для озвучки в динамик"
        return "**Ответ** " + "слово " * 200


def _make_index(path: Path) -> None:
    payload = {
        "docs": [
            {
                "id": "web:1",
                "source": "https://guap.ru",
                "source_type": "web",
                "title": "ГУАП",
                "text": "ГУАП университет в Санкт-Петербурге",
            }
        ],
        "doc_tfs": [{"гуап": 1, "университет": 1, "санкт": 1, "петербурге": 1}],
        "doc_lens": [4],
        "doc_freqs": {"гуап": 1, "университет": 1, "санкт": 1, "петербурге": 1},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_service_limits_and_memory(tmp_path: Path) -> None:
    index_path = tmp_path / "index.json"
    _make_index(index_path)

    rag = LexicalRAG(index_path)
    service = LLMService(rag=rag, llm=FakeLLM())

    result = service.handle_query("Расскажи про ГУАП", "s1")

    assert len(result.display_text.split()) <= 120
    assert len(result.speaker_text.split()) <= 20
    assert result.used_rag is True
    assert result.fallback_used is False
    assert service.memory.get("s1")


def test_service_speaker_fallback(tmp_path: Path) -> None:
    index_path = tmp_path / "index.json"
    _make_index(index_path)

    rag = LexicalRAG(index_path)
    service = LLMService(rag=rag, llm=FakeLLM(speaker_should_fail=True))

    result = service.handle_query("Расскажи про ГУАП", "s2")

    assert result.display_text
    assert result.speaker_text == ""
    assert result.fallback_used is True
