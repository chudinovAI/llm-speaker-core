import json
from pathlib import Path

from llm_speaker_core.rag import LexicalRAG
from llm_speaker_core.service import LLMService
from llm_speaker_core.settings import SETTINGS


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


class FakeHallucinatingLLM:
    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 180) -> str:
        return "Точная стоимость обучения составляет уникальную сумму, недоступную в контексте."


class FakeListOnlyLLM:
    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 180) -> str:
        if "Сожми текст" in system_prompt:
            return "2."
        return "2."


class FakeMenuLikeLLM:
    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 180) -> str:
        if "Сожми текст" in system_prompt:
            return "Стоимость зависит от программы и формы."
        return (
            "Информация о порядке оказания платных образовательных услуг "
            "Положение о платных образовательных услугах Формы договоров "
            "Стоимость обучения Комиссия по внебюджетной деятельности "
            "Оплата обучения и проживания в общежитиях Формы договоров."
        )


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
    service = LLMService(rag=rag, llm=FakeLLM(), speaker_mode="local")

    result = service.handle_query("Расскажи про ГУАП", "s1")

    assert len(result.display_text.split()) <= SETTINGS.max_display_words
    assert len(result.speaker_text.split()) <= SETTINGS.max_speaker_words
    assert result.used_rag is True
    assert result.fallback_used is False
    assert service.memory.get("s1")
    assert service.llm.calls == 1


def test_service_speaker_fallback(tmp_path: Path) -> None:
    index_path = tmp_path / "index.json"
    _make_index(index_path)

    rag = LexicalRAG(index_path)
    service = LLMService(rag=rag, llm=FakeLLM(speaker_should_fail=True), speaker_mode="llm")

    result = service.handle_query("Расскажи про ГУАП", "s2")

    assert result.display_text
    assert result.speaker_text == ""
    assert result.fallback_used is True


def test_service_uses_extractive_fallback_when_grounding_drops_answer(tmp_path: Path) -> None:
    index_path = tmp_path / "index.json"
    payload = {
        "docs": [
            {
                "id": "web:1",
                "source": "https://guap.ru/eif/price",
                "source_type": "web",
                "title": "Стоимость обучения",
                "text": (
                    "Стоимость обучения по программам бакалавриата и магистратуры "
                    "указана в действующем разделе с формами договора."
                ),
            }
        ],
        "doc_tfs": [
            {
                "стоимость": 1,
                "обучения": 1,
                "программам": 1,
                "бакалавриата": 1,
                "магистратуры": 1,
                "формами": 1,
                "договора": 1,
            }
        ],
        "doc_lens": [7],
        "doc_freqs": {
            "стоимость": 1,
            "обучения": 1,
            "программам": 1,
            "бакалавриата": 1,
            "магистратуры": 1,
            "формами": 1,
            "договора": 1,
        },
    }
    index_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    rag = LexicalRAG(index_path)
    service = LLMService(rag=rag, llm=FakeHallucinatingLLM())

    result = service.handle_query("Стоимость обучения в ГУАП?", "s3")
    assert "нет точных подтвержденных данных" not in result.display_text.lower()
    assert "стоимость обучения" in result.display_text.lower()


def test_tuition_speaker_policy_prefers_actionable_summary(tmp_path: Path) -> None:
    index_path = tmp_path / "index.json"
    payload = {
        "docs": [
            {
                "id": "web:1",
                "source": "https://guap.ru/eif/price",
                "source_type": "web",
                "title": "Стоимость обучения",
                "text": "Стоимость обучения зависит от программы и формы обучения.",
            }
        ],
        "doc_tfs": [{"стоимость": 1, "обучения": 1, "зависит": 1, "программы": 1, "формы": 1}],
        "doc_lens": [5],
        "doc_freqs": {"стоимость": 1, "обучения": 1, "зависит": 1, "программы": 1, "формы": 1},
    }
    index_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    rag = LexicalRAG(index_path)
    service = LLMService(rag=rag, llm=FakeLLM(), speaker_mode="llm")
    result = service.handle_query("Сколько стоит обучение в ГУАП?", "s4")

    assert "платных услуг гуап" in result.speaker_text.lower()


def test_low_info_display_is_replaced_by_extractive(tmp_path: Path) -> None:
    index_path = tmp_path / "index.json"
    payload = {
        "docs": [
            {
                "id": "web:1",
                "source": "https://guap.ru/eif/price",
                "source_type": "web",
                "title": "Стоимость обучения",
                "text": "Стоимость обучения зависит от программы и формы обучения.",
            }
        ],
        "doc_tfs": [{"стоимость": 1, "обучения": 1, "зависит": 1, "программы": 1, "формы": 1}],
        "doc_lens": [5],
        "doc_freqs": {"стоимость": 1, "обучения": 1, "зависит": 1, "программы": 1, "формы": 1},
    }
    index_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    rag = LexicalRAG(index_path)
    service = LLMService(rag=rag, llm=FakeListOnlyLLM(), speaker_mode="local")
    result = service.handle_query("Стоимость обучения в ГУАП?", "s5")

    assert result.display_text != "2."
    assert "стоимость обучения" in result.display_text.lower()


def test_tuition_menu_like_display_is_replaced_by_summary(tmp_path: Path) -> None:
    index_path = tmp_path / "index.json"
    payload = {
        "docs": [
            {
                "id": "web:1",
                "source": "https://guap.ru/eif/pol_usl",
                "source_type": "web",
                "title": "Платные образовательные услуги",
                "text": "Стоимость обучения и формы договоров в разделе платных услуг.",
            }
        ],
        "doc_tfs": [{"стоимость": 1, "обучения": 1, "формы": 1, "договоров": 1, "разделе": 1}],
        "doc_lens": [5],
        "doc_freqs": {"стоимость": 1, "обучения": 1, "формы": 1, "договоров": 1, "разделе": 1},
    }
    index_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    rag = LexicalRAG(index_path)
    service = LLMService(rag=rag, llm=FakeMenuLikeLLM(), speaker_mode="llm")
    result = service.handle_query("Сколько стоит обучение в ГУАП?", "s6")

    assert "актуальные цены" in result.display_text.lower()
