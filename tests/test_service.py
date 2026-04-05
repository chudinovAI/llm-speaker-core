import json
from typing import cast

from llm_speaker_core.service import LLMService
from llm_speaker_core.settings import SETTINGS


class FakeLLM:
    def __init__(self, speaker_should_fail: bool = False) -> None:
        self.speaker_should_fail = speaker_should_fail
        self.calls = 0

    def generate(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 180
    ) -> str:
        self.calls += 1
        if "Сожми текст" in system_prompt:
            if self.speaker_should_fail:
                raise RuntimeError("speaker down")
            return "Кратко: это ключевой ответ для озвучки в динамик"
        return "**Ответ** " + "слово " * 200


class FakeHallucinatingLLM:
    def generate(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 180
    ) -> str:
        return "Точная стоимость обучения составляет уникальную сумму, недоступную в контексте."


class FakeListOnlyLLM:
    def generate(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 180
    ) -> str:
        if "Сожми текст" in system_prompt:
            return "2."
        return "2."


class FakeMenuLikeLLM:
    def generate(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 180
    ) -> str:
        if "Сожми текст" in system_prompt:
            return "Стоимость зависит от программы и формы."
        return (
            "Информация о порядке оказания платных образовательных услуг "
            "Положение о платных образовательных услугах Формы договоров "
            "Стоимость обучения Комиссия по внебюджетной деятельности "
            "Оплата обучения и проживания в общежитиях Формы договоров."
        )


class FakeAdmissionVagueLLM:
    def generate(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 180
    ) -> str:
        if "Сожми текст" in system_prompt:
            return "Уточняйте правила в разделе для абитуриентов."
        return (
            "Однако, исходя из предоставленного списка возможностей сервиса, "
            "можно выделить основные категории направлений подготовки."
        )


class StubRetrieval:
    version = "test-retrieval"

    def __init__(self, docs: list[dict]) -> None:
        self.docs = docs

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        return self.docs[:top_k]

    def detect_intents(self, text: str) -> list[str]:
        low = text.lower()
        if "стоим" in low or "обучен" in low:
            return ["tuition"]
        if "направлен" in low or "поступ" in low:
            return ["admission"]
        return ["general"]


def _doc(
    source: str,
    text: str,
    *,
    doc_id: str = "web:1",
    title: str = "ГУАП",
    metadata: dict | None = None,
) -> dict:
    payload = {
        "id": doc_id,
        "doc_id": doc_id,
        "source": source,
        "source_type": "web",
        "title": title,
        "text": text,
    }
    if metadata:
        payload.update(metadata)
    return payload


def test_service_limits_and_memory() -> None:
    rag = StubRetrieval([_doc("https://guap.ru", "ГУАП университет в Санкт-Петербурге")])
    service = LLMService(rag=rag, llm=FakeLLM(), speaker_mode="local")

    result = service.handle_query("Расскажи про ГУАП", "s1")

    assert len(result.display_text.split()) <= SETTINGS.max_display_words
    assert len(result.speaker_text.split()) <= SETTINGS.max_speaker_words
    assert result.used_rag is True
    assert result.fallback_used is False
    assert service.memory.get("s1")
    assert cast(FakeLLM, service.llm).calls == 1


def test_service_speaker_fallback() -> None:
    rag = StubRetrieval([_doc("https://guap.ru", "ГУАП университет в Санкт-Петербурге")])
    service = LLMService(
        rag=rag, llm=FakeLLM(speaker_should_fail=True), speaker_mode="llm"
    )

    result = service.handle_query("Расскажи про ГУАП", "s2")

    assert result.display_text
    assert result.speaker_text == ""
    assert result.fallback_used is True


def test_service_uses_extractive_fallback_when_grounding_drops_answer() -> None:
    rag = StubRetrieval(
        [
            _doc(
                "https://guap.ru/eif/price",
                (
                    "Стоимость обучения по программам бакалавриата и магистратуры "
                    "указана в действующем разделе с формами договора."
                ),
            )
        ]
    )
    service = LLMService(rag=rag, llm=FakeHallucinatingLLM())

    result = service.handle_query("Стоимость обучения в ГУАП?", "s3")
    assert "нет точных подтвержденных данных" not in result.display_text.lower()
    assert "стоимость обучения" in result.display_text.lower()


def test_tuition_speaker_policy_prefers_actionable_summary() -> None:
    rag = StubRetrieval(
        [
            _doc(
                "https://guap.ru/eif/price",
                "Стоимость обучения зависит от программы и формы обучения.",
            )
        ]
    )
    service = LLMService(rag=rag, llm=FakeLLM(), speaker_mode="llm")
    result = service.handle_query("Сколько стоит обучение в ГУАП?", "s4")

    assert "платных услуг гуап" in result.speaker_text.lower()


def test_low_info_display_is_replaced_by_extractive() -> None:
    rag = StubRetrieval(
        [
            _doc(
                "https://guap.ru/eif/price",
                "Стоимость обучения зависит от программы и формы обучения.",
            )
        ]
    )
    service = LLMService(rag=rag, llm=FakeListOnlyLLM(), speaker_mode="local")
    result = service.handle_query("Стоимость обучения в ГУАП?", "s5")

    assert result.display_text != "2."
    assert "стоимость обучения" in result.display_text.lower()


def test_tuition_menu_like_display_is_replaced_by_summary() -> None:
    rag = StubRetrieval(
        [
            _doc(
                "https://guap.ru/eif/pol_usl",
                "Стоимость обучения и формы договоров в разделе платных услуг.",
            )
        ]
    )
    service = LLMService(rag=rag, llm=FakeMenuLikeLLM(), speaker_mode="llm")
    result = service.handle_query("Сколько стоит обучение в ГУАП?", "s6")

    assert "актуальные цены" in result.display_text.lower()


def test_admission_low_evidence_uses_rule_summary() -> None:
    rag = StubRetrieval(
        [
            _doc(
                "https://guap.ru/abitur",
                "Раздел для абитуриентов с правилами и сроками приема.",
            )
        ]
    )
    service = LLMService(rag=rag, llm=FakeAdmissionVagueLLM(), speaker_mode="llm")
    result = service.handle_query("Какие направления есть в ГУАПе?", "s7")

    assert "проверьте актуальные правила и сроки" in result.display_text.lower()


def test_general_query_with_irrelevant_science_sources_becomes_uncertain() -> None:
    rag = StubRetrieval(
        [
            _doc(
                "https://guap.ru/science/sciact-projects",
                (
                    "Разработанные методы планирования трафика позволяют составлять "
                    "расписание для узлов сети."
                ),
                title="Научные проекты",
                metadata={"source_facets": [], "page_type": "detail"},
            ),
            _doc(
                "https://guap.ru/pubs/25195",
                "В публикации рассматривается планирование трафика и расписание отправки данных.",
                doc_id="web:2",
                title="Публикации",
                metadata={"source_facets": [], "page_type": "document"},
            ),
        ]
    )
    service = LLMService(rag=rag, llm=FakeLLM(), speaker_mode="local")

    result = service.handle_query("Расскажи про расписание библиотеки.", "s8")

    assert result.grounding_score < 0.3


def test_general_query_with_supported_facet_remains_grounded() -> None:
    rag = StubRetrieval(
        [
            _doc(
                "https://guap.ru/vrmp/tochka",
                "Точка кипения и ВРМП координируют молодежные проекты и инициативы ГУАП.",
                title="ВРМП",
                metadata={"source_facets": ["vrmp"], "page_type": "profile"},
            )
        ]
    )
    service = LLMService(rag=rag, llm=FakeLLM(), speaker_mode="local")

    result = service.handle_query("Что такое ВРМП в ГУАП?", "s9")

    assert result.answer_mode == "grounded"
    assert "нет точных подтвержденных данных" not in result.display_text.lower()


def test_contacts_operational_query_without_contact_sources_becomes_uncertain() -> None:
    rag = StubRetrieval(
        [
            _doc(
                "https://guap.ru/sgo",
                "Отдел взаимодействует с сотрудниками и публикует внутренние новости.",
                title="СГО",
                metadata={"source_facets": [], "page_type": "detail"},
            ),
            _doc(
                "https://guap.ru/it/struct/oib",
                "Подразделение обеспечивает безопасность и внутренние коммуникации.",
                doc_id="web:2",
                title="Информационная безопасность",
                metadata={"source_facets": [], "page_type": "detail"},
            ),
        ]
    )
    service = LLMService(rag=rag, llm=FakeLLM(), speaker_mode="local")

    result = service.handle_query("А какое расписание отдела кадров?", "s10")

    assert result.answer_mode == "uncertain"
    assert "нет точных подтвержденных данных" in result.display_text.lower()

