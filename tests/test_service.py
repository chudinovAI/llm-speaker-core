import json
from typing import cast

from llm_speaker_core.service import LLMService
from llm_speaker_core.settings import SETTINGS


class FakeLLM:
    def __init__(self, with_links: bool = False) -> None:
        self.with_links = with_links
        self.calls = 0

    def generate(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 180
    ) -> str:
        self.calls += 1
        display = "**Ответ** " + "слово " * 200
        if self.with_links:
            display += " [Контакты](https://guap.ru/contacts) https://guap.ru/rasp "
        return (
            f"<display>{display}</display>"
            "<speaker>Кратко: это ключевой ответ для озвучки в динамик</speaker>"
        )


class FakeHallucinatingLLM:
    def generate(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 180
    ) -> str:
        return (
            "<display>Точная стоимость обучения составляет уникальную сумму, "
            "недоступную в контексте.</display><speaker>Уникальная сумма.</speaker>"
        )


class FakeListOnlyLLM:
    def generate(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 180
    ) -> str:
        return "<display>2.</display><speaker>2.</speaker>"


class FakeMenuLikeLLM:
    def generate(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 180
    ) -> str:
        return (
            "<display>Информация о порядке оказания платных образовательных услуг "
            "Положение о платных образовательных услугах Формы договоров "
            "Стоимость обучения Комиссия по внебюджетной деятельности "
            "Оплата обучения и проживания в общежитиях Формы договоров.</display>"
            "<speaker>Стоимость зависит от программы и формы.</speaker>"
        )


class FakeAdmissionVagueLLM:
    def generate(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 180
    ) -> str:
        return (
            "<display>Однако, исходя из предоставленного списка возможностей сервиса, "
            "можно выделить основные категории направлений подготовки.</display>"
            "<speaker>Уточняйте правила в разделе для абитуриентов.</speaker>"
        )


class FakeMissingSpeakerTagLLM:
    def generate(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 180
    ) -> str:
        return "<display>ГУАП университет в Санкт-Петербурге.</display>"


class FakeLongBrokenTailLLM:
    def generate(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 180
    ) -> str:
        display = (
            "<display>Первое короткое предложение. "
            "Второе предложение достаточно длинное и содержит много слов для проверки "
            "ограничения длины ответа и того как сервис обрабатывает обрезание текста "
            "после лимита слов и не оставляет ответ в виде оборванного хвоста и</display>"
        )
        speaker = (
            "<speaker>Второе предложение достаточно длинное и содержит много слов "
            "для проверки ограничения длины ответа и</speaker>"
        )
        return display + speaker


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
    assert len(result.speaker_text) <= SETTINGS.max_speaker_chars + 1
    assert result.used_rag is True
    assert result.fallback_used is False
    assert service.memory.get("s1")
    assert cast(FakeLLM, service.llm).calls == 1


def test_service_falls_back_to_local_speaker_when_tag_missing() -> None:
    rag = StubRetrieval([_doc("https://guap.ru", "ГУАП университет в Санкт-Петербурге")])
    service = LLMService(rag=rag, llm=FakeMissingSpeakerTagLLM(), speaker_mode="llm")

    result = service.handle_query("Расскажи про ГУАП", "s2")

    assert result.display_text
    assert result.speaker_text


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
    assert cast(FakeLLM, service.llm).calls == 1


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


def test_service_removes_links_from_display_and_speech() -> None:
    rag = StubRetrieval([_doc("https://guap.ru", "ГУАП университет в Санкт-Петербурге")])
    service = LLMService(rag=rag, llm=FakeLLM(with_links=True), speaker_mode="llm")

    result = service.handle_query("Расскажи про ГУАП", "s-links")

    assert "https://" not in result.display_text
    assert "[" not in result.display_text
    assert "https://" not in result.speaker_text


def test_service_limit_words_does_not_leave_broken_tail() -> None:
    rag = StubRetrieval([_doc("https://guap.ru", "ГУАП университет в Санкт-Петербурге")])
    service = LLMService(rag=rag, llm=FakeLongBrokenTailLLM(), speaker_mode="llm")

    result = service.handle_query("Расскажи про ГУАП", "s-tail")

    assert not result.display_text.endswith(" и.")
    assert not result.speaker_text.endswith(" и.")


def test_service_speaker_is_limited_by_natural_boundary_not_word_cap() -> None:
    rag = StubRetrieval([_doc("https://guap.ru", "ГУАП университет в Санкт-Петербурге")])
    service = LLMService(rag=rag, llm=FakeLongBrokenTailLLM(), speaker_mode="llm")

    result = service.handle_query("Расскажи про ГУАП", "s-speaker-limit")

    assert len(result.speaker_text) <= SETTINGS.max_speaker_chars + 1
    assert not result.speaker_text.endswith(" и.")


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


def test_contacts_operational_query_irrelevant_sources_low_evidence_coverage() -> None:
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

    assert result.used_rag is True
    assert result.evidence_coverage < 0.35
