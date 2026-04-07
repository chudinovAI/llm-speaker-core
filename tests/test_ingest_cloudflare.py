from __future__ import annotations

from llm_speaker_core.ingest.extractors.cloudflare import canonicalize_url
from llm_speaker_core.ingest.normalize import (
    build_chunk_corpus,
    dedupe_documents,
    document_from_cloudflare_record,
)


def test_canonicalize_url_normalizes_mobile_and_content_mirrors() -> None:
    assert canonicalize_url("https://guap.ru/m/eif/pay?x=1") == "https://guap.ru/eif/pay"
    assert canonicalize_url("https://guap.ru/c/vrmp/tochka#top") == "https://guap.ru/vrmp/tochka"
    assert canonicalize_url("https://guap.ru/m/eif/m/eif/price") == "https://guap.ru/eif/price"


def test_document_from_cloudflare_record_extracts_text_and_links() -> None:
    record = {
        "url": "https://guap.ru/abitur?utm_source=test",
        "title": "Поступление в ГУАП",
        "markdown": "# Поступление\n\nГУАП принимает документы на программы бакалавриата.\n\n[Правила приема](https://guap.ru/priem/rules.pdf)",
    }

    doc = document_from_cloudflare_record(record, crawl_job_id="job-1")

    assert doc is not None
    assert doc.canonical_url == "https://guap.ru/abitur"
    assert doc.section == "abitur"
    assert doc.crawl_job_id == "job-1"
    assert "бакалавриата" in doc.text
    assert doc.metadata["document_links"] == ["https://guap.ru/priem/rules.pdf"]


def test_document_from_cloudflare_record_keeps_subdomain_document_links() -> None:
    record = {
        "url": "https://guap.ru/library",
        "title": "Библиотека ГУАП",
        "markdown": (
            "# Библиотека\n\n"
            "Официальные документы библиотеки.\n\n"
            "[Правила пользования](https://lib.guap.ru/docs/rules.pdf)"
        ),
    }

    doc = document_from_cloudflare_record(record, crawl_job_id="job-subdomain")

    assert doc is not None
    assert doc.metadata["document_links"] == ["https://lib.guap.ru/docs/rules.pdf"]


def test_document_from_cloudflare_record_trims_header_and_footer_noise() -> None:
    record = {
        "url": "https://guap.ru/faq",
        "title": "Вопросы и ответы",
        "markdown": (
            "[ГУАП](https://guap.ru)\n\n1. [ГУАП](https://guap.ru)\n2. Вопросы и ответы\n\n"
            "# Вопросы и ответы\n\n"
            "ГУАП отвечает на популярные вопросы абитуриентов.\n\n"
            "#### Почтовый адрес:\n\n"
            "190000, Россия, Санкт-Петербург, ул. Большая Морская, д. 67\n\n"
            "Разработка сайта — УЦР ГУАП\n"
        ),
    }

    doc = document_from_cloudflare_record(record, crawl_job_id="job-2")

    assert doc is not None
    assert doc.text.startswith("# Вопросы и ответы")
    assert "Почтовый адрес" not in doc.text
    assert "Разработка сайта" not in doc.text
    assert "is_low_priority_source" in doc.quality_flags


def test_document_from_cloudflare_record_trims_priem_footer_sections() -> None:
    record = {
        "url": "https://priem.guap.ru/bach/plan",
        "title": "Количество мест для приема",
        "markdown": (
            "# Количество мест для приема\n\n"
            "Информация о количестве мест для приема на обучение.\n\n"
            "## Основные документы ГУАП\n\n"
            "* [Лицензия](https://guap.ru/sveden/common)\n\n"
            "## Полезные ресурсы\n\n"
            "* [Как нас найти?](https://priem.guap.ru/contacts)\n"
        ),
    }

    doc = document_from_cloudflare_record(record, crawl_job_id="job-3")

    assert doc is not None
    assert "Основные документы ГУАП" not in doc.text
    assert "Как нас найти?" not in doc.text


def test_dedupe_documents_marks_cross_source_duplicates() -> None:
    first = document_from_cloudflare_record(
        {"url": "https://guap.ru/example", "markdown": "# Заголовок\nОдинаковый текст для проверки."}
    )
    second = document_from_cloudflare_record(
        {"url": "https://guap.ru/example?x=1", "markdown": "# Заголовок\nОдинаковый текст для проверки."}
    )
    assert first is not None
    assert second is not None

    docs = dedupe_documents([first, second])

    assert len(docs) == 2
    assert docs[0].is_duplicate is False
    assert docs[1].is_duplicate is True


def test_build_chunk_corpus_keeps_specific_dorm_page() -> None:
    doc = document_from_cloudflare_record(
        {
            "url": "https://guap.ru/dom/2",
            "title": "Общежитие №2",
            "markdown": (
                "# Общежитие №2\n\n"
                "Адрес общежития ГУАП, транспорт и сведения о проживании.\n"
                "Б. Морская 67, корпус общежития.\n"
                "В разделе размещены контактные данные, режим работы администрации, "
                "правила проживания и информация для заселения студентов.\n"
            ),
        }
    )
    assert doc is not None
    assert "dorm" in doc.metadata["source_facets"]
    assert doc.is_low_text is False

    chunks = build_chunk_corpus([doc])

    assert chunks
    assert chunks[0].canonical_url == "https://guap.ru/dom/2"


def test_build_chunk_corpus_keeps_short_official_org_page() -> None:
    doc = document_from_cloudflare_record(
        {
            "url": "https://guap.ru/struct/hr",
            "title": "Отдел кадров",
            "markdown": (
                "# Отдел кадров\n\n"
                "Телефон: +7 (812) 000-00-00\n\n"
                "Режим работы: пн-пт 10:00-18:00."
            ),
        }
    )

    assert doc is not None
    assert "org_unit" in doc.metadata["source_facets"]
    assert doc.is_low_text is False
    assert doc.is_low_signal is False

    chunks = build_chunk_corpus([doc])

    assert chunks
    assert chunks[0].canonical_url == "https://guap.ru/struct/hr"
