from __future__ import annotations

from llm_speaker_core.ingest.extractors.cloudflare import canonicalize_url
from llm_speaker_core.ingest.normalize import document_from_cloudflare_record


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
