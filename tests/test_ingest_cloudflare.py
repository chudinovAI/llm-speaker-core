from __future__ import annotations

import json
from pathlib import Path

from llm_speaker_core.ingest.extractors.cloudflare import canonicalize_url
from llm_speaker_core.ingest.normalize import (
    dedupe_documents,
    document_from_cloudflare_record,
    load_firecrawl_documents,
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


def test_load_firecrawl_documents_reads_manifest_and_sets_urls(tmp_path: Path) -> None:
    root = tmp_path / "firecrawl"
    root.mkdir()
    (root / "sample.md").write_text("# Библиотека ГУАП\nКонтакты и часы работы библиотеки.", encoding="utf-8")
    (root / "firecrawl_manifest.json").write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "file_name": "sample.md",
                        "source_url": "http://lib.guap.ru/",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    docs = load_firecrawl_documents(root)

    assert len(docs) == 1
    assert docs[0].source_type == "firecrawl"
    assert docs[0].canonical_url == "http://lib.guap.ru/"
    assert docs[0].metadata["ingest_source"] == "firecrawl"
    assert docs[0].metadata["page_type"] == "library"
    assert "library" in docs[0].metadata["source_facets"]


def test_load_firecrawl_documents_marks_faculty_contacts_without_admission_bias(tmp_path: Path) -> None:
    root = tmp_path / "firecrawl"
    root.mkdir()
    (root / "faculty.md").write_text(
        "# Контакты\nТелефон: (812) 571-16-89\nЭл. почта: aerospace1@guap.ru",
        encoding="utf-8",
    )
    (root / "firecrawl_manifest.json").write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "file_name": "faculty.md",
                        "source_url": "https://new.guap.ru/i01/contacts",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    docs = load_firecrawl_documents(root)

    assert len(docs) == 1
    assert "faculty_contacts" in docs[0].metadata["source_facets"]
    assert "admission_contacts" not in docs[0].metadata["source_facets"]


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


def test_dedupe_documents_preserves_template_like_faculty_contacts(tmp_path: Path) -> None:
    root = tmp_path / "firecrawl"
    root.mkdir()
    common_text = (
        "# Контакты\n"
        "Телефон: (812) 571-16-89\n"
        "Эл. почта: aerospace1@guap.ru\n"
        "Директор института и заместители."
    )
    (root / "i01.md").write_text(common_text, encoding="utf-8")
    (root / "i02.md").write_text(common_text.replace("aerospace1", "radio2"), encoding="utf-8")
    (root / "firecrawl_manifest.json").write_text(
        json.dumps(
            {
                "entries": [
                    {"file_name": "i01.md", "source_url": "https://new.guap.ru/i01/contacts"},
                    {"file_name": "i02.md", "source_url": "https://new.guap.ru/i02/contacts"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    docs = dedupe_documents(load_firecrawl_documents(root))

    assert len(docs) == 2
    assert docs[0].is_duplicate is False
    assert docs[1].is_duplicate is False
