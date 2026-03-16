from __future__ import annotations

from llm_speaker_core.ingest.normalize import document_from_cloudflare_record


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
