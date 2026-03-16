from __future__ import annotations

from llm_speaker_core.ingest.connectors.cloudflare import _dedupe_records


def test_dedupe_records_prefers_richer_markdown() -> None:
    records = [
        {"url": "https://guap.ru/eif?x=1", "markdown": "short"},
        {"url": "https://guap.ru/eif", "markdown": "much richer markdown content"},
    ]

    deduped = _dedupe_records(records)

    assert len(deduped) == 1
    assert deduped[0]["markdown"] == "much richer markdown content"
