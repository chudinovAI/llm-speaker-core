from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path
from typing import Iterator

from llm_speaker_core.ingest.connectors.cloudflare import _dedupe_records
from llm_speaker_core.ingest.connectors.cloudflare import (
    CloudflareCrawlConfig,
    CloudflareCrawlerClient,
)


def test_dedupe_records_prefers_richer_markdown() -> None:
    records = [
        {"url": "https://guap.ru/eif?x=1", "markdown": "short"},
        {"url": "https://guap.ru/eif", "markdown": "much richer markdown content"},
    ]

    deduped = _dedupe_records(records)

    assert len(deduped) == 1
    assert deduped[0]["markdown"] == "much richer markdown content"


def test_wait_until_done_flushes_status_output(monkeypatch, tmp_path: Path) -> None:
    config = CloudflareCrawlConfig(
        account_id="acc",
        token="tok",
        start_urls=("https://guap.ru",),
        raw_dir=tmp_path,
        poll_interval_sec=0.0,
        timeout_sec=1,
    )
    client = CloudflareCrawlerClient(config)
    states: Iterator[dict[str, object]] = iter(
        [
            {"status": "running"},
            {"status": "completed"},
        ]
    )

    def fake_get_job(job_id: str, **params: object) -> dict[str, object]:
        return next(states)

    monkeypatch.setattr(client, "get_job", fake_get_job)
    buf = io.StringIO()
    with redirect_stdout(buf):
        result = client.wait_until_done("job-1")

    assert result["status"] == "completed"
    output = buf.getvalue()
    assert "job=job-1 status=running" in output
    assert "job=job-1 status=completed" in output
