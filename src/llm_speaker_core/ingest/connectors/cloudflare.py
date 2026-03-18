from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests  # type: ignore[import-untyped]

from llm_speaker_core.ingest.extractors.cloudflare import (
    canonicalize_url,
    extract_document_links,
)

TERMINAL_STATUSES = {
    "completed",
    "errored",
    "cancelled_by_user",
    "cancelled_due_to_limits",
    "cancelled_due_to_timeout",
}


@dataclass(slots=True)
class CloudflareCrawlConfig:
    account_id: str
    token: str
    start_urls: tuple[str, ...]
    raw_dir: Path
    limit: int = 100
    depth: int = 4
    render: bool = True
    source: str = "all"
    formats: tuple[str, ...] = ("markdown", "html")
    poll_interval_sec: float = 5.0
    timeout_sec: int = 1800
    page_size: int = 100
    max_retries: int = 5
    retry_backoff_sec: float = 10.0
    seed_cooldown_sec: float = 30.0

    @property
    def base_url(self) -> str:
        return (
            "https://api.cloudflare.com/client/v4/accounts/"
            f"{self.account_id}/browser-rendering/crawl"
        )


class CloudflareCrawlerClient:
    def __init__(self, config: CloudflareCrawlConfig) -> None:
        self.config = config
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.config.token}",
                "Content-Type": "application/json",
            }
        )

    def _request(
        self,
        method: str,
        path: str = "",
        *,
        params: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = self.config.base_url if not path else f"{self.config.base_url}/{path}"
        attempt = 0
        while True:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=payload,
                timeout=60,
            )
            try:
                data = response.json()
            except ValueError:
                data = {"success": False, "errors": [{"message": response.text.strip()}]}

            if response.status_code == 429 and attempt < self.config.max_retries:
                retry_after_raw = response.headers.get("Retry-After", "").strip()
                wait_s = self.config.retry_backoff_sec * (attempt + 1)
                if retry_after_raw.isdigit():
                    wait_s = max(wait_s, float(retry_after_raw))
                print(
                    f"[Cloudflare] rate limited on {method} {path or 'crawl'}; retry in {wait_s:.0f}s",
                    flush=True,
                )
                time.sleep(wait_s)
                attempt += 1
                continue

            if response.status_code >= 400:
                raise RuntimeError(
                    f"Cloudflare API HTTP {response.status_code}: {data.get('errors', data)}"
                )
            if not data.get("success", False):
                raise RuntimeError(f"Cloudflare API error: {data.get('errors', [])}")
            return data

    def start_job(self, start_url: str) -> str:
        payload: dict[str, Any] = {
            "url": start_url,
            "limit": self.config.limit,
            "depth": self.config.depth,
            "formats": list(self.config.formats),
        }
        if self.config.render is False:
            payload["render"] = False
        if self.config.source != "all":
            payload["source"] = self.config.source
        data = self._request("POST", payload=payload)
        job_id = data.get("result")
        if not isinstance(job_id, str) or not job_id:
            raise RuntimeError(f"Unexpected start response: {data}")
        return job_id

    def get_job(self, job_id: str, **params: Any) -> dict[str, Any]:
        data = self._request("GET", path=job_id, params=params or None)
        result = data.get("result")
        if not isinstance(result, dict):
            raise RuntimeError(f"Unexpected job response: {data}")
        return result

    def wait_until_done(self, job_id: str) -> dict[str, Any]:
        started = time.time()
        last_status = ""
        while True:
            result = self.get_job(job_id, limit=1)
            status = str(result.get("status", "unknown"))
            if status != last_status:
                print(f"[Cloudflare] job={job_id} status={status}", flush=True)
                last_status = status
            if status in TERMINAL_STATUSES:
                return result
            if time.time() - started > self.config.timeout_sec:
                raise TimeoutError(
                    f"Timeout after {self.config.timeout_sec}s waiting for job {job_id}"
                )
            time.sleep(self.config.poll_interval_sec)

    def fetch_all_completed_records(self, job_id: str) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {
                "status": "completed",
                "limit": self.config.page_size,
            }
            if cursor:
                params["cursor"] = cursor
            result = self.get_job(job_id, **params)
            chunk = result.get("records", [])
            if isinstance(chunk, list):
                records.extend([item for item in chunk if isinstance(item, dict)])
            next_cursor = result.get("cursor")
            if not isinstance(next_cursor, str) or not next_cursor.strip():
                break
            cursor = next_cursor
        return records


def save_raw_artifacts(
    config: CloudflareCrawlConfig,
    job_results: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> dict[str, str]:
    config.raw_dir.mkdir(parents=True, exist_ok=True)
    records_path = config.raw_dir / "records.jsonl"
    manifest_path = config.raw_dir / "crawl_manifest.json"
    doc_links_path = config.raw_dir / "document_links.jsonl"
    allowed_domains = {
        canonicalize_url(url).split("/")[2]
        for url in config.start_urls
        if canonicalize_url(url)
    }
    if not allowed_domains:
        allowed_domains = {"guap.ru"}

    with records_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    doc_links: set[str] = set()
    for record in records:
        for allowed_domain in allowed_domains:
            doc_links.update(extract_document_links(record, allowed_domain))
    with doc_links_path.open("w", encoding="utf-8") as f:
        for url in sorted(doc_links):
            f.write(json.dumps({"url": url}, ensure_ascii=False) + "\n")

    manifest = {
        "job_ids": [row["job_id"] for row in job_results],
        "start_urls": [str(row.get("start_url", "")) for row in job_results if row.get("start_url")],
        "record_count": len(records),
        "status": "completed",
        "jobs": job_results,
        "records_path": str(records_path),
        "document_links_path": str(doc_links_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "records_path": str(records_path),
        "manifest_path": str(manifest_path),
        "document_links_path": str(doc_links_path),
    }


def _seed_slug(start_url: str) -> str:
    canonical = canonicalize_url(start_url)
    if not canonical:
        return "seed"
    slug = canonical.split("://", 1)[-1].strip("/")
    return slug.replace("/", "__").replace(".", "_") or "seed"


def save_seed_artifacts(
    config: CloudflareCrawlConfig,
    start_url: str,
    job_id: str,
    final_result: dict[str, Any],
    records: list[dict[str, Any]],
) -> None:
    seeds_dir = config.raw_dir / "seeds"
    seeds_dir.mkdir(parents=True, exist_ok=True)
    slug = _seed_slug(start_url)
    records_path = seeds_dir / f"{slug}.records.jsonl"
    manifest_path = seeds_dir / f"{slug}.manifest.json"
    with records_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest_path.write_text(
        json.dumps(
            {
                "job_id": job_id,
                "start_url": start_url,
                "status": final_result.get("status"),
                "record_count": len(records),
                "records_path": str(records_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def load_saved_seed_records(raw_dir: Path) -> list[dict[str, Any]]:
    seeds_dir = raw_dir / "seeds"
    if not seeds_dir.exists():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(seeds_dir.glob("*.records.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                records.append(row)
    return records


def load_saved_seed_jobs(raw_dir: Path) -> list[dict[str, Any]]:
    seeds_dir = raw_dir / "seeds"
    if not seeds_dir.exists():
        return []
    jobs: list[dict[str, Any]] = []
    for path in sorted(seeds_dir.glob("*.manifest.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(row, dict):
            jobs.append(row)
    return jobs


def _dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for record in records:
        url = canonicalize_url(str(record.get("url", "")).strip())
        key = url or str(record.get("url", "")).strip()
        if not key:
            continue
        current = deduped.get(key)
        if current is None:
            deduped[key] = record
            continue
        current_md = len(str(current.get("markdown", "")))
        candidate_md = len(str(record.get("markdown", "")))
        if candidate_md > current_md:
            deduped[key] = record
    return list(deduped.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Cloudflare crawl and persist raw artifacts.")
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--api-token", required=True)
    parser.add_argument("--start-url", action="append", dest="start_urls")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/cloudflare/latest"))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--seed-cooldown", type=float, default=30.0)
    parser.add_argument("--poll-job-id", default="")
    parser.add_argument("--export-job-id", default="")
    parser.add_argument("--start-only", action="store_true")
    parser.add_argument("--rebuild-only", action="store_true")
    args = parser.parse_args()

    start_urls = tuple(
        args.start_urls
        or [
            "https://guap.ru",
            "https://priem.guap.ru",
            "https://guap.ru/sveden",
            "https://guap.ru/eif",
            "https://guap.ru/studlife",
            "https://guap.ru/vrmp",
        ]
    )

    config = CloudflareCrawlConfig(
        account_id=args.account_id,
        token=args.api_token,
        start_urls=start_urls,
        raw_dir=args.raw_dir,
        limit=args.limit,
        depth=args.depth,
        poll_interval_sec=args.poll_interval,
        timeout_sec=args.timeout,
        seed_cooldown_sec=args.seed_cooldown,
    )
    client = CloudflareCrawlerClient(config)
    all_records: list[dict[str, Any]] = []
    job_results: list[dict[str, Any]] = []
    if args.rebuild_only:
        payload = save_raw_artifacts(
            config,
            load_saved_seed_jobs(config.raw_dir),
            _dedupe_records(load_saved_seed_records(config.raw_dir)),
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    if args.poll_job_id:
        result = client.get_job(args.poll_job_id, limit=1)
        print(
            json.dumps(
                {
                    "job_id": args.poll_job_id,
                    "status": result.get("status"),
                    "records": result.get("total_records"),
                    "pages_crawled": result.get("pages_crawled"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return
    if args.export_job_id:
        print(
            f"[Cloudflare] exporting existing job={args.export_job_id} seed={start_urls[0]}",
            flush=True,
        )
        final_result = client.wait_until_done(args.export_job_id)
        if str(final_result.get("status")) != "completed":
            raise SystemExit(
                f"Crawl finished with non-completed status: {final_result.get('status')}"
            )
        records = client.fetch_all_completed_records(args.export_job_id)
        all_records.extend(records)
        save_seed_artifacts(
            config,
            start_urls[0],
            args.export_job_id,
            final_result,
            records,
        )
        save_raw_artifacts(
            config,
            load_saved_seed_jobs(config.raw_dir),
            _dedupe_records(load_saved_seed_records(config.raw_dir)),
        )
    else:
        for start_url in start_urls:
            print(f"[Cloudflare] starting crawl for {start_url}", flush=True)
            job_id = client.start_job(start_url)
            print(
                json.dumps(
                    {"job_id": job_id, "start_url": start_url, "action": "started"},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if args.start_only:
                continue
            final_result = client.wait_until_done(job_id)
            if str(final_result.get("status")) != "completed":
                raise SystemExit(
                    f"Crawl for {start_url} finished with non-completed status: {final_result.get('status')}"
                )
            records = client.fetch_all_completed_records(job_id)
            all_records.extend(records)
            save_seed_artifacts(config, start_url, job_id, final_result, records)
            job_results.append(
                {
                    "job_id": job_id,
                    "start_url": start_url,
                    "status": final_result.get("status"),
                    "record_count": len(records),
                }
            )
            save_raw_artifacts(
                config,
                load_saved_seed_jobs(config.raw_dir),
                _dedupe_records(load_saved_seed_records(config.raw_dir)),
            )
            print(
                f"[Cloudflare] cooldown before next seed: {config.seed_cooldown_sec:.0f}s",
                flush=True,
            )
            time.sleep(config.seed_cooldown_sec)
        if args.start_only:
            return
    payload = save_raw_artifacts(
        config,
        load_saved_seed_jobs(config.raw_dir),
        _dedupe_records(load_saved_seed_records(config.raw_dir) or all_records),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
