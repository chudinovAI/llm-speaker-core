from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

# Config can be set here or via env vars.
ACCOUNT_ID = ""
TOKEN = ""
JOB_ID = ""

# Env vars override local constants.
ACCOUNT_ID = os.getenv("CF_ACCOUNT_ID", ACCOUNT_ID)
TOKEN = os.getenv("CF_API_TOKEN", TOKEN)
JOB_ID = os.getenv("CF_CRAWL_JOB_ID", JOB_ID)

TERMINAL_STATUSES = {
    "completed",
    "errored",
    "cancelled_by_user",
    "cancelled_due_to_limits",
    "cancelled_due_to_timeout",
}


@dataclass(slots=True)
class CrawlConfig:
    account_id: str
    token: str
    output_file: str = "cloudflare_crawl_output.txt"
    output_json: str = "cloudflare_crawl_records.json"
    poll_interval_sec: float = 5.0
    timeout_sec: int = 1800
    page_size: int = 100

    @property
    def base_url(self) -> str:
        return (
            "https://api.cloudflare.com/client/v4/accounts/"
            f"{self.account_id}/browser-rendering/crawl"
        )


class CloudflareCrawlerClient:
    def __init__(self, config: CrawlConfig) -> None:
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
        response = self.session.request(
            method=method,
            url=url,
            params=params,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        if not data.get("success", False):
            errors = data.get("errors", [])
            raise RuntimeError(f"Cloudflare API error: {errors}")
        return data

    def start_job(
        self,
        start_url: str,
        *,
        limit: int,
        depth: int,
        render: bool,
        source: str,
        formats: list[str],
    ) -> str:
        payload: dict[str, Any] = {
            "url": start_url,
            "limit": limit,
            "depth": depth,
            "render": render,
            "source": source,
            "formats": formats,
        }
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
        while True:
            result = self.get_job(job_id, limit=1)
            status = str(result.get("status", "unknown"))
            finished = result.get("finished")
            total = result.get("total")
            print(f"status={status}, finished={finished}, total={total}")

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
                valid_items = [item for item in chunk if isinstance(item, dict)]
                records.extend(valid_items)

            next_cursor = result.get("cursor")
            if not isinstance(next_cursor, str) or not next_cursor.strip():
                break
            cursor = next_cursor

        return records


def record_to_text(record: dict[str, Any]) -> str:
    url = str(record.get("url", ""))
    status = str(record.get("status", ""))
    metadata = record.get("metadata", {})
    title = ""
    if isinstance(metadata, dict):
        title = str(metadata.get("title", ""))

    content = ""
    for key in ("markdown", "html", "json"):
        if key in record and record[key] is not None:
            value = record[key]
            if isinstance(value, str):
                content = value.strip()
            else:
                content = json.dumps(value, ensure_ascii=False, indent=2)
            break

    if not content:
        content = "<empty>"

    header = f"--- Source: {url} | status={status}"
    if title:
        header += f" | title={title}"
    header += " ---"
    return f"{header}\n{content}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cloudflare /crawl parser")
    parser.add_argument("--url", help="Start URL for a new crawl job", default="")
    parser.add_argument("--job-id", help="Existing crawl job id", default=JOB_ID)
    parser.add_argument(
        "--output", default="cloudflare_crawl_output.txt", help="Text output path"
    )
    parser.add_argument(
        "--output-json",
        default="cloudflare_crawl_records.json",
        help="Raw records JSON path",
    )
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--render", action="store_true", default=False)
    parser.add_argument("--source", default="all", choices=["all", "sitemaps", "links"])
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["markdown"],
        help="Response formats, e.g. markdown html json",
    )
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Do not wait for completion, only print job id",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not ACCOUNT_ID or not TOKEN:
        raise SystemExit("Set CF_ACCOUNT_ID and CF_API_TOKEN environment variables.")

    config = CrawlConfig(
        account_id=ACCOUNT_ID,
        token=TOKEN,
        output_file=args.output,
        output_json=args.output_json,
        poll_interval_sec=args.poll_interval,
        timeout_sec=args.timeout,
        page_size=args.page_size,
    )
    client = CloudflareCrawlerClient(config)

    job_id = args.job_id
    if args.url:
        print(f"Starting crawl for: {args.url}")
        job_id = client.start_job(
            args.url,
            limit=args.limit,
            depth=args.depth,
            render=args.render,
            source=args.source,
            formats=args.formats,
        )
        print(f"Created job_id: {job_id}")

    if not job_id:
        raise SystemExit("Provide --url to start a crawl or --job-id to read an existing one.")

    if args.no_wait:
        print(job_id)
        return

    final_result = client.wait_until_done(job_id)
    final_status = str(final_result.get("status", "unknown"))
    print(f"Final status: {final_status}")

    if final_status != "completed":
        raise SystemExit(f"Crawl finished with status: {final_status}")

    records = client.fetch_all_completed_records(job_id)
    print(f"Fetched records: {len(records)}")

    output_json_path = Path(config.output_json)
    output_json_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    text_blocks = [record_to_text(record) for record in records]
    output_txt_path = Path(config.output_file)
    output_txt_path.write_text("\n\n".join(text_blocks), encoding="utf-8")

    print(f"Saved JSON: {output_json_path}")
    print(f"Saved text: {output_txt_path}")


if __name__ == "__main__":
    main()
