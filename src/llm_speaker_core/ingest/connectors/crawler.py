"""Standalone website crawler that outputs records.jsonl compatible with the existing pipeline."""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse

import html2text
import requests
from bs4 import BeautifulSoup

from llm_speaker_core.ingest.extractors.cloudflare import canonicalize_url

DEFAULT_SEEDS = [
    "https://guap.ru",
    "https://guap.ru/sveden",
    "https://guap.ru/studlife",
    "https://guap.ru/eif",
    "https://guap.ru/vrmp",
    "https://guap.ru/struct",
    "https://guap.ru/career",
    "https://guap.ru/aspira",
    "https://guap.ru/fdpo",
    "https://guap.ru/dom",
    "https://guap.ru/med",
    "https://guap.ru/sport",
    "https://guap.ru/science",
    "https://guap.ru/inter",
    "https://guap.ru/it",
    "https://guap.ru/pubs",
    "https://priem.guap.ru",
    "https://lib.guap.ru",
]

ALLOWED_DOMAINS = {"guap.ru", "priem.guap.ru", "lib.guap.ru"}

SKIP_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".rar", ".7z", ".tar", ".gz",
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico",
    ".mp3", ".mp4", ".avi", ".mov", ".wmv",
    ".css", ".js", ".json", ".xml", ".rss",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.5",
}


def _is_crawlable(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    if parsed.netloc not in ALLOWED_DOMAINS:
        return False
    suffix = Path(parsed.path).suffix.lower()
    if suffix in SKIP_EXTENSIONS:
        return False
    return True


def _extract_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: list[str] = []
    for tag in soup.find_all("a", href=True):
        href = str(tag["href"]).strip()
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        absolute = urljoin(base_url, href)
        absolute = absolute.split("#")[0].split("?")[0]
        if _is_crawlable(absolute):
            links.append(absolute)
    return links


def _extract_title(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    tag = soup.find("title")
    if tag and tag.string:
        return tag.string.strip()
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)
    return ""


def _html_to_markdown(html: str) -> str:
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = True
    h.ignore_emphasis = False
    h.body_width = 0
    h.unicode_snob = True
    return h.handle(html).strip()


def crawl(
    seeds: list[str],
    max_pages: int = 2000,
    delay: float = 0.3,
    timeout: int = 15,
) -> list[dict]:
    visited: set[str] = set()
    queue: deque[str] = deque()

    for seed in seeds:
        canonical = canonicalize_url(seed)
        if canonical and canonical not in visited:
            queue.append(seed)
            visited.add(canonical)

    records: list[dict] = []
    session = requests.Session()
    session.headers.update(HEADERS)
    errors = 0

    print(f"[crawler] Starting with {len(seeds)} seeds, max_pages={max_pages}", flush=True)

    while queue and len(records) < max_pages:
        url = queue.popleft()
        try:
            resp = session.get(url, timeout=timeout, allow_redirects=True)
            if resp.status_code != 200:
                continue
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                continue
            html = resp.text
        except Exception:
            errors += 1
            continue

        title = _extract_title(html)
        markdown = _html_to_markdown(html)

        if len(markdown.split()) < 10:
            continue

        records.append({
            "url": resp.url,
            "html": html,
            "markdown": markdown,
            "title": title,
        })

        if len(records) % 50 == 0:
            print(f"[crawler] {len(records)} pages, queue={len(queue)}, errors={errors}", flush=True)

        for link in _extract_links(html, resp.url):
            canonical = canonicalize_url(link)
            if canonical and canonical not in visited:
                visited.add(canonical)
                queue.append(link)

        time.sleep(delay)

    print(
        f"[crawler] Done: {len(records)} pages, {errors} errors, {len(visited)} URLs seen",
        flush=True,
    )
    return records


def save_records(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[crawler] Saved {len(records)} records to {output_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Crawl GUAP website and save records.jsonl")
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/raw/cloudflare/latest/records.jsonl"),
    )
    parser.add_argument("--max-pages", type=int, default=2000)
    parser.add_argument("--delay", type=float, default=0.3)
    parser.add_argument("--seed", action="append", dest="seeds")
    args = parser.parse_args()

    seeds = args.seeds or DEFAULT_SEEDS
    records = crawl(seeds, max_pages=args.max_pages, delay=args.delay)
    save_records(records, args.output)


if __name__ == "__main__":
    main()
