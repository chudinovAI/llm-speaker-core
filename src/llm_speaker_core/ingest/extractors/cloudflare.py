from __future__ import annotations

from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup


def canonicalize_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"}:
        return ""
    path = parsed.path.rstrip("/") or "/"
    return parsed._replace(path=path, query="", fragment="").geturl()


def extract_primary_text(record: dict) -> tuple[str, str]:
    for key in ("markdown", "html", "json"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip(), key
    return "empty", ""


def extract_document_links(record: dict, allowed_domain: str) -> list[str]:
    html = record.get("html")
    markdown = record.get("markdown")
    source_url = str(record.get("url", ""))
    links: list[str] = []
    if isinstance(html, str) and html.strip():
        soup = BeautifulSoup(html, "lxml")
        for anchor in soup.find_all("a", href=True):
            href = str(anchor.get("href", "")).strip()
            if not href:
                continue
            absolute = canonicalize_url(urljoin(source_url, href))
            if not absolute:
                continue
            parsed = urlparse(absolute)
            if parsed.netloc != allowed_domain:
                continue
            if not parsed.path.lower().endswith((".pdf", ".doc", ".docx")):
                continue
            links.append(absolute)
    if isinstance(markdown, str) and markdown.strip():
        for match in __import__("re").finditer(r"\[[^\]]+\]\(([^)]+)\)", markdown):
            absolute = canonicalize_url(urljoin(source_url, match.group(1)))
            if not absolute:
                continue
            parsed = urlparse(absolute)
            if parsed.netloc != allowed_domain:
                continue
            if not parsed.path.lower().endswith((".pdf", ".doc", ".docx")):
                continue
            links.append(absolute)
    return sorted(set(links))
