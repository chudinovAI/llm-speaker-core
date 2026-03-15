import asyncio
import hashlib
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse
import aiohttp
import pypdf
from bs4 import BeautifulSoup
from docx import Document
from tqdm import tqdm as sync_tqdm
from tqdm.asyncio import tqdm as tqdm_asyncio
from tqdm.std import tqdm as tqdm_base

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Config:
    START_URL: str = "https://guap.ru"
    SEED_URLS: tuple[str, ...] = (
        "https://guap.ru/sveden/common",
        "https://guap.ru/eif/pay",
        "https://guap.ru/eif/inf_dog",
        "https://guap.ru/studlife/theatre",
        "https://guap.ru/targets/studs",
        "https://guap.ru/priem",
        "https://guap.ru/abitur",
    )
    OUTPUT_FILE: str = "suai_facts.txt"
    DOC_DIR: str = "guap_docs"
    MAX_PAGES: int = 200
    MAX_CONCURRENT: int = 10
    MAX_FILE_SIZE_MB: int = 25
    USER_AGENT: str = "Mozilla/5.0 (compatible; GuapStudentCrawler/1.0)"
    CONCURRENT_DOC_DOWNLOADS: int = 5
    SECTION_PAGE_LIMITS: dict[str, int] = {"pubs": 8, "messages": 8}
    DOC_URL_ALLOW_HINTS: tuple[str, ...] = (
        "/sveden/",
        "/eif/",
        "/priem",
        "/abitur",
        "stoim",
        "oplata",
        "dogovor",
        "pravila",
        "polozhen",
        "contact",
        "address",
    )
    DOC_URL_SKIP_HINTS: tuple[str, ...] = (
        "sbor",
        "sputnik",
        "program",
        "/pubs/",
        "/messages/",
    )


class ContentProcessor:
    def __init__(self) -> None:
        self.content_hashes: set[str] = set()

    def clean_text(self, html: str, url: str) -> tuple[str, list[str]]:
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(
            ["script", "style", "header", "footer", "nav", "aside", "form"]
        ):
            tag.decompose()
        main_content = soup.find(
            ["main", "article"],
            class_=re.compile("(main-content|contnet|post-body|body-content)"),
        )
        if not main_content:
            main_content = soup.body
        if not main_content:
            return "", []

        text = main_content.getText(separator="\n", strip=True)
        text = re.sub(r"(\n\s*){2,}", "\n\n", text)

        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if content_hash in self.content_hashes:
            logging.debug(f"Duplicate content skipped: {url}")
            return "", []

        self.content_hashes.add(content_hash)

        new_links: list[str] = []
        for link in main_content.find_all("a", href=True):
            href = link.get("href")
            if not isinstance(href, str):
                continue
            absolute_url = urljoin(url, href)
            new_links.append(absolute_url)

        return text, new_links


class DocumentHandler:
    def __init__(self, output_dir: str) -> None:
        self.output_dir: str = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def process_single_doc(self, filepath: str) -> str:
        text = ""
        try:
            if filepath.lower().endswith(".pdf"):
                reader = pypdf.PdfReader(filepath)
                for page in reader.pages:
                    text += (page.extract_text() or "") + "\n"
            elif filepath.lower().endswith(".docx"):
                document = Document(filepath)
                paragraphs: list[str] = [p.text for p in document.paragraphs]
                text = "\n".join(paragraphs)
            return text.strip()
        except Exception as e:
            logging.error(f"Error processing {filepath}: {e}")
            return ""

    def process_all_docs_parallel(self, doc_paths: list[str]) -> str:
        all_doc_text: list[str] = []
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
            results = list(
                sync_tqdm(
                    executor.map(self.process_single_doc, doc_paths),
                    total=len(doc_paths),
                    desc="Processing documents",
                )
            )
            for i, text in enumerate(results):
                if text:
                    filename = os.path.basename(doc_paths[i])
                    all_doc_text.append(
                        f"--- Источник Документ: {filename} ---\n{text}"
                    )
        return "\n\n".join(all_doc_text)


class Crawler:
    def __init__(
        self, config: Config, processor: ContentProcessor, doc_handler: DocumentHandler
    ) -> None:
        self.config: Config = config
        self.processor: ContentProcessor = processor
        self.doc_handler: DocumentHandler = doc_handler
        self.visited_urls: set[str] = set()
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.document_links: set[str] = set()
        self.all_crawled_text: list[str] = []
        self.session: aiohttp.ClientSession | None = None
        self.section_counts: dict[str, int] = {}
        self.doc_download_semaphore = asyncio.Semaphore(
            self.config.CONCURRENT_DOC_DOWNLOADS
        )

    def canonicalize_url(self, url: str) -> str:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return ""
        if parsed.netloc != urlparse(self.config.START_URL).netloc:
            return ""

        query_pairs = parse_qsl(parsed.query, keep_blank_values=False)
        kept_pairs = [
            (k, v)
            for k, v in query_pairs
            if not (k.lower().startswith("utm_") or k.lower() in {"yclid", "gclid"})
        ]
        normalized_query = urlencode(sorted(kept_pairs))
        parsed = parsed._replace(fragment="", query=normalized_query)
        return urlunparse(parsed)

    def is_valid_url(self, url: str) -> bool:
        parsed = urlparse(url)
        if not url or parsed.netloc != urlparse(self.config.START_URL).netloc:
            return False
        if parsed.scheme not in {"http", "https"}:
            return False
        if parsed.path.startswith("/!http") or parsed.path.startswith("/!https"):
            return False
        if parsed.path.endswith("/search") or parsed.path.endswith("/search/"):
            return False
        if re.search(r"\.(zip|rar|tar|gz|exe|jpg|png|gif|mp4)$", parsed.path.lower()):
            return False
        return True

    def _section_key(self, url: str) -> str:
        parsed = urlparse(url)
        parts = [p for p in parsed.path.lower().split("/") if p]
        return parts[0] if parts else "root"

    def _section_limit_reached(self, url: str) -> bool:
        key = self._section_key(url)
        limit = self.config.SECTION_PAGE_LIMITS.get(key)
        if limit is None:
            return False
        return self.section_counts.get(key, 0) >= limit

    def _register_section_visit(self, url: str) -> None:
        key = self._section_key(url)
        self.section_counts[key] = self.section_counts.get(key, 0) + 1

    def is_relevant_doc_link(self, url: str) -> bool:
        path = urlparse(url).path.lower()
        if any(h in path for h in self.config.DOC_URL_SKIP_HINTS):
            return False
        allow_hints = self.config.DOC_URL_ALLOW_HINTS
        if allow_hints and not any(h in path for h in allow_hints):
            return False
        return True

    async def download_file(self, url: str) -> str | None:
        parsed = urlparse(url)
        basename = os.path.basename(parsed.path) or "document.bin"
        safe_basename = re.sub(r"[^a-zA-Z0-9._-]+", "_", basename)
        unique_prefix = hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
        local_filename = os.path.join(
            self.config.DOC_DIR, f"{unique_prefix}_{safe_basename}"
        )

        try:
            assert self.session is not None
            timeout = aiohttp.ClientTimeout(total=15)
            async with self.session.get(url, timeout=timeout) as response:
                if response.status != 200:
                    logging.warning(f"File download failed ({response.status}): {url}")
                    return None

                if "Content-Length" in response.headers:
                    size_bytes = int(response.headers["Content-Length"])
                    if size_bytes > self.config.MAX_FILE_SIZE_MB * 1024 * 1024:
                        logging.warning(
                            f"Skipping large file ({size_bytes / 1e6:.1f}MB): {url}"
                        )
                        return None

                with open(local_filename, "wb") as f:
                    while True:
                        chunk = await response.content.read(8192)
                        if not chunk:
                            break
                        _ = f.write(chunk)

            return local_filename
        except Exception as e:
            logging.error(f"Error downloading {url}: {e}")
            return None

    async def worker(self, progress_bar: tqdm_base) -> None:
        while len(self.visited_urls) < self.config.MAX_PAGES:
            try:
                url = await asyncio.wait_for(self.queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                break

            if url in self.visited_urls or not self.is_valid_url(url):
                self.queue.task_done()
                continue
            if self._section_limit_reached(url):
                self.queue.task_done()
                continue

            self.visited_urls.add(url)
            self._register_section_visit(url)
            _ = progress_bar.update(1)

            try:
                assert self.session is not None
                headers = {"User-Agent": self.config.USER_AGENT}
                timeout = aiohttp.ClientTimeout(total=15)
                async with self.session.get(
                    url, headers=headers, timeout=timeout
                ) as response:
                    content_type = response.headers.get("Content-Type", "").lower()
                    if "text/html" not in content_type:
                        self.queue.task_done()
                        continue
                    html = await response.text()
            except Exception as e:
                logging.error(f"Network error on {url}: {e}")
                self.queue.task_done()
                continue

            loop = asyncio.get_event_loop()

            def clean_fn() -> tuple[str, list[str]]:
                return self.processor.clean_text(html, url)

            result: tuple[str, list[str]] = await loop.run_in_executor(
                None,
                clean_fn,
            )
            text, links = result

            if text:
                self.all_crawled_text.append(f"--- Источник: {url} ---\n{text}")

            for link in links:
                if link.startswith(("mailto:", "javascript:", "tel:")):
                    continue

                normalized_link = self.canonicalize_url(link)
                if not normalized_link:
                    continue

                path_lower = urlparse(normalized_link).path.lower()
                if path_lower.endswith((".pdf", ".docx", ".doc")):
                    if self.is_relevant_doc_link(normalized_link):
                        self.document_links.add(normalized_link)
                elif (
                    normalized_link not in self.visited_urls
                    and len(self.visited_urls) < self.config.MAX_PAGES
                ):
                    self.queue.put_nowait(normalized_link)

            self.queue.task_done()

    async def _download_with_progress(self, url: str, pbar: tqdm_base) -> str | None:
        async with self.doc_download_semaphore:
            res = await self.download_file(url)
        _ = pbar.update(1)
        return res

    async def run(self) -> str:
        start_url = self.canonicalize_url(self.config.START_URL)
        if not start_url:
            raise ValueError(f"Invalid START_URL: {self.config.START_URL}")
        seed_urls = [self.config.START_URL, *self.config.SEED_URLS]
        for seed in seed_urls:
            normalized = self.canonicalize_url(seed)
            if normalized:
                self.queue.put_nowait(normalized)
        async with aiohttp.ClientSession(
            headers={"User-Agent": self.config.USER_AGENT}
        ) as session:
            self.session = session

            with tqdm_asyncio(
                total=self.config.MAX_PAGES, desc="Crawling web pages"
            ) as progress_bar:
                workers: list[asyncio.Task[None]] = [
                    asyncio.create_task(self.worker(progress_bar))
                    for _ in range(self.config.MAX_CONCURRENT)
                ]
                _ = await asyncio.gather(*workers)

            with tqdm_asyncio(
                total=len(self.document_links), desc="Downloading documents"
            ) as doc_pbar:
                download_tasks: list[asyncio.Task[str | None]] = [
                    asyncio.create_task(self._download_with_progress(url, doc_pbar))
                    for url in self.document_links
                ]

                results: list[str | None] = await asyncio.gather(*download_tasks)

            doc_paths: list[str] = [path for path in results if path is not None]

        doc_text = self.doc_handler.process_all_docs_parallel(doc_paths)

        final_text = "\n\n".join(self.all_crawled_text) + "\n\n" + doc_text

        with open(self.config.OUTPUT_FILE, "w", encoding="utf-8") as f:
            _ = f.write(final_text)

        return (
            f"Scraping complete. Total pages: {len(self.visited_urls)}. "
            f"Docs found: {len(self.document_links)}. Docs processed: {len(doc_paths)}. "
            f"Output: {self.config.OUTPUT_FILE}"
        )


if __name__ == "__main__":
    processor = ContentProcessor()
    doc_handler = DocumentHandler(Config.DOC_DIR)
    crawler = Crawler(Config(), processor, doc_handler)
    print(asyncio.run(crawler.run()))
