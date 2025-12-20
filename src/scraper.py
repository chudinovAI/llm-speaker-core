import asyncio
import hashlib
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin, urlparse
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
    OUTPUT_FILE: str = "suai_facts.txt"
    DOC_DIR: str = "guap_docs"
    MAX_PAGES: int = 200
    MAX_CONCURRENT: int = 10
    MAX_FILE_SIZE_MB: int = 10
    USER_AGENT: str = "Mozilla/5.0 (compatible; GuapStudentCrawler/1.0)"
    CONCURRENT_DOC_DOWNLOADS: int = 5


class ContentProcessor:
    def __init__(self) -> None:
        self.content_hashes: set[str] = set()

    def clean_text(self, html: str, url: str) -> tuple[None | str, list[str]]:
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
        for link in soup.find_all("a", href=True):
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
                    text += page.extract_text() + "\n"
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
                    all_doc_text.append(f"Source: {filename} ---\n{text}")
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

    def is_valid_url(self, url: str) -> bool:
        parsed = urlparse(url)
        if parsed.netloc != urlparse(self.config.START_URL).netloc:
            return False
        if re.search(r"\.(zip|rar|tar|gz|exe|jpg|png|gif|mp4)$", parsed.path.lower()):
            return False
        return True

    async def download_file(self, url: str) -> str | None:
        local_filename = os.path.join(self.config.DOC_DIR, url.split("/")[-1])

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

            self.visited_urls.add(url)
            _ = progress_bar.update(1)

            try:
                assert self.session is not None
                headers = {"User-Agent": self.config.USER_AGENT}
                timeout = aiohttp.ClientTimeout(total=15)
                async with self.session.get(
                    url, headers=headers, timeout=timeout
                ) as response:
                    html = await response.text()
            except Exception as e:
                logging.error(f"Network error on {url}: {e}")
                self.queue.task_done()
                continue

            loop = asyncio.get_event_loop()

            def clean_fn() -> tuple[str | None, list[str]]:
                return self.processor.clean_text(html, url)

            result: tuple[str | None, list[str]] = await loop.run_in_executor(
                None,
                clean_fn,
            )
            text, links = result

            if text:
                self.all_crawled_text.append(f"Source: {url} ---\n{text}")

            for link in links:
                if link.lower().endswith((".pdf", ".docx", ".doc")):
                    self.document_links.add(link)
                elif (
                    link not in self.visited_urls
                    and len(self.visited_urls) < self.config.MAX_PAGES
                ):
                    self.queue.put_nowait(link)

            self.queue.task_done()

    async def _download_with_progress(self, url: str, pbar: tqdm_base) -> str | None:
        res = await self.download_file(url)
        _ = pbar.update(1)
        return res

    async def run(self) -> str:
        self.queue.put_nowait(self.config.START_URL)
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
