from __future__ import annotations

import hashlib
import re
from urllib.parse import urlparse

from llm_speaker_core.retrieval.schemas import DocumentRecord

NAVIGATION_MARKERS = {
    "главная",
    "подробнее",
    "расписание",
    "контакты",
    "обучающимся",
    "абитуриентам",
}
LOW_SIGNAL_PATH_MARKERS = ("/faq", "/greeting", "/sitemap", "/media", "/smi")
BOILERPLATE_MARKERS = (
    "разработка сайта",
    "почтовый адрес:",
    "отдел делопроизводства",
    "вопросы по работе сайта",
    "политика обработки персональных данных",
    "сведения об образовательной организации",
)


def detect_language(text: str) -> str:
    ru = len(re.findall(r"[А-Яа-яЁё]", text))
    en = len(re.findall(r"[A-Za-z]", text))
    if ru >= en:
        return "ru"
    return "other"


def normalized_fingerprint(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def similarity_signature(text: str, max_chars: int = 1200) -> str:
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    return normalized[:max_chars]


def assess_document_quality(doc: DocumentRecord) -> DocumentRecord:
    text = doc.text.strip()
    flags: set[str] = set(doc.quality_flags)
    path = urlparse(doc.canonical_url).path.rstrip("/") or "/"

    if len(text) < 120:
        flags.add("is_low_text")
    if detect_language(text) != "ru":
        flags.add("is_foreign_language")
    low = text.lower()
    if sum(marker in low for marker in NAVIGATION_MARKERS) >= 3:
        flags.add("is_navigation")
    if any(marker in path for marker in LOW_SIGNAL_PATH_MARKERS):
        flags.add("is_low_priority_source")
    if path.endswith("/sitemap"):
        flags.add("is_navigation")
        flags.add("is_low_signal")
    if low.count("гуап") == 0 and doc.source_type == "web":
        flags.add("is_low_signal")
    if sum(marker in low for marker in BOILERPLATE_MARKERS) >= 3:
        flags.add("is_boilerplate_heavy")
    quality_score = 1.0
    if "is_low_text" in flags:
        quality_score -= 0.25
    if "is_foreign_language" in flags:
        quality_score -= 0.4
    if "is_navigation" in flags:
        quality_score -= 0.35
    if "is_low_signal" in flags:
        quality_score -= 0.2
    if "is_low_priority_source" in flags:
        quality_score -= 0.18
    if "is_boilerplate_heavy" in flags:
        quality_score -= 0.22
    doc.quality_flags = sorted(flags)
    doc.is_low_text = "is_low_text" in flags
    doc.is_foreign_language = "is_foreign_language" in flags
    doc.is_navigation = "is_navigation" in flags
    doc.is_low_signal = "is_low_signal" in flags
    doc.is_boilerplate_heavy = "is_boilerplate_heavy" in flags
    doc.quality_score = round(max(0.0, quality_score), 4)
    return doc


def is_near_duplicate(
    text: str,
    seen_fingerprints: set[str],
    seen_signatures: set[str],
) -> bool:
    fingerprint = normalized_fingerprint(text)
    if fingerprint in seen_fingerprints:
        return True
    signature = similarity_signature(text)
    if signature and signature in seen_signatures:
        return True
    return False
