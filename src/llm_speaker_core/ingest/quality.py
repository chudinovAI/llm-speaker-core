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


def _classify_page_type(doc: DocumentRecord) -> str:
    path = urlparse(doc.canonical_url).path.rstrip("/") or "/"
    title = doc.title.lower()
    if doc.source_type == "doc":
        return "document"
    if path.endswith("/faq") or "вопросы и ответы" in title:
        return "faq"
    if path.endswith("/contacts") or "как нас найти" in title or "приемная комиссия" in title:
        return "contacts"
    if path.endswith("/rules") or "правила приема" in title or "положение" in title:
        return "policy"
    if path.endswith("/dates") or "сроки проведения приема" in title:
        return "schedule"
    if path.endswith("/calc") or "калькулятор" in title:
        return "catalog"
    if path.endswith("/plan") or path.endswith("/budget") or "количество мест" in title:
        return "plan"
    if path.endswith("/managers") or path.endswith("/employees"):
        return "directory"
    if path.endswith("/common") or path.endswith("/document") or path.endswith("/paid_edu") or path.endswith("/pay_edu"):
        return "reference"
    if path.endswith("/objects") or "общежит" in title or "материально-техническое обеспечение" in title:
        return "facilities"
    if any(path == value for value in ("/studlife", "/vrmp", "/sveden", "/eif", "/bach", "/mag", "/")):
        return "hub"
    if any(path.endswith(value) for value in ("/pposa", "/starsovet", "/domsovet")):
        return "organization"
    if path.startswith("/studlife/") or path.startswith("/vrmp/"):
        return "profile"
    return "detail"


def _classify_source_facets(doc: DocumentRecord) -> list[str]:
    facets: set[str] = set()
    path = urlparse(doc.canonical_url).path.rstrip("/") or "/"
    low_title = doc.title.lower()
    low_text = doc.text.lower()
    if "priem.guap.ru" in doc.canonical_url or path.startswith("/bach") or path.startswith("/mag") or path == "/":
        facets.add("admission")
    if path.startswith("/bach"):
        facets.add("admission_bach")
    if path.startswith("/mag"):
        facets.add("admission_mag")
    if path.endswith("/calc") or "калькулятор" in low_title:
        facets.add("admission_directions")
    if path.endswith("/dates") or "сроки" in low_title:
        facets.add("admission_dates")
    if path.endswith("/contacts") or "приемная комиссия" in low_title or "как нас найти" in low_title:
        facets.add("admission_contacts")
        facets.add("contacts")
    if path.endswith("/budget") or path.endswith("/plan") or "количество мест" in low_title:
        facets.add("admission_budget")
    if path.startswith("/eif") or path.startswith("/sveden/pay") or path.startswith("/sveden/paid"):
        facets.add("tuition")
    if (
        path.endswith("/price")
        or path.endswith("/paid_edu")
        or path.endswith("/pay_edu")
        or "цены обучения" in low_text
        or "стоимость обучения" in low_text
    ):
        facets.add("tuition_price")
    if path.endswith("/pay") or path.endswith("/pol_usl") or path.endswith("/inf_dog") or "оплата обучения" in low_title:
        facets.add("tuition_payment")
    if path.startswith("/sveden"):
        facets.add("official_info")
    if path.endswith("/common") or "адрес" in low_title:
        facets.add("location")
    if path.endswith("/objects") or "общежит" in low_title:
        facets.add("dorm")
        facets.add("location")
    if path.startswith("/studlife"):
        facets.add("student_life")
    if any(path.endswith(value) for value in ("/pposa", "/starsovet", "/domsovet")) or any(
        marker in low_title for marker in ("самоуправление", "профком", "совет старост", "советы общежитий")
    ):
        facets.add("student_unions")
    if path.startswith("/vrmp") or "врмп" in low_title:
        facets.add("vrmp")
    return sorted(facets)


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
    page_type = _classify_page_type(doc)
    source_facets = _classify_source_facets(doc)
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
    doc.metadata["page_type"] = page_type
    doc.metadata["source_facets"] = source_facets
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
