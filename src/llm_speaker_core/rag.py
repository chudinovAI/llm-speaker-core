from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

TOKEN_RE = re.compile(r"[a-zA-Zа-яА-Я0-9_]+")
RU_STOPWORDS = {
    "и",
    "в",
    "на",
    "по",
    "с",
    "к",
    "о",
    "об",
    "у",
    "за",
    "из",
    "для",
    "что",
    "как",
    "какие",
    "какой",
    "где",
    "это",
    "есть",
    "ли",
    "или",
}
QUERY_EXPANSIONS = {
    "поступ": ("абитуриент", "прием", "приём", "баллы", "направления"),
    "приемн": ("приемная", "приёмная", "комиссия", "контакты", "поступление"),
    "общежит": ("проживание", "заселение", "кампус"),
    "стипенд": ("выплаты", "поддержка", "социальная"),
    "гуап": ("университет", "аэрокосмического", "приборостроения"),
}
GENERIC_BRAND_TOKENS = {
    "гуап",
    "университет",
    "аэрокосмического",
    "приборостроения",
    "санкт",
    "петербургский",
}
INTENT_HINTS: dict[str, tuple[str, ...]] = {
    "admission": ("поступ", "абитуриент", "прием", "приём", "баллы", "комисси"),
    "contacts": ("контакт", "связ", "телефон", "почт", "email"),
    "location": ("где", "адрес", "корпус", "кампус", "наход"),
    "student_life": ("студен", "актив", "круж", "клуб", "театр", "спорт"),
}
INTENT_SOURCE_HINTS: dict[str, tuple[str, ...]] = {
    "admission": ("/priem", "/abitur", "комис", "admission"),
    "contacts": ("/contacts", "/contact", "/sveden/common#address", "address"),
    "location": ("/contacts", "/sveden/common#address", "address"),
    "student_life": ("/studlife", "/students", "/clubs", "/theatre", "/sport"),
}
GENERIC_SOURCE_PATTERNS = (
    "/sveden/common",
    "/prioritet2030/cons/ufs",
    "/search",
)


def tokenize(text: str) -> list[str]:
    tokens = [t.lower() for t in TOKEN_RE.findall(text)]
    return [t for t in tokens if t not in RU_STOPWORDS and len(t) > 1]


def expand_query(tokens: list[str]) -> list[str]:
    expanded = list(tokens)
    for token in tokens:
        for stem, extra in QUERY_EXPANSIONS.items():
            if token.startswith(stem):
                expanded.extend(extra)
    return expanded


class LexicalRAG:
    """BM25-like lexical retriever with lightweight reranking."""

    def __init__(self, index_path: Path) -> None:
        self.index_path = index_path
        self.docs: list[dict] = []
        self.doc_freqs: dict[str, int] = {}
        self.doc_tfs: list[dict[str, int]] = []
        self.doc_lens: list[int] = []
        self.avg_doc_len: float = 1.0
        self._loaded = False

    def load(self) -> None:
        if not self.index_path.exists():
            raise FileNotFoundError(f"RAG index not found: {self.index_path}")

        payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        self.docs = payload["docs"]
        self.doc_freqs = {k: int(v) for k, v in payload["doc_freqs"].items()}
        self.doc_tfs = [{k: int(v) for k, v in tf.items()} for tf in payload["doc_tfs"]]
        self.doc_lens = [int(x) for x in payload["doc_lens"]]
        self.avg_doc_len = sum(self.doc_lens) / max(len(self.doc_lens), 1)
        self._loaded = True

    def _keyword_coverage(self, query_tokens: set[str], text_tokens: set[str]) -> float:
        if not query_tokens:
            return 0.0
        return len(query_tokens & text_tokens) / len(query_tokens)

    def _source_bonus(self, doc: dict, query_tokens: set[str]) -> float:
        source = str(doc.get("source", "")).lower()
        title = str(doc.get("title", "")).lower()
        bonus = 0.0
        for token in query_tokens:
            if token in title:
                bonus += 0.07
            if token in source:
                bonus += 0.05
        if doc.get("source_type") == "web":
            bonus += 0.02
        return min(bonus, 0.28)

    def _detect_intents(self, query_tokens: set[str]) -> set[str]:
        intents: set[str] = set()
        for intent, hints in INTENT_HINTS.items():
            if any(any(token.startswith(h) for h in hints) for token in query_tokens):
                intents.add(intent)
        return intents

    def _source_path(self, source: str) -> str:
        if source.startswith("http://") or source.startswith("https://"):
            return urlparse(source).path.lower()
        return source.lower()

    def _generic_source_penalty(self, doc: dict, intents: set[str]) -> float:
        source = str(doc.get("source", "")).lower()
        path = self._source_path(source)
        penalty = 0.0
        if any(pattern in source or pattern in path for pattern in GENERIC_SOURCE_PATTERNS):
            penalty += 0.28
        # Penalize huge generic pages harder for intent-driven queries.
        if intents and ("sveden/common" in source or "prioritet2030/cons/ufs" in source):
            penalty += 0.25
        return penalty

    def _intent_source_bonus(self, doc: dict, intents: set[str]) -> float:
        if not intents:
            return 0.0
        source = str(doc.get("source", "")).lower()
        path = self._source_path(source)
        bonus = 0.0
        for intent in intents:
            hints = INTENT_SOURCE_HINTS.get(intent, ())
            if any(h in source or h in path for h in hints):
                bonus += 0.24
        return min(bonus, 0.45)

    def _intent_content_penalty(
        self, text_tokens: set[str], query_tokens: set[str], intents: set[str]
    ) -> float:
        if not intents:
            return 0.0
        intent_tokens = {t for t in query_tokens if t not in GENERIC_BRAND_TOKENS}
        if not intent_tokens:
            return 0.0
        # If document only matches brand/university tokens but not intent tokens,
        # strongly down-rank it for intent queries.
        if not (intent_tokens & text_tokens):
            return 8.0
        return 0.0

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        if not self._loaded:
            self.load()

        q_tokens = tokenize(query)
        q_tokens = expand_query(q_tokens)
        if not q_tokens:
            return []

        q_tf = Counter(q_tokens)
        scores: list[tuple[float, int]] = []
        q_unique = set(q_tf.keys())
        intents = self._detect_intents(q_unique)
        k1 = 1.2
        b = 0.75

        for i, doc_tf in enumerate(self.doc_tfs):
            score = 0.0
            for token in q_unique:
                tf = doc_tf.get(token, 0)
                if tf == 0:
                    continue
                df = self.doc_freqs.get(token, 1)
                idf = math.log((len(self.docs) - df + 0.5) / (df + 0.5) + 1.0)
                doc_len = max(self.doc_lens[i], 1)
                norm = k1 * (1.0 - b + b * (doc_len / max(self.avg_doc_len, 1.0)))
                score += idf * ((tf * (k1 + 1.0)) / (tf + norm))

            if score > 0:
                doc = self.docs[i]
                text_tokens = set(tokenize(str(doc.get("text", ""))))
                coverage = self._keyword_coverage(q_unique, text_tokens)
                source_bonus = self._source_bonus(doc, q_unique)
                intent_bonus = self._intent_source_bonus(doc, intents)
                penalty = self._generic_source_penalty(doc, intents)
                content_penalty = self._intent_content_penalty(text_tokens, q_unique, intents)
                score = (
                    score * (1.0 + 0.35 * coverage)
                    + source_bonus
                    + intent_bonus
                    - penalty
                    - content_penalty
                )
                scores.append((score, i))

        scores.sort(key=lambda x: x[0], reverse=True)
        result: list[dict] = []
        seen_sources: set[str] = set()
        candidate_limit = min(len(scores), max(top_k * 12, 20))

        for score, idx in scores[:candidate_limit]:
            doc = self.docs[idx]
            source = str(doc["source"])
            if source in seen_sources:
                continue
            seen_sources.add(source)
            result.append(
                {
                    "score": round(score, 6),
                    "id": doc["id"],
                    "source": doc["source"],
                    "source_type": doc["source_type"],
                    "title": doc["title"],
                    "text": doc["text"],
                }
            )
            if len(result) >= top_k:
                break

        return result
