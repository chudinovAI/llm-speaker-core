from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
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
    "направлен": ("специальности", "программы", "бакалавриат", "магистратура"),
    "специальн": ("направления", "программы", "бакалавриат", "магистратура"),
    "приемн": ("приемная", "приёмная", "комиссия", "контакты", "поступление"),
    "общежит": ("проживание", "заселение", "кампус"),
    "стипенд": ("выплаты", "поддержка", "социальная"),
    "стои": ("стоимость", "цена", "оплата", "платное", "договор"),
    "скольк": ("стоимость", "цена", "оплата"),
    "стоим": ("цена", "обучение", "оплата", "платное", "договор"),
    "оплат": ("стоимость", "цена", "обучение", "договор"),
    "цена": ("стоимость", "оплата", "обучение", "договор"),
    "контакт": ("приемная", "комиссия", "телефон", "почта", "email"),
    "связ": ("контакты", "телефон", "почта", "email"),
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
QUESTION_TOKENS = {
    "как",
    "где",
    "когда",
    "кто",
    "что",
    "сколько",
    "какой",
    "какая",
    "какие",
    "каким",
    "какую",
    "почему",
    "зачем",
    "ли",
    "стоит",
}
INTENT_HINTS: dict[str, tuple[str, ...]] = {
    "admission": (
        "поступ",
        "абитуриент",
        "прием",
        "приём",
        "баллы",
        "комисси",
        "направлен",
        "специальн",
        "программ",
        "бакалавр",
        "магистр",
    ),
    "contacts": ("контакт", "связ", "телефон", "почт", "email"),
    "location": ("где", "адрес", "корпус", "кампус", "наход"),
    "student_life": ("студен", "актив", "круж", "клуб", "театр", "спорт"),
    "tuition": (
        "стоим",
        "стои",
        "оплат",
        "цена",
        "обучен",
        "платн",
        "договор",
        "контракт",
    ),
}
INTENT_SOURCE_HINTS: dict[str, tuple[str, ...]] = {
    "admission": ("/priem", "/abitur", "комис", "admission"),
    "contacts": ("/contacts", "/contact", "/sveden/common", "/priem", "address"),
    "location": ("/contacts", "/sveden/common", "address"),
    "student_life": ("/studlife", "/students", "/clubs", "/theatre", "/sport"),
    "tuition": (
        "/priem",
        "/abitur",
        "/eif/pay",
        "/eif/inf_dog",
        "stoimost",
        "oplata",
        "dogovor",
        "price",
        "tuition",
    ),
}
LOW_SIGNAL_WEB_SOURCE_PATTERNS = (
    "/messages/",
    "/pubs/",
    "/news/",
    "/calendar",
    "/event",
)
LOW_SIGNAL_DOC_SOURCE_RE = re.compile(
    r"(?:^|_)(sbor\d+(?:_\d+)?|sputnik|program\d+|20\d{2}_\d+)\.pdf$"
)
HIGH_SIGNAL_DOC_SOURCE_HINTS = (
    "pravila",
    "priem",
    "abitur",
    "dogovor",
    "stoim",
    "oplata",
    "polozhen",
)
INTENT_HIGH_TRUST_PATHS: dict[str, tuple[str, ...]] = {
    "admission": ("/priem", "/abitur", "/admission"),
    "contacts": ("/contacts", "/sveden/common", "/priem"),
    "location": ("/contacts", "/sveden/common"),
    "student_life": ("/studlife", "/students"),
    "tuition": (
        "/eif/pay",
        "/eif/inf_dog",
        "/eif/price",
        "/eif/pol_usl",
        "/sveden/paid_edu",
        "/priem",
        "/abitur",
    ),
}
INDEX_BY_INTENT = {
    "admission": "admission",
    "contacts": "contacts",
    "location": "location",
    "student_life": "student_life",
    "tuition": "tuition",
}
FACT_PRIORITY_TOKENS = {
    "tuition": {"стоимость", "цена", "оплата", "договор", "руб", "платное", "обучение"},
    "contacts": {"контакты", "телефон", "почта", "email", "приемная", "комиссия"},
    "location": {"адрес", "кампус", "корпус", "аудитория"},
    "admission": {"поступление", "баллы", "направления", "абитуриент", "документы"},
}


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


@dataclass
class IndexData:
    docs: list[dict]
    doc_freqs: dict[str, int]
    doc_tfs: list[dict[str, int]]
    doc_lens: list[int]
    avg_doc_len: float


class LexicalRAG:
    """BM25-like lexical retriever with intent-aware multi-index routing."""

    def __init__(self, index_path: Path) -> None:
        self.index_path = index_path
        self.indexes: dict[str, IndexData] = {}
        self._loaded = False

    def _load_payload(self, path: Path) -> IndexData:
        payload = json.loads(path.read_text(encoding="utf-8"))
        docs = payload["docs"]
        doc_freqs = {k: int(v) for k, v in payload["doc_freqs"].items()}
        doc_tfs = [{k: int(v) for k, v in tf.items()} for tf in payload["doc_tfs"]]
        doc_lens = [int(x) for x in payload["doc_lens"]]
        avg_doc_len = sum(doc_lens) / max(len(doc_lens), 1)
        return IndexData(
            docs=docs,
            doc_freqs=doc_freqs,
            doc_tfs=doc_tfs,
            doc_lens=doc_lens,
            avg_doc_len=avg_doc_len,
        )

    def _resolve_index_path(self, sub_path: str) -> Path:
        raw_path = Path(sub_path)
        if raw_path.is_absolute() and raw_path.exists():
            return raw_path
        if not raw_path.is_absolute():
            candidate = (self.index_path.parent / raw_path).resolve()
            if candidate.exists():
                return candidate
        fallback = (self.index_path.parent / raw_path.name).resolve()
        if fallback.exists():
            return fallback
        return raw_path

    def load(self) -> None:
        if not self.index_path.exists():
            raise FileNotFoundError(f"RAG index not found: {self.index_path}")

        root_payload = json.loads(self.index_path.read_text(encoding="utf-8"))

        if root_payload.get("format") == "multi_v1":
            index_map = root_payload.get("indexes", {})
            for name, sub_path in index_map.items():
                p = self._resolve_index_path(str(sub_path))
                if p.exists():
                    self.indexes[name] = self._load_payload(p)
            if "general" not in self.indexes:
                raise RuntimeError(
                    "multi-index manifest does not include general index"
                )
        else:
            self.indexes = {"general": self._load_payload(self.index_path)}

        self._loaded = True

    def _keyword_coverage(self, query_tokens: set[str], text_tokens: set[str]) -> float:
        if not query_tokens:
            return 0.0
        return len(query_tokens & text_tokens) / len(query_tokens)

    def _priority_tokens_for_intents(self, intents: set[str]) -> set[str]:
        tokens: set[str] = set()
        for intent in intents:
            tokens.update(FACT_PRIORITY_TOKENS.get(intent, set()))
        return tokens

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

    def detect_intents(self, query: str) -> list[str]:
        q_tokens = set(expand_query(tokenize(query)))
        intents: set[str] = set()
        for intent, hints in INTENT_HINTS.items():
            if any(any(token.startswith(h) for h in hints) for token in q_tokens):
                intents.add(intent)
        return sorted(intents)

    def _source_path(self, source: str) -> str:
        if source.startswith("http://") or source.startswith("https://"):
            return urlparse(source).path.lower()
        return source.lower()

    def _generic_source_penalty(self, doc: dict, intents: set[str]) -> float:
        source = str(doc.get("source", "")).lower()
        path = self._source_path(source)
        penalty = 0.0
        if "/search" in source or "/search" in path:
            penalty += 0.4
        if "prioritet2030/cons/ufs" in source or "prioritet2030/cons/ufs" in path:
            penalty += 0.55 if intents else 0.28
        if "/sveden/common" in source or "/sveden/common" in path:
            if intents and not (intents & {"contacts", "location"}):
                penalty += 0.22
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

    def _source_quality_adjustment(self, doc: dict, intents: set[str]) -> float:
        if not intents:
            return 0.0

        source = str(doc.get("source", "")).lower()
        path = self._source_path(source)
        source_type = str(doc.get("source_type", "")).lower()
        adjustment = 0.0

        for intent in intents:
            trusted = INTENT_HIGH_TRUST_PATHS.get(intent, ())
            if any(p in path for p in trusted):
                adjustment += 0.3

        if source_type == "web":
            if any(p in path for p in LOW_SIGNAL_WEB_SOURCE_PATTERNS):
                adjustment -= 1.4
            if "tuition" in intents and "/sveden/inter" in path:
                adjustment -= 2.2
        elif source_type == "doc":
            if LOW_SIGNAL_DOC_SOURCE_RE.search(path):
                adjustment -= 3.2
            elif any(h in path for h in HIGH_SIGNAL_DOC_SOURCE_HINTS):
                adjustment += 0.18

        if bool(doc.get("is_archived", False)) and intents & {
            "tuition",
            "admission",
            "contacts",
        }:
            adjustment -= 1.6

        return max(min(adjustment, 0.9), -3.5)

    def _intent_content_penalty(
        self, text_tokens: set[str], query_tokens: set[str], intents: set[str]
    ) -> float:
        if not intents:
            return 0.0
        penalty = 0.0

        intent_tokens = {
            t
            for t in query_tokens
            if t not in GENERIC_BRAND_TOKENS and t not in QUESTION_TOKENS
        }
        if intent_tokens and not (intent_tokens & text_tokens):
            penalty += 2.4

        priority_tokens = self._priority_tokens_for_intents(intents)
        if priority_tokens and not (priority_tokens & text_tokens):
            penalty += 2.0

        return penalty

    def _search_index(
        self,
        index: IndexData,
        q_unique: set[str],
        intents: set[str],
        k1: float = 1.2,
        b: float = 0.75,
    ) -> list[tuple[float, dict]]:
        scored_docs: list[tuple[float, dict]] = []
        for i, doc_tf in enumerate(index.doc_tfs):
            score = 0.0
            for token in q_unique:
                tf = doc_tf.get(token, 0)
                if tf == 0:
                    continue
                df = index.doc_freqs.get(token, 1)
                idf = math.log((len(index.docs) - df + 0.5) / (df + 0.5) + 1.0)
                doc_len = max(index.doc_lens[i], 1)
                norm = k1 * (1.0 - b + b * (doc_len / max(index.avg_doc_len, 1.0)))
                score += idf * ((tf * (k1 + 1.0)) / (tf + norm))

            if score <= 0:
                continue

            doc = index.docs[i]
            text_tokens = set(tokenize(str(doc.get("text", ""))))
            coverage = self._keyword_coverage(q_unique, text_tokens)
            source_bonus = self._source_bonus(doc, q_unique)
            intent_bonus = self._intent_source_bonus(doc, intents)
            penalty = self._generic_source_penalty(doc, intents)
            source_quality = self._source_quality_adjustment(doc, intents)
            content_penalty = self._intent_content_penalty(
                text_tokens, q_unique, intents
            )
            score = (
                score * (1.0 + 0.35 * coverage)
                + source_bonus
                + intent_bonus
                + source_quality
                - penalty
                - content_penalty
            )
            scored_docs.append((score, doc))
        return scored_docs

    def _stage2_rerank(
        self,
        query_tokens: set[str],
        intents: set[str],
        candidates: list[tuple[float, dict]],
    ) -> list[tuple[float, dict]]:
        q_tokens = set(query_tokens)
        priority_tokens = self._priority_tokens_for_intents(intents)

        reranked: list[tuple[float, dict]] = []
        for base_score, doc in candidates:
            text = str(doc.get("text", "")).lower()
            title = str(doc.get("title", "")).lower()
            source = str(doc.get("source", "")).lower()
            doc_tokens = set(tokenize(text))

            # Evidence-style boosts.
            q_overlap = len(q_tokens & doc_tokens)
            priority_overlap = (
                len(priority_tokens & doc_tokens) if priority_tokens else 0
            )
            title_match = sum(1 for t in q_tokens if t in title)
            source_match = sum(1 for t in q_tokens if t in source)

            # Penalize too-generic snippets for intent-heavy queries.
            generic_penalty = 0.0
            if intents and len(doc_tokens & q_tokens) <= 1:
                generic_penalty = 0.8

            source_quality = self._source_quality_adjustment(doc, intents)
            rerank_score = (
                base_score
                + 0.22 * q_overlap
                + 0.35 * priority_overlap
                + 0.12 * title_match
                + 0.06 * source_match
                + 0.25 * source_quality
                - generic_penalty
            )
            reranked.append((rerank_score, doc))

        reranked.sort(key=lambda x: x[0], reverse=True)
        return reranked

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        if not self._loaded:
            self.load()

        q_tokens = expand_query(tokenize(query))
        if not q_tokens:
            return []

        intents = set(self.detect_intents(query))
        q_unique = set(Counter(q_tokens).keys())
        q_unique.update(self._priority_tokens_for_intents(intents))
        if intents:
            q_focus = {
                t
                for t in q_unique
                if t not in GENERIC_BRAND_TOKENS and t not in QUESTION_TOKENS
            }
            if not q_focus:
                q_focus = set(q_unique)
        else:
            q_focus = set(q_unique)

        target_indexes: list[str] = []
        for intent in intents:
            idx = INDEX_BY_INTENT.get(intent)
            if idx and idx in self.indexes and idx not in target_indexes:
                target_indexes.append(idx)

        merged: list[tuple[float, dict]] = []
        if target_indexes:
            for idx_name in target_indexes:
                idx_data = self.indexes.get(idx_name)
                if not idx_data:
                    continue
                scored = self._search_index(idx_data, q_focus, intents)
                for score, doc in scored:
                    merged.append((score + 0.32, doc))
            # Intent indexes are primary; use general only if they produced no candidates.
            if not merged:
                general_idx = self.indexes.get("general")
                if general_idx is not None:
                    scored = self._search_index(general_idx, q_focus, intents)
                    for score, doc in scored:
                        merged.append((score - 1.4, doc))
        else:
            general_idx = self.indexes.get("general")
            if general_idx is not None:
                scored = self._search_index(general_idx, q_focus, intents)
                for score, doc in scored:
                    merged.append((score, doc))

        merged.sort(key=lambda x: x[0], reverse=True)
        # Stage-2 rerank: lightweight cross-signal reranker over top candidates.
        rerank_pool = merged[: min(len(merged), max(top_k * 24, 48))]
        merged = (
            self._stage2_rerank(q_focus, intents, rerank_pool)
            + merged[min(len(merged), max(top_k * 24, 48)) :]
        )

        result: list[dict] = []
        seen_sources: set[str] = set()
        candidate_limit = min(len(merged), max(top_k * 16, 32))

        for score, doc in merged[:candidate_limit]:
            source = str(doc.get("source", ""))
            if source in seen_sources:
                continue
            seen_sources.add(source)
            result.append(
                {
                    "score": round(score, 6),
                    "id": doc.get("id"),
                    "source": source,
                    "source_type": doc.get("source_type"),
                    "title": doc.get("title"),
                    "text": doc.get("text"),
                }
            )
            if len(result) >= top_k:
                break

        return result
