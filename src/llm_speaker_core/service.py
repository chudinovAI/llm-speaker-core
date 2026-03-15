from __future__ import annotations

import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Protocol

from llm_speaker_core.rag import LexicalRAG, expand_query, tokenize
from llm_speaker_core.settings import SETTINGS

logger = logging.getLogger(__name__)
SPEAKER_STOPWORDS = {
    "что",
    "как",
    "где",
    "когда",
    "про",
    "это",
    "есть",
    "для",
    "или",
    "ли",
}
FACT_TOKENS_BY_INTENT = {
    "tuition": {"стоимость", "цена", "оплата", "договор", "руб", "обучение", "платное"},
    "contacts": {"контакты", "телефон", "почта", "email", "комиссия", "приемная"},
    "location": {"адрес", "корпус", "кампус"},
    "admission": {"баллы", "абитуриент", "документы", "направления", "поступление"},
}


@dataclass(frozen=True)
class IntentRule:
    fact_tokens: set[str] = field(default_factory=set)
    trusted_source_hints: tuple[str, ...] = ()
    allowed_source_hints: tuple[str, ...] = ()
    denied_source_hints: tuple[str, ...] = ()
    noisy_sentence_markers: tuple[str, ...] = ()
    query_boost_terms: tuple[str, ...] = ()
    display_summary: str = ""
    speaker_summary: str = ""
    min_evidence_for_freeform: float = 0.35
    min_fact_token_hits: int = 2


INTENT_RULES: dict[str, IntentRule] = {
    "tuition": IntentRule(
        fact_tokens={
            "стоимость",
            "цена",
            "оплата",
            "договор",
            "руб",
            "обучение",
            "платное",
        },
        trusted_source_hints=(
            "/eif/pay",
            "/eif/inf_dog",
            "/eif/price",
            "/eif/pol_usl",
            "/sveden/paid_edu",
            "/priem",
            "/abitur",
        ),
        allowed_source_hints=(
            "/eif/pay",
            "/eif/inf_dog",
            "/eif/price",
            "/eif/pol_usl",
            "/sveden/paid_edu",
            "/priem",
            "/abitur",
        ),
        denied_source_hints=("/sveden/inter",),
        noisy_sentence_markers=(
            "кнр",
            "сиань",
            "обучение заграницей",
            "договор о сотрудничестве",
            "мониторинг до",
        ),
        query_boost_terms=("стоимость", "оплата", "договор", "платное образование"),
        display_summary=(
            "**Стоимость обучения в ГУАП зависит от программы и формы обучения.**\n"
            "Актуальные цены и условия оплаты смотрите в разделе платных образовательных услуг."
        ),
        speaker_summary="Стоимость зависит от программы и формы; актуально в разделе платных услуг ГУАП.",
        min_evidence_for_freeform=0.3,
        min_fact_token_hits=2,
    ),
    "admission": IntentRule(
        fact_tokens={"баллы", "абитуриент", "документы", "направления", "поступление"},
        trusted_source_hints=("/abitur", "/priem", "/admission", "/sveden"),
        query_boost_terms=("приемная комиссия", "документы", "сроки приема"),
        display_summary=(
            "Условия поступления зависят от программы и уровня обучения. "
            "Проверьте актуальные правила и сроки в разделе для абитуриентов."
        ),
        speaker_summary="Проверьте актуальные правила и сроки в разделе для абитуриентов ГУАП.",
        min_evidence_for_freeform=0.32,
        min_fact_token_hits=2,
    ),
    "contacts": IntentRule(
        fact_tokens={"контакты", "телефон", "почта", "email", "комиссия", "приемная"},
        trusted_source_hints=("/sveden/common", "/contacts", "/contact", "/abitur"),
        query_boost_terms=("контакты", "телефон", "адрес"),
        display_summary="Актуальные контакты лучше смотреть в официальном разделе сведений и контактов ГУАП.",
        speaker_summary="Актуальные контакты смотрите в официальном разделе контактов ГУАП.",
    ),
    "location": IntentRule(
        fact_tokens={"адрес", "корпус", "кампус"},
        trusted_source_hints=("/sveden/common", "/contacts", "/address", "/objects"),
        query_boost_terms=("адрес", "корпус", "кампус"),
        display_summary="Адреса и корпуса университета доступны в официальном разделе сведений ГУАП.",
        speaker_summary="Актуальные адреса и корпуса смотрите в официальных сведениях ГУАП.",
    ),
    "student_life": IntentRule(
        fact_tokens={"студенты", "клуб", "театр", "спорт", "активности"},
        trusted_source_hints=("/studlife", "/students", "/clubs", "/sport"),
        denied_source_hints=("/it/service/",),
        query_boost_terms=("студенческие активности", "клубы", "спорт", "театр"),
        display_summary="Студенческие активности включают клубы, спорт и творческие объединения ГУАП.",
        speaker_summary="Студенческие активности ГУАП включают клубы, спорт и творческие объединения.",
    ),
}


@dataclass
class GenerationResult:
    display_text: str
    speaker_text: str
    used_rag: bool
    fallback_used: bool
    limits_applied: bool
    rag_hits: int
    rag_sources: list[str]
    intent: str
    evidence_coverage: float
    answer_mode: str
    latency_ms: int


class LLMProtocol(Protocol):
    def generate(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 180
    ) -> str: ...


class SessionMemory:
    def __init__(self, max_turns: int = 5) -> None:
        self.max_turns = max_turns
        self._store: dict[str, list[dict[str, str]]] = defaultdict(list)

    def add_turn(self, session_id: str, user_text: str, assistant_text: str) -> None:
        history = self._store[session_id]
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": assistant_text})
        max_messages = self.max_turns * 2
        if len(history) > max_messages:
            self._store[session_id] = history[-max_messages:]

    def get(self, session_id: str) -> list[dict[str, str]]:
        return list(self._store.get(session_id, []))

    def set_external(self, session_id: str, history: list[dict[str, str]]) -> None:
        max_messages = self.max_turns * 2
        self._store[session_id] = history[-max_messages:]


class Metrics:
    def __init__(self) -> None:
        self.requests = 0
        self.fallbacks = 0
        self.empty_speaker = 0
        self.rag_hits = 0

    def snapshot(self) -> dict[str, int]:
        return {
            "requests": self.requests,
            "fallbacks": self.fallbacks,
            "empty_speaker": self.empty_speaker,
            "rag_hits": self.rag_hits,
        }


class LLMService:
    def __init__(
        self, rag: LexicalRAG, llm: LLMProtocol, speaker_mode: str | None = None
    ) -> None:
        self.rag = rag
        self.llm = llm
        self.memory = SessionMemory(max_turns=SETTINGS.max_memory_turns)
        self.metrics = Metrics()
        mode = (speaker_mode or SETTINGS.speaker_mode).lower()
        self.speaker_mode = mode if mode in {"local", "llm"} else "local"

    def _build_display_system_prompt(self) -> str:
        return (
            "Ты голосовой ассистент СПбГУАП. "
            "Отвечай строго на русском языке. "
            "Критично важно: ГУАП расшифровывается только как "
            "'Санкт-Петербургский государственный университет аэрокосмического приборостроения'. "
            "Запрещено путать ГУАП с другими вузами и давать ложные расшифровки. "
            "Формат для дисплея: разрешены только короткие абзацы, переносы строк, "
            "короткие списки и **жирный**. Без HTML и без сложного markdown. "
            "Избегай токсичности, оскорблений и негативных необоснованных суждений о ГУАП. "
            "Если вопрос провокационный, отвечай нейтрально и конструктивно. "
            "Ответ должен быть фактологичным и полезным. "
            "Запрещено придумывать даты, названия, статусы и исторические факты. "
            "Если в доступном фрагменте контекста нет точного факта, прямо скажи: "
            "'В доступном фрагменте контекста нет точных подтвержденных данных'."
        )

    def _build_speaker_system_prompt(self) -> str:
        return (
            "Сожми текст ответа до одного короткого предложения для озвучки. "
            "Только русский язык, без markdown, без списков. "
            "Оставь только ключевую информацию. "
            "Запрещено давать неверную расшифровку ГУАП."
        )

    def _sanitize_markdown_lite(self, text: str) -> str:
        text = re.sub(r"[`#>]+", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"\*\*\s+", "**", text)
        return text.strip()

    def _to_plain_text(self, text: str) -> str:
        text = re.sub(r"[*_`#>-]", "", text)
        text = re.sub(
            r"\bСанкт[\s-]?Петербург", "Санкт-Петербург", text, flags=re.IGNORECASE
        )
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _dedupe_sentences(self, text: str) -> str:
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
        seen: set[str] = set()
        unique: list[str] = []
        for part in parts:
            key = re.sub(r"\s+", " ", part.lower())
            if key in seen:
                continue
            seen.add(key)
            unique.append(part)
        return " ".join(unique).strip() if unique else text.strip()

    def _remove_policy_artifacts(self, text: str) -> tuple[str, bool]:
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
        filtered: list[str] = []
        changed = False
        banned_patterns = (
            r"^важно[:\s]",
            r"запрещено",
            r"не может быть другой",
            r"в других вузах используется похожая аббревиатура",
        )
        for part in parts:
            low = part.lower()
            if any(re.search(pattern, low) for pattern in banned_patterns):
                changed = True
                continue
            filtered.append(part)

        if not filtered:
            return text, changed
        return " ".join(filtered).strip(), changed

    def _force_single_sentence(self, text: str) -> str:
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
        if not parts:
            return text.strip()
        first = parts[0].rstrip(".!?")
        return f"{first}."

    def _finalize_display_tail(self, text: str) -> tuple[str, bool]:
        text = text.strip()
        if not text:
            return text, False

        if re.search(r"[.!?]$", text):
            return text, False

        # Prefer cutting at the last full sentence if enough content remains.
        last_punct = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
        if last_punct >= 0 and last_punct >= int(len(text) * 0.55):
            return text[: last_punct + 1].strip(), True

        # Otherwise drop the trailing token to avoid hanging partial words.
        words = text.split()
        if len(words) > 1:
            trimmed = " ".join(words[:-1]).rstrip(",;:-")
            if trimmed:
                return f"{trimmed}.", True

        return f"{text}.", True

    def _enforce_university_identity(self, text: str) -> tuple[str, bool]:
        lower = text.lower()
        wrong_markers = (
            "архитектурно-строительный",
            "архитектурностроительный",
            "строительный университет",
        )
        changed = False

        if any(marker in lower for marker in wrong_markers):
            cleaned = re.sub(
                r"[^.?!]*архитектурно[- ]?строительн[^.?!]*[.?!]?\s*",
                "",
                text,
                flags=re.IGNORECASE,
            ).strip()
            text = f"{self._GUAP_CANONICAL} {cleaned}".strip()
            changed = True

        if "гуап" in lower and "аэрокосмического приборостроения" not in lower:
            text = f"{self._GUAP_CANONICAL} {text}".strip()
            changed = True

        return text, changed

    def _limit_words(self, text: str, max_words: int) -> tuple[str, bool]:
        words = text.split()
        if len(words) <= max_words:
            return text, False
        return " ".join(words[:max_words]).strip(), True

    def _compose_user_prompt(
        self, session_id: str, text: str, rag_chunks: list[dict]
    ) -> str:
        history = self.memory.get(session_id)
        history_str = "\n".join(
            f"{item['role']}: {item['content']}" for item in history[-6:]
        )
        rag_context = "\n\n".join(chunk["text"] for chunk in rag_chunks)
        return (
            f"История диалога:\n{history_str or 'история пуста'}\n\n"
            f"Контекст RAG:\n{rag_context or 'контекст не найден'}\n\n"
            f"Запрос пользователя:\n{text}"
        )

    def _normalize_external_history(self, history: list | None) -> list[dict[str, str]]:
        if not history:
            return []
        normalized: list[dict[str, str]] = []
        for item in history:
            if isinstance(item, str):
                normalized.append({"role": "user", "content": item})
                continue
            if isinstance(item, dict):
                role = str(item.get("role", "user"))
                content = str(item.get("content", "")).strip()
                if content:
                    normalized.append({"role": role, "content": content})
        return normalized

    def _primary_intent(self, text: str) -> str:
        intents = self.rag.detect_intents(text)
        if not intents:
            return "general"
        return intents[0]

    def _intent_rule(self, intent: str) -> IntentRule:
        return INTENT_RULES.get(intent, IntentRule())

    def _matches_any_hint(self, source: str, hints: tuple[str, ...]) -> bool:
        if not hints:
            return False
        source_low = source.lower()
        return any(h in source_low for h in hints)

    def _estimate_evidence_coverage(self, text: str, rag_chunks: list[dict]) -> float:
        intent = self._primary_intent(text)
        query_tokens = set(expand_query(tokenize(text)))
        fact_tokens = self._intent_rule(
            intent
        ).fact_tokens or FACT_TOKENS_BY_INTENT.get(intent, set())
        if fact_tokens:
            prioritized = {t for t in query_tokens if t in fact_tokens}
            if prioritized:
                query_tokens = prioritized
        if not query_tokens:
            return 0.0
        ctx_tokens: set[str] = set()
        for chunk in rag_chunks:
            ctx_tokens.update(tokenize(str(chunk.get("text", ""))))
        coverage = len(query_tokens & ctx_tokens) / max(len(query_tokens), 1)
        return round(coverage, 4)

    def _verify_grounding(
        self, answer: str, rag_chunks: list[dict]
    ) -> tuple[str, bool]:
        """Drop unsupported claims to reduce hallucinations for broad questions."""
        ctx_tokens: set[str] = set()
        for chunk in rag_chunks:
            ctx_tokens.update(tokenize(str(chunk.get("text", ""))))

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
        if not sentences:
            return answer, False

        kept: list[str] = []
        changed = False
        for sent in sentences:
            if re.fullmatch(r"\s*[\d]+[.)]?\s*", sent) or re.fullmatch(
                r"\s*[-*•]+\s*", sent
            ):
                changed = True
                continue
            sent_tokens = set(tokenize(sent))
            # Keep short connective sentences.
            if len(sent_tokens) <= 3:
                kept.append(sent)
                continue
            overlap = len(sent_tokens & ctx_tokens) / max(len(sent_tokens), 1)
            if overlap >= 0.28:
                kept.append(sent)
            else:
                changed = True

        if not kept:
            return (
                "В доступном фрагменте контекста нет точных подтвержденных данных.",
                True,
            )
        return " ".join(kept).strip(), changed

    def _is_low_info_display(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return True
        if re.fullmatch(r"\s*[\d]+[.)]?\s*", stripped):
            return True
        if re.fullmatch(r"\s*[-*•]+\s*", stripped):
            return True
        tokens = tokenize(stripped)
        return len(tokens) < 4

    def _extractive_grounded_fallback(
        self, text: str, intent: str, rag_chunks: list[dict]
    ) -> str:
        if not rag_chunks:
            return ""

        rule = self._intent_rule(intent)
        query_tokens = set(expand_query(tokenize(text)))
        fact_tokens = rule.fact_tokens or FACT_TOKENS_BY_INTENT.get(intent, set())
        if fact_tokens:
            focused = {t for t in query_tokens if t in fact_tokens}
            if focused:
                query_tokens = focused
        if not query_tokens:
            query_tokens = set(tokenize(text))

        candidates: list[tuple[float, str]] = []
        for chunk in rag_chunks:
            source = str(chunk.get("source", "")).lower()
            if rule.allowed_source_hints and not self._matches_any_hint(
                source, rule.allowed_source_hints
            ):
                continue
            if rule.denied_source_hints and self._matches_any_hint(
                source, rule.denied_source_hints
            ):
                continue

            chunk_text = str(chunk.get("text", "")).strip()
            if not chunk_text:
                continue
            sentences = [
                s.strip()
                for s in re.split(r"(?<=[.!?])\s+|\n+", chunk_text)
                if s.strip() and len(s.split()) >= 4
            ]
            for sent in sentences:
                sent_low = sent.lower()
                if rule.noisy_sentence_markers and any(
                    marker in sent_low for marker in rule.noisy_sentence_markers
                ):
                    continue
                if (
                    len(sent.split()) > 32
                    and fact_tokens
                    and not (set(tokenize(sent)) & fact_tokens)
                ):
                    continue

                sent_tokens = set(tokenize(sent))
                if not sent_tokens:
                    continue
                overlap = len(query_tokens & sent_tokens) / max(len(query_tokens), 1)
                fact_overlap = (
                    len(fact_tokens & sent_tokens) / max(len(fact_tokens), 1)
                    if fact_tokens
                    else 0.0
                )
                if overlap == 0 and fact_overlap == 0:
                    continue
                source_bonus = 0.0
                if rule.trusted_source_hints and self._matches_any_hint(
                    source, rule.trusted_source_hints
                ):
                    source_bonus = 0.18
                score = overlap + 0.6 * fact_overlap + source_bonus
                candidates.append((score, sent))

        if not candidates:
            return ""

        candidates.sort(key=lambda x: x[0], reverse=True)
        chosen: list[str] = []
        seen: set[str] = set()
        for _, sent in candidates:
            key = re.sub(r"\s+", " ", sent.lower())
            if key in seen:
                continue
            seen.add(key)
            chosen.append(sent)
            if len(chosen) >= 2:
                break

        if not chosen:
            return ""
        return " ".join(chosen).strip()

    def _has_high_trust_source(self, intent: str, rag_chunks: list[dict]) -> bool:
        hints = self._intent_rule(intent).trusted_source_hints
        if not hints:
            return False
        for chunk in rag_chunks:
            source = str(chunk.get("source", "")).lower()
            if any(h in source for h in hints):
                return True
        return False

    def _apply_intent_speaker_policy(
        self, intent: str, speaker_text: str, rag_sources: list[str], display_text: str
    ) -> str:
        rule = self._intent_rule(intent)
        if not rule.speaker_summary:
            return speaker_text

        has_trusted_source = any(
            self._matches_any_hint(src.lower(), rule.trusted_source_hints)
            for src in rag_sources
        )

        has_specific_price = bool(
            re.search(r"\b\d[\d\s]{2,}\s*(руб|₽)\b", display_text.lower())
        )
        if has_specific_price and intent == "tuition":
            return speaker_text

        if has_trusted_source or self._is_low_info_display(speaker_text):
            return rule.speaker_summary
        return speaker_text

    def _apply_intent_display_policy(
        self,
        intent: str,
        display_text: str,
        rag_sources: list[str],
        evidence_coverage: float,
    ) -> str:
        rule = self._intent_rule(intent)
        if not rule.display_summary:
            return display_text

        low = display_text.lower()
        fact_token_hits = 0
        if rule.fact_tokens:
            display_tokens = set(tokenize(display_text))
            fact_token_hits = len(display_tokens & rule.fact_tokens)
        has_specific_number = bool(
            re.search(r"\b\d[\d\s]{2,}\s*(руб|₽|год|г\.)\b", low)
        )
        if has_specific_number and intent in {"tuition", "admission"}:
            return display_text

        # Detect menu-like extraction: many short title fragments and weak punctuation.
        tokens = tokenize(display_text)
        punctuation = sum(display_text.count(ch) for ch in ".!?")
        menu_like = (
            len(tokens) > 20
            and punctuation <= 1
            and any(
                k in low
                for k in ("положение", "формы договоров", "комиссия", "структура")
            )
        )
        quality_gate_failed = (
            evidence_coverage < rule.min_evidence_for_freeform
            and fact_token_hits < rule.min_fact_token_hits
        )

        if (
            not menu_like
            and len(tokens) <= 36
            and not self._is_low_info_display(display_text)
            and not quality_gate_failed
        ):
            return display_text

        has_trusted_source = any(
            self._matches_any_hint(src.lower(), rule.trusted_source_hints)
            for src in rag_sources
        )
        if (
            has_trusted_source
            or menu_like
            or self._is_low_info_display(display_text)
            or quality_gate_failed
        ):
            return rule.display_summary
        return display_text

    def _filter_rag_chunks_by_intent(
        self, intent: str, rag_chunks: list[dict]
    ) -> list[dict]:
        rule = self._intent_rule(intent)
        if not rule.allowed_source_hints and not rule.denied_source_hints:
            return rag_chunks

        filtered: list[dict] = []
        for chunk in rag_chunks:
            source = str(chunk.get("source", "")).lower()
            if rule.denied_source_hints and self._matches_any_hint(
                source, rule.denied_source_hints
            ):
                continue
            if rule.allowed_source_hints and not self._matches_any_hint(
                source, rule.allowed_source_hints
            ):
                continue
            filtered.append(chunk)
        return filtered or rag_chunks

    def _extract_speaker_local(self, display_text: str, user_text: str) -> str:
        plain = self._to_plain_text(display_text)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", plain) if s.strip()]
        if not sentences:
            return plain

        query_tokens = {
            token.lower()
            for token in re.findall(r"[a-zA-Zа-яА-Я0-9_]+", user_text)
            if len(token) > 2 and token.lower() not in SPEAKER_STOPWORDS
        }

        best_sentence = sentences[0]
        best_score = float(-(10**9))
        for sentence in sentences:
            sent_tokens = {
                token.lower()
                for token in re.findall(r"[a-zA-Zа-яА-Я0-9_]+", sentence)
                if len(token) > 2
            }
            overlap = len(query_tokens & sent_tokens) if query_tokens else 0
            length_penalty = abs(len(sentence.split()) - 10) * 0.05
            guap_bonus = 0.35 if "гуап" in sent_tokens else 0.0
            score = overlap + guap_bonus - length_penalty
            if score > best_score:
                best_score = score
                best_sentence = sentence

        return best_sentence

    def _is_fact_sensitive_query(self, text: str) -> bool:
        low = text.lower()
        base_markers = {"адрес", "контакт", "связ", "дата", "когда", "основан", "где"}
        dynamic_markers: set[str] = set(base_markers)
        for rule in INTENT_RULES.values():
            for token in rule.fact_tokens:
                if len(token) >= 4:
                    dynamic_markers.add(token[:5])
        return any(marker in low for marker in dynamic_markers)

    def _augment_rag_context(self, text: str, rag_chunks: list[dict]) -> list[dict]:
        if not self._is_fact_sensitive_query(text):
            return rag_chunks

        intent = self._primary_intent(text)
        rule = self._intent_rule(intent)
        if not rule.query_boost_terms:
            return rag_chunks

        fallback_query = f"{text} {' '.join(rule.query_boost_terms)}"
        extra_chunks = self.rag.search(
            fallback_query, top_k=max(SETTINGS.rag_top_k + 2, 5)
        )
        for chunk in extra_chunks:
            source = str(chunk.get("source", "")).lower()
            if rule.trusted_source_hints and not self._matches_any_hint(
                source, rule.trusted_source_hints
            ):
                continue
            if rule.denied_source_hints and self._matches_any_hint(
                source, rule.denied_source_hints
            ):
                continue
            existing_ids = {str(c.get("id")) for c in rag_chunks}
            if str(chunk.get("id")) not in existing_ids:
                rag_chunks = [*rag_chunks, chunk]
            break
        return rag_chunks

    def handle_query(
        self, text: str, session_id: str, history: list | None = None
    ) -> GenerationResult:
        started = time.perf_counter()
        fallback_used = False
        limits_applied = False

        external_history = self._normalize_external_history(history)
        if external_history:
            self.memory.set_external(session_id, external_history)

        intent = self._primary_intent(text)
        rag_chunks = self.rag.search(text, top_k=SETTINGS.rag_top_k)
        rag_chunks = self._augment_rag_context(text, rag_chunks)
        rag_chunks = self._filter_rag_chunks_by_intent(intent, rag_chunks)
        rag_hits = len(rag_chunks)
        used_rag = rag_hits > 0
        rag_sources = [str(c.get("source", "")) for c in rag_chunks]
        evidence_coverage = self._estimate_evidence_coverage(text, rag_chunks)

        display_raw = self.llm.generate(
            system_prompt=self._build_display_system_prompt(),
            user_prompt=self._compose_user_prompt(session_id, text, rag_chunks),
            max_tokens=110,
        )
        display_text = self._sanitize_markdown_lite(display_raw)
        display_text = self._dedupe_sentences(display_text)
        display_text, policy_removed = self._remove_policy_artifacts(display_text)
        limits_applied = limits_applied or policy_removed
        display_text, grounding_changed = self._verify_grounding(
            display_text, rag_chunks
        )
        limits_applied = limits_applied or grounding_changed
        display_text, changed_identity = self._enforce_university_identity(display_text)
        limits_applied = limits_applied or changed_identity
        display_text, limited = self._limit_words(
            display_text, SETTINGS.max_display_words
        )
        limits_applied = limits_applied or limited
        display_text, tail_fixed = self._finalize_display_tail(display_text)
        limits_applied = limits_applied or tail_fixed
        should_force_extractive = (
            "нет точных подтвержденных данных" in display_text.lower()
            or self._is_low_info_display(display_text)
        )
        if (
            should_force_extractive
            and rag_hits > 0
            and (
                evidence_coverage >= 0.3
                or self._has_high_trust_source(intent, rag_chunks)
            )
        ):
            extractive = self._extractive_grounded_fallback(text, intent, rag_chunks)
            if extractive:
                display_text = self._sanitize_markdown_lite(extractive)
                display_text = self._dedupe_sentences(display_text)
                display_text, changed_identity = self._enforce_university_identity(
                    display_text
                )
                limits_applied = limits_applied or changed_identity
                display_text, limited = self._limit_words(
                    display_text, SETTINGS.max_display_words
                )
                limits_applied = limits_applied or limited
                display_text, tail_fixed = self._finalize_display_tail(display_text)
                limits_applied = limits_applied or tail_fixed
        display_text = self._apply_intent_display_policy(
            intent=intent,
            display_text=display_text,
            rag_sources=rag_sources,
            evidence_coverage=evidence_coverage,
        )
        display_text, limited = self._limit_words(
            display_text, SETTINGS.max_display_words
        )
        limits_applied = limits_applied or limited
        display_text, tail_fixed = self._finalize_display_tail(display_text)
        limits_applied = limits_applied or tail_fixed

        speaker_text = ""
        if self.speaker_mode == "local":
            speaker_text = self._extract_speaker_local(display_text, text)
        else:
            try:
                speaker_raw = self.llm.generate(
                    system_prompt=self._build_speaker_system_prompt(),
                    user_prompt=display_text,
                    max_tokens=28,
                )
                speaker_text = speaker_raw
            except Exception as exc:  # noqa: BLE001
                logger.warning("speaker generation failed: %s", exc)
                fallback_used = True
                speaker_text = ""

        if speaker_text:
            speaker_text = self._to_plain_text(speaker_text)
            speaker_text = self._dedupe_sentences(speaker_text)
            speaker_text, speaker_policy_removed = self._remove_policy_artifacts(
                speaker_text
            )
            limits_applied = limits_applied or speaker_policy_removed
            speaker_text, changed_speaker_identity = self._enforce_university_identity(
                speaker_text
            )
            limits_applied = limits_applied or changed_speaker_identity
            speaker_text = self._force_single_sentence(speaker_text)
            speaker_text = self._apply_intent_speaker_policy(
                intent=intent,
                speaker_text=speaker_text,
                rag_sources=rag_sources,
                display_text=display_text,
            )
            speaker_text, speaker_limited = self._limit_words(
                speaker_text, SETTINGS.max_speaker_words
            )
            limits_applied = limits_applied or speaker_limited

        if not display_text:
            fallback_used = True
            display_text = SETTINGS.fallback_text

        answer_mode = "grounded"
        if "нет точных подтвержденных данных" in display_text.lower():
            answer_mode = "uncertain"

        self.memory.add_turn(
            session_id=session_id, user_text=text, assistant_text=display_text
        )

        elapsed_ms = int((time.perf_counter() - started) * 1000)

        self.metrics.requests += 1
        self.metrics.rag_hits += rag_hits
        if fallback_used:
            self.metrics.fallbacks += 1
        if not speaker_text:
            self.metrics.empty_speaker += 1

        return GenerationResult(
            display_text=display_text,
            speaker_text=speaker_text,
            used_rag=used_rag,
            fallback_used=fallback_used,
            limits_applied=limits_applied,
            rag_hits=rag_hits,
            rag_sources=rag_sources,
            intent=intent,
            evidence_coverage=evidence_coverage,
            answer_mode=answer_mode,
            latency_ms=elapsed_ms,
        )

    _GUAP_CANONICAL = "ГУАП — Санкт-Петербургский государственный университет аэрокосмического приборостроения."
