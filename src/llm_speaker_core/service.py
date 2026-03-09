from __future__ import annotations

import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass

from llm_speaker_core.llm import OllamaClient
from llm_speaker_core.rag import LexicalRAG
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


@dataclass
class GenerationResult:
    display_text: str
    speaker_text: str
    used_rag: bool
    fallback_used: bool
    limits_applied: bool
    rag_hits: int
    latency_ms: int


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
        self, rag: LexicalRAG, llm: OllamaClient, speaker_mode: str | None = None
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
            "Если в предоставленном контексте нет точного факта, прямо скажи: "
            "'В текущем контексте нет подтвержденных данных'."
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
        text = re.sub(r"\bСанкт[\s-]?Петербург", "Санкт-Петербург", text, flags=re.IGNORECASE)
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

    def _compose_user_prompt(self, session_id: str, text: str, rag_chunks: list[dict]) -> str:
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
        best_score = -10**9
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

    def handle_query(self, text: str, session_id: str, history: list | None = None) -> GenerationResult:
        started = time.perf_counter()
        fallback_used = False
        limits_applied = False

        external_history = self._normalize_external_history(history)
        if external_history:
            self.memory.set_external(session_id, external_history)

        rag_chunks = self.rag.search(text, top_k=SETTINGS.rag_top_k)
        rag_hits = len(rag_chunks)
        used_rag = rag_hits > 0

        display_raw = self.llm.generate(
            system_prompt=self._build_display_system_prompt(),
            user_prompt=self._compose_user_prompt(session_id, text, rag_chunks),
            max_tokens=110,
        )
        display_text = self._sanitize_markdown_lite(display_raw)
        display_text = self._dedupe_sentences(display_text)
        display_text, changed_identity = self._enforce_university_identity(display_text)
        limits_applied = limits_applied or changed_identity
        display_text, limited = self._limit_words(display_text, SETTINGS.max_display_words)
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
            speaker_text, changed_speaker_identity = self._enforce_university_identity(
                speaker_text
            )
            limits_applied = limits_applied or changed_speaker_identity
            speaker_text = self._force_single_sentence(speaker_text)
            speaker_text, speaker_limited = self._limit_words(
                speaker_text, SETTINGS.max_speaker_words
            )
            limits_applied = limits_applied or speaker_limited

        if not display_text:
            fallback_used = True
            display_text = SETTINGS.fallback_text

        self.memory.add_turn(session_id=session_id, user_text=text, assistant_text=display_text)

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
            latency_ms=elapsed_ms,
        )
    _GUAP_CANONICAL = (
        "ГУАП — Санкт-Петербургский государственный университет аэрокосмического приборостроения."
    )
