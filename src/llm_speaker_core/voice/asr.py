"""
ASR pipeline based on GigaAM v3 streaming.

CLI examples:
  uv run llm-asr --list-devices
  uv run llm-asr --device 1
  uv run llm-asr --wake-word "привет гуап" --no-speaker-verify
"""

import argparse
import queue
import re
import sys
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch

# - webrtcvad: ставим если есть, иначе мокаем для resemblyzer -
try:
    import webrtcvad  # noqa: F401
except ImportError:
    import types as _types

    _mock = _types.ModuleType("webrtcvad")

    class _FakeVad:
        def __init__(self, *a, **kw):
            pass

        def set_mode(self, *a):
            pass

        def is_speech(self, *a, **kw):
            return False

    _mock.Vad = _FakeVad  # type: ignore[attr-defined]
    sys.modules["webrtcvad"] = _mock

import gigaam
import gigaam.preprocess as _giga_preprocess
import gigaam.model as _giga_model
from resemblyzer import VoiceEncoder


def _load_audio_no_ffmpeg(path: str) -> torch.Tensor:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        ratio = 16000 / sr
        new_len = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_len)
        audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
    return torch.from_numpy(audio)


_giga_preprocess.load_audio = _load_audio_no_ffmpeg
_giga_model.load_audio = _load_audio_no_ffmpeg  # ← ключевое: патчим и в model


# ==========================================================================
#  НАСТРОЙКИ
# ==========================================================================

# - Аудио -
SAMPLE_RATE = 16000
CHUNK_MS = 32
CHUNK_SAMPLES = 512  # Silero VAD требует ровно 512 при 16kHz

# - VAD пороги -
VAD_THRESHOLD = 0.5
VAD_NEG_THRESHOLD = 0.35
SILENCE_TIMEOUT_MS = 700
MIN_SPEECH_MS = 250
MAX_SPEECH_S = 25  # ← ограничение GigaAM transcribe (25с)
PRE_BUFFER_CHUNKS = 16

# - Стриминг -
PARTIAL_INTERVAL_S = 0.6  # частичный результат каждые 600мс
MIN_PARTIAL_AUDIO_S = 0.5  # минимальная длина для partial-инференса

# - Wake Word -
DEFAULT_WAKE_WORD = "привет коробка"
WAKE_WORD_MIN_MATCH_RATIO = 0.8

# - Stop Words -
DEFAULT_STOP_WORDS = [
    "спасибо",
    "стоп",
    "пока",
    "до свидания",
    "хватит",
    "всё",
    "достаточно",
]

# - Сессия -
SESSION_TIMEOUT_S = 15
SESSION_MAX_DURATION_S = 120

# - Speaker Verification -
SPEAKER_SIMILARITY_THRESHOLD = 0.75
SPEAKER_REJECT_THRESHOLD = 0.60

# - ASR фильтр (для CTC — упрощённый) -
MIN_TEXT_LENGTH = 2
MAX_REPEAT_RATIO = 0.6


# ==========================================================================
#  ОБЩИЕ УТИЛИТЫ
# ==========================================================================


def fuzzy_match(word_a: str, word_b: str) -> bool:
    """Нечеткое сравнение двух слов. Единая функция для wake/stop."""
    if word_a == word_b:
        return True
    a_clean = re.sub(r"[^\w]", "", word_a)
    b_clean = re.sub(r"[^\w]", "", word_b)
    if a_clean == b_clean:
        return True
    if len(a_clean) > 2 and len(b_clean) > 2:
        if a_clean in b_clean or b_clean in a_clean:
            return True
    return False


# ==========================================================================
#  ВЫХОД ДЛЯ LLM (JSONL)
# ==========================================================================


class ASROutput:
    """
    Пишет распознанный текст в файл — по одной фразе на строку.

    LLM-модуль читает файл и получает готовый промпт.
    Файл line-buffered: каждая строка доступна сразу после записи.

    Чтение из LLM-модуля:
        with open("asr_output.txt", "r") as f:
            f.seek(0, 2)
            while True:
                line = f.readline().strip()
                if not line:
                    time.sleep(0.1)
                    continue
                response = llm.generate(line)
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._file = open(
            self.path, "a", encoding="utf-8", buffering=1
        )  # line-buffered
        print(f"[OUTPUT] → {self.path.resolve()}")

    def write(self, text: str):
        self._file.write(text.strip() + "\n")

    def close(self):
        if self._file and not self._file.closed:
            self._file.close()


# ==========================================================================
#  СТРУКТУРЫ ДАННЫХ
# ==========================================================================


class TaskType(Enum):
    PARTIAL = auto()
    FINAL = auto()


@dataclass
class ASRTask:
    """Задание для ASR-воркера"""

    task_type: TaskType
    audio: np.ndarray
    generation: int  # счётчик сегментов, для отброса устаревших partial


@dataclass
class TranscriptionResult:
    """Результат ASR"""

    text: str
    is_valid: bool
    reject_reason: str | None = None
    audio_duration: float = 0.0
    inference_time: float = 0.0


@dataclass
class SessionState:
    """Состояние голосовой сессии."""

    active: bool = False
    speaker_embedding: np.ndarray | None = None
    started_at: float = 0.0
    last_speech_at: float = 0.0

    def reset(self):
        self.active = False
        self.speaker_embedding = None
        self.started_at = 0.0
        self.last_speech_at = 0.0


# ==========================================================================
#  ASR ФИЛЬТР (упрощённый для CTC)
# ==========================================================================


class ASRFilter:
    HALLUCINATION_PATTERNS: list[str] = [
        r"^[.\s,!?…\-]+$",
        r"^(а|и|э|м|ну|да|ага|угу|хм+)$",
    ]

    def __init__(self):
        self._patterns = [
            re.compile(p, re.IGNORECASE) for p in self.HALLUCINATION_PATTERNS
        ]

    def validate(self, result: TranscriptionResult) -> TranscriptionResult:
        text = result.text.strip()

        cleaned = re.sub(r"[^\w]", "", text)
        if len(cleaned) < MIN_TEXT_LENGTH:
            result.is_valid = False
            result.reject_reason = "text_too_short"
            return result

        for pattern in self._patterns:
            if pattern.search(text):
                result.is_valid = False
                result.reject_reason = f"noise_pattern: {pattern.pattern}"
                return result

        words = text.lower().split()
        if len(words) >= 3:
            unique = set(words)
            repeat_ratio = 1 - len(unique) / len(words)
            if repeat_ratio > MAX_REPEAT_RATIO:
                result.is_valid = False
                result.reject_reason = f"word_repeat_ratio={repeat_ratio:.2f}"
                return result

        result.is_valid = True
        return result


# ==========================================================================
#  SILERO VAD
# ==========================================================================


class SileroVAD:
    def __init__(self, threshold: float = VAD_THRESHOLD):
        self.threshold = threshold
        self.model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        self.model.eval()
        self.reset()

    def reset(self):
        self.model.reset_states()

    def is_speech(self, audio_chunk: np.ndarray) -> float:
        tensor = torch.from_numpy(audio_chunk).float()
        with torch.no_grad():
            prob = self.model(tensor, SAMPLE_RATE).item()
        return prob


# ==========================================================================
#  SPEAKER VERIFIER
# ==========================================================================


class SpeakerVerifier:
    """Верификация говорящего через resemblyzer (GE2E, ~17MB, CPU)."""

    def __init__(self):
        print("[SPEAKER] Загрузка модели верификации голоса...")
        self.encoder = VoiceEncoder(device="cpu")
        print("[SPEAKER] Модель загружена.")

    def extract_embedding(self, audio: np.ndarray) -> np.ndarray:
        wav = audio.astype(np.float32)
        if np.abs(wav).max() > 0:
            wav = wav / np.abs(wav).max()
        return self.encoder.embed_utterance(wav)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm < 1e-8:
            return 0.0
        return float(dot / norm)

    def verify(self, audio: np.ndarray, anchor: np.ndarray) -> tuple[bool, float]:
        embedding = self.extract_embedding(audio)
        sim = self.cosine_similarity(embedding, anchor)
        return sim >= SPEAKER_SIMILARITY_THRESHOLD, sim


# ==========================================================================
#  WAKE WORD DETECTOR
# ==========================================================================


class WakeWordDetector:
    """Нечёткий детектор wake word по тексту."""

    def __init__(self, wake_word: str):
        self.wake_word = wake_word.lower().strip()
        self.wake_words = self.wake_word.split()
        self.min_matches = max(1, int(len(self.wake_words) * WAKE_WORD_MIN_MATCH_RATIO))
        print(
            f'[WAKE] Wake word: "{self.wake_word}" '
            f"(нужно совпадений: {self.min_matches}/{len(self.wake_words)})"
        )

    def check(self, text: str) -> tuple[bool, str]:
        """Возвращает (detected, remaining_text)."""
        text_lower = text.lower().strip()
        text_words = text_lower.split()

        matches = 0
        matched_indices = []
        for ww in self.wake_words:
            for i, tw in enumerate(text_words):
                if fuzzy_match(ww, tw) and i not in matched_indices:
                    matches += 1
                    matched_indices.append(i)
                    break

        detected = matches >= self.min_matches

        remaining = ""
        if detected and matched_indices:
            last_idx = max(matched_indices)
            remaining = " ".join(text_words[last_idx + 1 :]).strip()
            remaining = re.sub(r"^[.,!?\s]+", "", remaining)

        return detected, remaining


# ==========================================================================
#  STOP WORD DETECTOR
# ==========================================================================


class StopWordDetector:
    """Детектор стоп-слов. Считается стопом только короткая фраза."""

    def __init__(self, stop_words: list[str]):
        self.stop_words = [w.lower().strip() for w in stop_words]
        print(f"[STOP] Стоп-слова: {', '.join(self.stop_words)}")

    def check(self, text: str) -> tuple[bool, str | None]:
        text_clean = text.lower().strip()
        text_clean = re.sub(r"[.,!?;:…\-\"']+", "", text_clean).strip()
        text_words = text_clean.split()

        if len(text_words) > 4:
            return False, None

        for sw in self.stop_words:
            sw_words = sw.split()
            if all(
                any(fuzzy_match(sw_w, tw) for tw in text_words) for sw_w in sw_words
            ):
                return True, sw

        return False, None


# ==========================================================================
#  GigaAM ASR ENGINE
# ==========================================================================


class GigaAMEngine:
    def __init__(self, model_name: str = "v3_e2e_ctc"):
        print(f"[ASR] Загрузка модели GigaAM '{model_name}'...")
        self.model = gigaam.load_model(model_name)
        self._tmp_dir = Path(tempfile.mkdtemp(prefix="asr_"))
        self._tmp_path = self._tmp_dir / "buffer.wav"
        print("[ASR] Модель загружена.")

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        """Транскрибирует numpy-массив (float32, 16kHz, mono)."""
        t0 = time.perf_counter()
        duration = len(audio) / SAMPLE_RATE

        # Запись во временный WAV и транскрипция
        sf.write(str(self._tmp_path), audio, SAMPLE_RATE, subtype="FLOAT")
        text = self.model.transcribe(str(self._tmp_path))

        elapsed = time.perf_counter() - t0

        return TranscriptionResult(
            text=text.strip() if text else "",
            is_valid=True,
            audio_duration=duration,
            inference_time=elapsed,
        )

    def cleanup(self):
        try:
            if self._tmp_path.exists():
                self._tmp_path.unlink()
            if self._tmp_dir.exists():
                self._tmp_dir.rmdir()
        except OSError:
            pass


# ==========================================================================
#  МИКРОФОН
# ==========================================================================


class MicrophoneStream:
    def __init__(self, device_index: int | None = None, max_queue: int = 200):
        self.device_index = device_index
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=max_queue)
        self.stream = None
        self._overflow_count = 0

    def _callback(self, indata, frames, time_info, status):
        if status:
            if status.input_overflow:
                self._overflow_count += 1
                if self._overflow_count % 10 == 1:
                    print(f"[MIC] ! overflow #{self._overflow_count}", file=sys.stderr)
            else:
                print(f"[MIC] {status}", file=sys.stderr)

        try:
            self.audio_queue.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass  # Дропаем чанк вместо блокировки callback

    def start(self):
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=CHUNK_SAMPLES,
            dtype="float32",
            channels=1,
            device=self.device_index,
            callback=self._callback,
            latency="high",  # больший буфер PortAudio
        )
        self.stream.start()
        actual_latency_ms = (
            self.stream.latency * 1000
        )  # latency возвращается в секундах
        print(
            f"[MIC] Стрим запущен "
            f"(device={self.device_index or 'default'}, "
            f"blocksize={CHUNK_SAMPLES}, "
            f"latency={actual_latency_ms:.0f}мс)"
        )

    def get_chunk(self, timeout: float = 0.5) -> np.ndarray | None:
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            if self._overflow_count > 0:
                print(f"[MIC] Стрим остановлен (overflow: {self._overflow_count})")
            else:
                print("[MIC] Стрим остановлен (overflow: 0)")


# ==========================================================================
#  ОСНОВНОЙ ПАЙПЛАЙН
# ==========================================================================


class ASRPipeline:
    """
    Полный ASR-пайплайн с live-стримингом.

    Архитектура:
        Главный поток: читает микрофон >> VAD >> копит аудио
        ASR-воркер:    получает PARTIAL/FINAL задания >> GigaAM >> результат

    Уровень сессии:
        WAITING >> (wake word) >> ACTIVE >> (stop / таймаут) >> WAITING

    Уровень VAD:
        IDLE >> SPEECH >> TRAILING >> IDLE

    Стриминг:
        Пока пользователь говорит, каждые PARTIAL_INTERVAL_S секунд
        запускается частичная транскрипция и результат отображается.
        По окончании речи — финальная транскрипция.
    """

    # VAD-состояния
    VAD_IDLE = "IDLE"
    VAD_SPEECH = "SPEECH"
    VAD_TRAILING = "TRAILING"

    def __init__(
        self,
        model_name: str = "v3_e2e_ctc",
        mic_device: int | None = None,
        wake_word: str = DEFAULT_WAKE_WORD,
        stop_words: list[str] | None = None,
        speaker_verify: bool = True,
        output_path: str | Path = "asr_output.txt",
    ):
        # - Компоненты -
        self.vad = SileroVAD(threshold=VAD_THRESHOLD)
        self.asr = GigaAMEngine(model_name=model_name)
        self.asr_filter = ASRFilter()
        self.wake_detector = WakeWordDetector(wake_word)
        self.stop_detector = StopWordDetector(stop_words or DEFAULT_STOP_WORDS)
        self.mic = MicrophoneStream(device_index=mic_device)
        self.output = ASROutput(output_path)

        self.speaker_verify = speaker_verify
        self.verifier: SpeakerVerifier | None = None
        if speaker_verify:
            self.verifier = SpeakerVerifier()

        # - Состояние -
        self.session = SessionState()
        self.vad_state = self.VAD_IDLE
        self.speech_buffer: list[np.ndarray] = []
        self.pre_buffer: deque[np.ndarray] = deque(maxlen=PRE_BUFFER_CHUNKS)
        self.silence_start: float = 0.0
        self.speech_start: float = 0.0

        # - Стриминг -
        self._last_partial_time: float = 0.0
        self._last_partial_text: str = ""
        self._segment_generation: int = 0  # для отбрасывания устаревших partial
        self._wake_detected_this_segment: bool = False

        # - Потоки -
        self._running = False
        self._asr_thread: threading.Thread | None = None
        self._asr_queue: queue.Queue[ASRTask] = queue.Queue(maxsize=8)
        self._session_lock = threading.Lock()
        self._mic_muted = threading.Event()

        # - Callback -
        self.on_transcription: Callable[[str], Any] | None = None

    # - Управление мьютом (для TTS) -

    def mute_mic(self):
        """Замьютить обработку речи (вызывать когда TTS говорит)."""
        self._mic_muted.set()
        print("  [MIC] Замьючен (TTS)")

    def unmute_mic(self):
        """Размьютить обработку речи."""
        self._mic_muted.clear()
        self.vad.reset()
        print("  [MIC] Размьючен")

    # - VAD стейт-машина -

    def _process_chunk(self, chunk: np.ndarray):
        # Если TTS говорит сразу выходим, не тратим VAD
        if self._mic_muted.is_set():
            return

        prob = self.vad.is_speech(chunk)
        now = time.monotonic()

        if self.vad_state == self.VAD_IDLE:
            self.pre_buffer.append(chunk)
            if prob >= VAD_THRESHOLD:
                self.vad_state = self.VAD_SPEECH
                self.speech_buffer = list(self.pre_buffer)
                self.pre_buffer.clear()
                self.speech_start = now
                self._last_partial_time = now
                self._last_partial_text = ""
                self._wake_detected_this_segment = False
                self._segment_generation += 1

                with self._session_lock:
                    mode = "[+]" if self.session.active else "[-]"
                print(f"  [VAD] {mode} Речь обнаружена")

        elif self.vad_state == self.VAD_SPEECH:
            self.speech_buffer.append(chunk)

            # Обновляем last_speech_at пока речь идет
            if prob >= VAD_THRESHOLD:
                with self._session_lock:
                    if self.session.active:
                        self.session.last_speech_at = now

            # Проверяем, пора ли отправить partial
            if now - self._last_partial_time >= PARTIAL_INTERVAL_S:
                self._send_partial()
                self._last_partial_time = now

            if prob < VAD_NEG_THRESHOLD:
                self.vad_state = self.VAD_TRAILING
                self.silence_start = now

            if now - self.speech_start > MAX_SPEECH_S:
                self._flush_buffer()

        elif self.vad_state == self.VAD_TRAILING:
            self.speech_buffer.append(chunk)

            # Продолжаем проверять partial и в trailing
            if now - self._last_partial_time >= PARTIAL_INTERVAL_S:
                self._send_partial()
                self._last_partial_time = now

            if prob >= VAD_THRESHOLD:
                self.vad_state = self.VAD_SPEECH
            elif (now - self.silence_start) * 1000 >= SILENCE_TIMEOUT_MS:
                self._flush_buffer()

    def _send_partial(self):
        """Отправляет текущий буфер как PARTIAL-задание."""
        if not self.speech_buffer:
            return
        audio = np.concatenate(self.speech_buffer)
        duration = len(audio) / SAMPLE_RATE
        if duration < MIN_PARTIAL_AUDIO_S:
            return

        task = ASRTask(
            task_type=TaskType.PARTIAL,
            audio=audio.copy(),
            generation=self._segment_generation,
        )
        try:
            self._asr_queue.put_nowait(task)
        except queue.Full:
            pass  # Если очередь забита пропускаем partial

    def _flush_buffer(self):
        """Отправляет буфер как FINAL-задание и сбрасывает VAD."""
        if self.speech_buffer:
            audio = np.concatenate(self.speech_buffer)
            duration = len(audio) / SAMPLE_RATE
            if duration >= MIN_SPEECH_MS / 1000:
                task = ASRTask(
                    task_type=TaskType.FINAL,
                    audio=audio.copy(),
                    generation=self._segment_generation,
                )
                # Для FINAL — ждём место в очереди (важно не потерять)
                try:
                    self._asr_queue.put(task, timeout=2.0)
                except queue.Full:
                    print("  [ASR] ! Очередь переполнена, сегмент потерян")

        self.speech_buffer = []
        self.pre_buffer.clear()
        self.vad.reset()
        self.vad_state = self.VAD_IDLE

    # - ASR-воркер -

    def _asr_worker(self):
        while self._running or not self._asr_queue.empty():
            try:
                task = self._asr_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # Устаревшие partial пропускаем
            if task.task_type == TaskType.PARTIAL:
                if task.generation != self._segment_generation:
                    continue
                # Если в очереди уже есть FINAL - partial не нужен
                if not self._asr_queue.empty():
                    continue

            with self._session_lock:
                session_active = self.session.active

            if task.task_type == TaskType.PARTIAL:
                self._handle_partial(task.audio, session_active)
            else:
                self._handle_final(task.audio, session_active)

    # - Partial: промежуточный результат -

    def _handle_partial(self, audio: np.ndarray, session_active: bool):
        result = self.asr.transcribe(audio)
        text = result.text.strip()
        if not text:
            return

        if not session_active:
            # Проверяем wake word в partial - для быстрого срабатывания
            if not self._wake_detected_this_segment:
                detected, remaining = self.wake_detector.check(text)
                if detected:
                    self._wake_detected_this_segment = True
                    self._activate_session(audio)
                    if remaining:
                        # Показываем начало команды
                        sys.stdout.write(f"\r  [LIVE] ... {remaining}...")
                        sys.stdout.flush()
                        self._last_partial_text = remaining
                else:
                    # Показываем что слышим (ожидание wake word)
                    sys.stdout.write(
                        f'\r  [WAKE] "{text}" ({result.inference_time:.2f}с)   '
                    )
                    sys.stdout.flush()
        else:
            # В активной сессии — показываем live-текст
            # Убираем wake word если он в начале
            display_text = text
            det, rem = self.wake_detector.check(text)
            if det and rem:
                display_text = rem

            if display_text != self._last_partial_text:
                sys.stdout.write(f"\r  [LIVE] ... {display_text}   ")
                sys.stdout.flush()
                self._last_partial_text = display_text

    # - Final: окончательный результат -

    def _handle_final(self, audio: np.ndarray, session_active: bool):
        # Очищаем строку partial
        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.flush()

        result = self.asr.transcribe(audio)
        result = self.asr_filter.validate(result)

        now = time.monotonic()

        if not session_active:
            self._handle_final_waiting(result, audio, now)
        else:
            self._handle_final_active(result, audio, now)

    def _handle_final_waiting(
        self, result: TranscriptionResult, audio: np.ndarray, now: float
    ):
        """Обработка финального результата в режиме ожидания wake word."""
        if not result.text.strip():
            return

        detected, remaining = self.wake_detector.check(result.text)

        if not detected:
            rtf = (
                result.inference_time / result.audio_duration
                if result.audio_duration > 0
                else 0
            )
            print(
                f'  [WAKE] "{result.text}" '
                f"({result.inference_time:.2f}с, RTF: {rtf:.2f}x)"
            )
            return

        # Wake word обнаружен!
        self._activate_session(audio, now)

        # Если есть команда после wake word — обрабатываем
        if remaining:
            print(f'  [SESSION] Команда: "{remaining}"')
            self.output.write(remaining)
            if self.on_transcription:
                self.on_transcription(remaining)

    def _handle_final_active(
        self, result: TranscriptionResult, audio: np.ndarray, now: float
    ):
        """Обработка финального результата в активной сессии."""
        with self._session_lock:
            self.session.last_speech_at = now

        if not result.text.strip():
            return

        if not result.is_valid:
            print(f'  [FILTER] X "{result.text}" → {result.reject_reason}')
            return

        # - Speaker verification (только при первом подозрении) -
        if self.verifier and self.session.speaker_embedding is not None:
            is_same, similarity = self.verifier.verify(
                audio, self.session.speaker_embedding
            )
            if not is_same:
                if similarity < SPEAKER_REJECT_THRESHOLD:
                    print(f"  [SPEAKER] X Чужой голос (sim={similarity:.2f})")
                    return
                else:
                    print(f"  [SPEAKER] ~ Неуверенно (sim={similarity:.2f})")

        rtf = (
            result.inference_time / result.audio_duration
            if result.audio_duration > 0
            else 0
        )

        # - Стоп-слово -
        is_stop, stop_word = self.stop_detector.check(result.text)
        if is_stop:
            with self._session_lock:
                self.session.reset()
            print()
            print("  ┌──────────────────────────────────────────────────")
            print(f'  │ Стоп-слово: "{result.text}" (→ "{stop_word}")')
            print("  │ Сессия завершена")
            print("  │ Жду wake word...")
            print("  └──────────────────────────────────────────────────")
            print()
            return

        # - Повторный wake word - обновляем отпечаток -
        detected, remaining = self.wake_detector.check(result.text)
        if detected:
            if self.verifier:
                new_emb = self.verifier.extract_embedding(audio)
                with self._session_lock:
                    self.session.speaker_embedding = new_emb
                    self.session.last_speech_at = now
                print("  [SESSION] Отпечаток обновлён")
            if remaining:
                result.text = remaining
            else:
                return

        # ── Вывод финального результата ──
        if result.text.strip():
            print()
            print("  ╔══════════════════════════════════════════════════")
            print(f"  ║ {result.text}")
            print(
                f"  ║ {result.inference_time:.2f}с "
                f"(RTF: {rtf:.2f}x, аудио: {result.audio_duration:.1f}с)"
            )
            print("  ╚══════════════════════════════════════════════════")
            print()

            if self.on_transcription:
                self.on_transcription(result.text)

            self.output.write(result.text)

    # - Активация сессии -

    def _activate_session(self, audio: np.ndarray, now: float | None = None):
        if now is None:
            now = time.monotonic()

        speaker_embedding = None
        if self.verifier:
            speaker_embedding = self.verifier.extract_embedding(audio)

        with self._session_lock:
            self.session.active = True
            self.session.speaker_embedding = speaker_embedding
            self.session.started_at = now
            self.session.last_speech_at = now

        print()
        print("  ┌──────────────────────────────────────────────────")
        print("  │ Wake word обнаружен!")
        if speaker_embedding is not None:
            print("  │ Голосовой отпечаток захвачен")
        print(f"  │ Сессия активна (таймаут: {SESSION_TIMEOUT_S}с)")
        print(f"  │ Стоп-слова: {', '.join(self.stop_detector.stop_words[:3])}...")
        print("  └──────────────────────────────────────────────────")
        print()

    # - Таймаут сессии -

    def _check_session_timeout(self):
        reason = None
        with self._session_lock:
            if not self.session.active:
                return

            now = time.monotonic()
            idle = now - self.session.last_speech_at
            total = now - self.session.started_at

            if idle > SESSION_TIMEOUT_S:
                reason = f"тишина {idle:.0f}с"
            elif total > SESSION_MAX_DURATION_S:
                reason = f"макс. время {SESSION_MAX_DURATION_S}с"

            if reason:
                self.session.reset()

        if reason:
            print()
            print("  ┌──────────────────────────────────────────────────")
            print(f"  │ Сессия завершена ({reason})")
            print("  │ Жду wake word...")
            print("  └──────────────────────────────────────────────────")
            print()

    # - Основной цикл -

    def run(self):
        import signal

        sv_status = "ON" if self.speaker_verify else "OFF"
        print()
        print("=" * 58)
        print("  University Terminal ASR")
        print("  Модель: GigaAM-v3 CTC (streaming)")
        print(f'  Wake word: "{self.wake_detector.wake_word}"')
        print(f"  Stop words: {', '.join(self.stop_detector.stop_words[:4])}...")
        print(f"  Speaker verification: {sv_status}")
        print(f"  Live ASR: partial каждые {PARTIAL_INTERVAL_S}с")
        print("  Ctrl+C для остановки")
        print("=" * 58)
        print()
        print("  Жду wake word...\n")

        self._running = True

        def _signal_handler(sig, frame):
            print("\n  Получен сигнал завершения...")
            self._running = False

        signal.signal(signal.SIGINT, _signal_handler)

        self._asr_thread = threading.Thread(target=self._asr_worker, daemon=True)
        self._asr_thread.start()
        self.mic.start()

        last_timeout_check = time.monotonic()

        try:
            while self._running:
                chunk = self.mic.get_chunk(timeout=0.5)
                if chunk is not None:
                    self._process_chunk(chunk)

                now = time.monotonic()
                if now - last_timeout_check >= 1.0:
                    self._check_session_timeout()
                    last_timeout_check = now

        except KeyboardInterrupt:
            pass
        finally:
            print("\n  Останавливаюсь...")
            self._running = False
            self.mic.stop()
            if self._asr_thread:
                self._asr_thread.join(timeout=3.0)
            self.asr.cleanup()
            self.output.close()
            print("  Готово.")


# ==========================================================================
#  CLI
# ==========================================================================


def list_audio_devices():
    print("\nДоступные аудио-устройства:\n")
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            marker = " ◀ default" if i == sd.default.device[0] else ""
            print(
                f"  [{i}] {d['name']} ({d['max_input_channels']}ch, "
                f"{int(d['default_samplerate'])}Hz){marker}"
            )
    print()


def main():
    parser = argparse.ArgumentParser(
        description="University Terminal ASR v2: GigaAM Streaming Pipeline"
    )
    parser.add_argument(
        "--list-devices", action="store_true", help="Показать список аудиоустройств"
    )
    parser.add_argument(
        "--device", type=int, default=None, help="Индекс микрофона (см. --list-devices)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="v3_e2e_ctc",
        help="Модель GigaAM (default: v3_e2e_ctc)",
    )
    parser.add_argument(
        "--wake-word",
        type=str,
        default=DEFAULT_WAKE_WORD,
        help=f'Wake word (default: "{DEFAULT_WAKE_WORD}")',
    )
    parser.add_argument(
        "--stop-words", type=str, default=None, help="Стоп-слова через запятую"
    )
    parser.add_argument(
        "--no-speaker-verify", action="store_true", help="Отключить верификацию голоса"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runtime/asr_output.txt",
        help="Путь к файлу для LLM (default: runtime/asr_output.txt)",
    )

    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        sys.exit(0)

    stop_words = None
    if args.stop_words:
        stop_words = [w.strip() for w in args.stop_words.split(",")]

    pipeline = ASRPipeline(
        model_name=args.model,
        mic_device=args.device,
        wake_word=args.wake_word,
        stop_words=stop_words,
        speaker_verify=not args.no_speaker_verify,
        output_path=args.output,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
