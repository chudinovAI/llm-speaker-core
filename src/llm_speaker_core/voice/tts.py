"""
Silero TTS pipeline.

Основные режимы:
- разовый синтез в WAV;
- watch-режим по текстовому файлу;
- проигрывание speaker_text локально через динамик.

CLI examples:
  uv run llm-tts --input runtime/speaker_output.txt --watch --output runtime/tts
  echo "проверка генерации" | uv run llm-tts --stdin --play
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch


# --------------------------------------------------------------
#  Конфигурация
# --------------------------------------------------------------

LANGUAGE = "ru"
MODEL_ID = "v5_ru"
DEFAULT_SPEAKER = "aidar"
DEFAULT_SAMPLE_RATE = 48000
SPEAKERS = ["aidar", "baya", "kseniya", "xenia", "eugene"]

# Разбиение текста на предложения
SENTENCE_SPLIT = re.compile(r'(?<=[.!?…])\s+')

# Минимальная длина фрагмента для синтеза (символов)
MIN_CHUNK_LEN = 10

# Максимум WAV файлов в watch-режиме (0 = без лимита)
DEFAULT_MAX_WAV_FILES = 50


# --------------------------------------------------------------
#  Текстовый препроцессор
# --------------------------------------------------------------

class TextPreprocessor:
    """
    Нормализует текст перед синтезом: сокращения, числа,
    спецсимволы, мусор от LLM. Быстрый — чистый regex/dict.
    """

    # Сокращения → полная форма
    ABBREVIATIONS = {
        # Адреса
        r"\bул\.":   "улица",
        r"\bпр\.":   "проспект",
        r"\bпр-т\.?": "проспект",
        r"\bд\.":    "дом",
        r"\bкорп\.": "корпус",
        r"\bстр\.":  "строение",
        r"\bкв\.":   "квартира",
        r"\bкаб\.":  "кабинет",
        r"\bауд\.":  "аудитория",
        r"\bэт\.":   "этаж",
        # Звания / обращения
        r"\bпроф\.":  "профессор",
        r"\bдоц\.":   "доцент",
        r"\bст\.\s*пр\.": "старший преподаватель",
        r"\bзав\.":   "заведующий",
        # Учебное
        r"\bлаб\.":   "лабораторная",
        r"\bпр\.\s*зан\.": "практическое занятие",
        r"\bсем\.":   "семестр",
        r"\bэкз\.":   "экзамен",
        r"\bзач\.":   "зачёт",
        # Общее
        r"\bг\.":     "город",
        r"\bт\.е\.":  "то есть",
        r"\bт\.д\.":  "так далее",
        r"\bт\.п\.":  "тому подобное",
        r"\bт\.к\.":  "так как",
        r"\bтел\.":   "телефон",
        r"\bдр\.":    "другое",
        r"\bсм\.":    "смотри",
        r"\bрис\.":   "рисунок",
        r"\bтабл\.":  "таблица",
        r"\bим\.":    "имени",
        r"\bнапр\.":  "например",
    }

    # Единицы измерения
    UNITS = {
        r"\bкг\b":   "килограмм",
        r"\bг\b(?=[\s,\.])":    "грамм",
        r"\bкм\b":   "километров",
        r"\bм\b(?=[\s,\.])":    "метров",
        r"\bсм\b":   "сантиметров",
        r"\bмм\b":   "миллиметров",
        r"\bч\b(?=[\s,\.])":    "часов",
        r"\bмин\b":  "минут",
        r"\bсек\b":  "секунд",
        r"\bруб\b\.?": "рублей",
    }

    # Числительные для простых случаев (1-20, десятки, сотни)
    _ONES = {
        "0": "ноль", "1": "один", "2": "два", "3": "три",
        "4": "четыре", "5": "пять", "6": "шесть", "7": "семь",
        "8": "восемь", "9": "девять", "10": "десять",
        "11": "одиннадцать", "12": "двенадцать", "13": "тринадцать",
        "14": "четырнадцать", "15": "пятнадцать", "16": "шестнадцать",
        "17": "семнадцать", "18": "восемнадцать", "19": "девятнадцать",
    }
    _TENS = {
        "20": "двадцать", "30": "тридцать", "40": "сорок",
        "50": "пятьдесят", "60": "шестьдесят", "70": "семьдесят",
        "80": "восемьдесят", "90": "девяносто",
    }
    _HUNDREDS = {
        "100": "сто", "200": "двести", "300": "триста",
        "400": "четыреста", "500": "пятьсот", "600": "шестьсот",
        "700": "семьсот", "800": "восемьсот", "900": "девятьсот",
    }

    @classmethod
    def _number_to_words(cls, n: int) -> str:
        """Конвертирует число 0-9999 в слова. Для больших - оставляет как есть."""
        if n < 0:
            return "минус " + cls._number_to_words(-n)
        s = str(n)
        if s in cls._ONES:
            return cls._ONES[s]
        if n < 100:
            tens = str(n // 10 * 10)
            ones = n % 10
            result = cls._TENS.get(tens, tens)
            if ones:
                result += " " + cls._ONES[str(ones)]
            return result
        if n < 1000:
            hundreds = str(n // 100 * 100)
            remainder = n % 100
            result = cls._HUNDREDS.get(hundreds, hundreds)
            if remainder:
                result += " " + cls._number_to_words(remainder)
            return result
        if n < 10000:
            thousands = n // 1000
            remainder = n % 1000
            prefix = cls._ONES.get(str(thousands), str(thousands))
            if thousands == 1:
                result = "одна тысяча"
            elif thousands == 2:
                result = "две тысячи"
            elif thousands in (3, 4):
                result = prefix + " тысячи"
            else:
                result = prefix + " тысяч"
            if remainder:
                result += " " + cls._number_to_words(remainder)
            return result
        return s

    @classmethod
    def _number_with_zeros(cls, s: str) -> str:
        #учет ведущих нулей
        if not s or not s.isdigit():
            return s

        leading_zeros = len(s) - len(s.lstrip("0"))

        if leading_zeros == len(s):
            return " ".join(["ноль"] * len(s))

        if leading_zeros > 0:
            prefix = " ".join(["ноль"] * leading_zeros)
            remainder = int(s[leading_zeros:])
            if remainder == 0:
                return prefix
            return prefix + " " + cls._number_to_words(remainder)

        n = int(s)
        if 0 <= n <= 9999:
            return cls._number_to_words(n)
        return s

    @classmethod
    def process(cls, text: str) -> str:
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)   # **bold**
        text = re.sub(r'\*(.+?)\*', r'\1', text)        # *italic*
        text = re.sub(r'`(.+?)`', r'\1', text)          # `code`
        text = re.sub(r'#{1,6}\s*', '', text)            # ### headers
        text = re.sub(r'^\s*[-*•]\s+', '', text, flags=re.MULTILINE)  # bullet points
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)  # numbered lists

        # 2. Спецсимволы
        text = text.replace("—", " — ")
        text = text.replace("–", " — ")
        text = text.replace("«", "").replace("»", "")
        text = text.replace('"', "").replace('"', "").replace('"', "")
        text = text.replace("(", ", ").replace(")", ", ")
        text = text.replace("/", " или ")

        for pattern, replacement in cls.ABBREVIATIONS.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        for pattern, replacement in cls.UNITS.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        def _replace_compound(m):
            parts = m.group(0).split("-")
            words = []
            for part in parts:
                words.append(cls._number_with_zeros(part))
            return " ".join(words)
        text = re.sub(r'\b\d{1,4}(?:-\d{1,4})+\b', _replace_compound, text)

        def _replace_time(m):
            h, mins = m.group(1), m.group(2)
            h_word = cls._number_to_words(int(h))
            m_word = cls._number_with_zeros(mins)
            if int(mins) == 0:
                return h_word
            return f"{h_word} {m_word}"
        text = re.sub(r'\b(\d{1,2}):(\d{2})\b', _replace_time, text)

        def _replace_number(m):
            s = m.group(0)
            return cls._number_with_zeros(s)
        text = re.sub(r'\b\d{1,4}\b', _replace_number, text)

        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s+([,\.!?;:])', r'\1', text)
        text = text.strip()

        return text


# --------------------------------------------------------------
#  Silero TTS Engine
# --------------------------------------------------------------

class SileroTTS:

    def __init__(
        self,
        speaker: str = DEFAULT_SPEAKER,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        speed: float = 1.0,
        device: str = "cpu",
        num_threads: int = 4,
    ):
        self.speaker = speaker
        self.sample_rate = sample_rate
        self.speed = speed
        self.device = torch.device(device)

        if device == "cpu":
            torch.set_num_threads(num_threads)

        print(f"[TTS] Загрузка Silero v5 (speaker={speaker}, sr={sample_rate}, speed={speed})...")
        t0 = time.perf_counter()

        self.model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_tts",
            language=LANGUAGE,
            speaker=MODEL_ID,
        )
        self.model.to(self.device)

        print(f"[TTS] Модель загружена за {time.perf_counter() - t0:.1f}с")

    @staticmethod
    def _change_speed(audio: np.ndarray, speed: float) -> np.ndarray:
        """Ускоряет/замедляет речь без внешних DSP-зависимостей."""
        if abs(speed - 1.0) < 0.01:
            return audio
        speed = max(speed, 0.5)
        new_len = max(1, int(len(audio) / speed))
        indices = np.linspace(0, len(audio) - 1, new_len)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    def synthesize(self, text: str) -> np.ndarray:
        """Синтез одного фрагмента текста >> numpy float32 массив."""
        audio = self.model.apply_tts(
            text=text,
            speaker=self.speaker,
            sample_rate=self.sample_rate,
            put_accent=True,
            put_yo=True,
        )
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        else:
            audio = np.array(audio, dtype=np.float32)

        return self._change_speed(audio, self.speed)

    def synthesize_text(self, text: str) -> tuple[np.ndarray, float]:
        """Препроцессинг >> разбиение на предложения >> единый audio array."""
        text = TextPreprocessor.process(text)
        chunks = split_into_sentences(text)
        if not chunks:
            return np.zeros(0, dtype=np.float32), 0.0

        audio_parts = []
        total_infer = 0.0

        for i, chunk in enumerate(chunks):
            t0 = time.perf_counter()
            audio = self.synthesize(chunk)
            elapsed = time.perf_counter() - t0
            total_infer += elapsed

            audio_parts.append(audio)

            duration = len(audio) / self.sample_rate
            print(
                f"  [{i + 1}/{len(chunks)}] "
                f"{len(chunk):>3} сим | "
                f"{elapsed * 1000:.0f}мс | "
                f"{duration:.1f}с аудио | "
                f"\"{chunk[:50]}{'…' if len(chunk) > 50 else ''}\""
            )

        # Склеиваем с маленькой паузой между предложениями
        pause_samples = int(self.sample_rate * 0.15)  # 150мс пауза
        pause = np.zeros(pause_samples, dtype=np.float32)

        parts_with_pauses = []
        for j, part in enumerate(audio_parts):
            parts_with_pauses.append(part)
            if j < len(audio_parts) - 1:
                parts_with_pauses.append(pause)

        full_audio = np.concatenate(parts_with_pauses)
        total_duration = len(full_audio) / self.sample_rate
        rtf = total_infer / total_duration if total_duration > 0 else 0

        print(
            f"  ── Готово: "
            f"{total_duration:.1f}с | "
            f"RTF: {rtf:.3f} | "
            f"infer: {total_infer * 1000:.0f}мс"
        )

        return full_audio, total_duration

    def synthesize_to_file(self, text: str, output_path: str) -> float:
        """Синтезирует текст и сохраняет WAV."""
        full_audio, total_duration = self.synthesize_text(text)
        if len(full_audio) == 0:
            return 0.0
        sf.write(output_path, full_audio, self.sample_rate, subtype="PCM_16")
        return total_duration

    def speak(self, text: str, output_path: str | None = None, play_audio: bool = False) -> float:
        """Синтезирует текст, по желанию пишет WAV и/или воспроизводит его."""
        full_audio, total_duration = self.synthesize_text(text)
        if len(full_audio) == 0:
            return 0.0
        if output_path:
            sf.write(output_path, full_audio, self.sample_rate, subtype="PCM_16")
        if play_audio:
            sd.play(full_audio, self.sample_rate)
            sd.wait()
        return total_duration


# --------------------------------------------------------------
#  Разбиение текста на предложения
# --------------------------------------------------------------

def split_into_sentences(text: str) -> list[str]:
    """Разбивает текст на предложения для потоковой генерации."""
    text = text.strip()
    if not text:
        return []

    parts = SENTENCE_SPLIT.split(text)

    result = []
    buffer = ""

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if buffer:
            buffer += " " + part
        else:
            buffer = part

        if len(buffer) >= MIN_CHUNK_LEN:
            result.append(buffer)
            buffer = ""

    if buffer:
        if result:
            result[-1] += " " + buffer
        else:
            result.append(buffer)

    return result


# --------------------------------------------------------------
#  Чтение из файла
# --------------------------------------------------------------

def read_file_once(path: str) -> str:
    """Читает весь файл целиком."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def cleanup_old_wavs(output_dir: Path, max_files: int):
    """Удаляет самые старые WAV файлы, если их больше max_files."""
    if max_files <= 0:
        return
    wavs = sorted(output_dir.glob("tts_*.wav"), key=lambda p: p.stat().st_mtime)
    while len(wavs) > max_files:
        old = wavs.pop(0)
        old.unlink()
        print(f"  [CLEANUP] Удалён: {old.name}")


def watch_file(path: str, tts: SileroTTS, output_dir: Path, max_files: int = 0):
    """
    Следит за файлом по mtime + размеру.
    Работает и с append (echo >>), и с перезаписью (редакторы).
    Каждая новая строка → отдельный WAV.
    """
    print(f"[WATCH] Слежу за {path}...")
    print(f"[WATCH] Новые строки → WAV в {output_dir}")
    if max_files > 0:
        print(f"[WATCH] Макс. файлов: {max_files} (старые удаляются)")
    print(f"[WATCH] Ctrl+C для остановки\n")

    counter = 0
    known_lines: list[str] = []

    # Запоминаем текущее содержимое как уже обработанное
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            known_lines = f.read().splitlines()

    last_mtime = os.path.getmtime(path) if os.path.exists(path) else 0
    last_size = os.path.getsize(path) if os.path.exists(path) else 0

    while True:
        time.sleep(0.2)

        if not os.path.exists(path):
            continue

        mtime = os.path.getmtime(path)
        size = os.path.getsize(path)

        # Файл не изменился
        if mtime == last_mtime and size == last_size:
            continue

        last_mtime = mtime
        last_size = size

        # Читаем все содержимое
        with open(path, "r", encoding="utf-8") as f:
            current_lines = f.read().splitlines()

        # Находим новые строки (те что появились в конце)
        new_lines = current_lines[len(known_lines):]

        # Если файл стал короче (перезаписан), сравниваем построчно
        if len(current_lines) < len(known_lines):
            new_lines = []
            for i, line in enumerate(current_lines):
                if i >= len(known_lines) or line != known_lines[i]:
                    new_lines = current_lines[i:]
                    break

        known_lines = current_lines[:]

        for line in new_lines:
            line = line.strip()
            if not line:
                continue

            counter += 1
            output_path = output_dir / f"tts_{counter:04d}.wav"

            print(f"\n[WATCH] Получен текст #{counter}:")
            print(f"  \"{line[:80]}{'…' if len(line) > 80 else ''}\"")

            tts.synthesize_to_file(line, str(output_path))
            cleanup_old_wavs(output_dir, max_files)


# --------------------------------------------------------------
#  CLI
# --------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Silero TTS — читает текст LLM, генерирует WAV"
    )

    # Источник текста
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input", type=str,
        help="Путь к текстовому файлу с ответом LLM",
    )
    source.add_argument(
        "--stdin", action="store_true",
        help="Читать текст из stdin",
    )

    # Режим
    parser.add_argument(
        "--watch", action="store_true",
        help="Следить за файлом (tail -f), каждая строка → WAV",
    )

    # Выход
    parser.add_argument(
        "--output", type=str, default=None,
        help="Путь для WAV файла (default: runtime/tts_output.wav / runtime/tts/)",
    )

    # Модель
    parser.add_argument(
        "--speaker", type=str, default=DEFAULT_SPEAKER,
        choices=SPEAKERS,
        help=f"Голос (default: {DEFAULT_SPEAKER})",
    )
    parser.add_argument(
        "--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE,
        choices=[8000, 24000, 48000],
        help=f"Sample rate (default: {DEFAULT_SAMPLE_RATE})",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="cpu или cuda (default: cpu)",
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="Скорость речи: 1.0 = норма, 1.3 = быстрее, 0.8 = медленнее (default: 1.0)",
    )
    parser.add_argument(
        "--max-files", type=int, default=DEFAULT_MAX_WAV_FILES,
        help=f"Макс. WAV файлов в watch-режиме, 0 = без лимита (default: {DEFAULT_MAX_WAV_FILES})",
    )
    parser.add_argument(
        "--play", action="store_true",
        help="Проигрывать синтезированный звук через sounddevice.",
    )

    args = parser.parse_args()

    # Инициализация TTS
    tts = SileroTTS(
        speaker=args.speaker,
        sample_rate=args.sample_rate,
        speed=args.speed,
        device=args.device,
    )

    # -- Режим watch --
    if args.watch:
        if not args.input:
            print("--watch требует --input")
            sys.exit(1)

        output_dir = Path(args.output or "runtime/tts")
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            watch_file(args.input, tts, output_dir, max_files=args.max_files)
        except KeyboardInterrupt:
            print("\n[WATCH] Остановлен.")
        return

    # -- Разовая генерация --
    if args.stdin:
        text = sys.stdin.read().strip()
    else:
        text = read_file_once(args.input)

    if not text:
        print("Пустой текст, нечего синтезировать.")
        sys.exit(0)

    output_path = args.output or "runtime/tts_output.wav"

    print(f"\n  Текст ({len(text)} сим):")
    print(f"  \"{text[:100]}{'…' if len(text) > 100 else ''}\"\n")

    tts.speak(text, output_path=output_path, play_audio=args.play)
    print()


if __name__ == "__main__":
    main()
