from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import httpx

from llm_speaker_core.api import build_service
from llm_speaker_core.voice.tts import DEFAULT_SAMPLE_RATE, DEFAULT_SPEAKER, SPEAKERS, SileroTTS

SKIP_TEXTS = {
    "stop",
    "стоп",
    "спасибо",
    "спасибо, стоп",
    "пока",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bridge ASR text output to LLM and optional TTS."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("runtime/asr_output.txt"),
        help="ASR text file (one utterance per line).",
    )
    parser.add_argument(
        "--mode",
        choices=("direct", "api"),
        default="direct",
        help="direct = call LLM service in-process, api = POST to /query.",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://127.0.0.1:8000/query",
        help="LLM API /query URL for --mode api.",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default="asr-live-1",
        help="session_id sent to the LLM service.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runtime/asr_llm_output.jsonl"),
        help="Output JSONL path for LLM responses.",
    )
    parser.add_argument(
        "--speaker-output",
        type=Path,
        default=Path("runtime/speaker_output.txt"),
        help="Optional text output for speaker_text stream.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.15,
        help="Polling interval for file tailing.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=35.0,
        help="HTTP timeout seconds for --mode api.",
    )
    parser.add_argument(
        "--from-start",
        action="store_true",
        help="Read existing file contents from beginning.",
    )
    parser.add_argument(
        "--tts-enabled",
        action="store_true",
        help="Synthesize speaker_text to WAV and optionally play it.",
    )
    parser.add_argument(
        "--tts-output-dir",
        type=Path,
        default=Path("runtime/tts"),
        help="Directory for generated TTS WAV files.",
    )
    parser.add_argument(
        "--tts-play",
        action="store_true",
        help="Play synthesized audio through the local speaker.",
    )
    parser.add_argument(
        "--tts-speaker",
        type=str,
        default=DEFAULT_SPEAKER,
        choices=SPEAKERS,
        help="Silero speaker voice.",
    )
    parser.add_argument(
        "--tts-sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        choices=[8000, 24000, 48000],
        help="Silero sample rate.",
    )
    parser.add_argument(
        "--tts-device",
        type=str,
        default="cpu",
        help="TTS device: cpu or cuda.",
    )
    parser.add_argument(
        "--tts-speed",
        type=float,
        default=1.0,
        help="TTS speed multiplier.",
    )
    return parser


def _should_skip_text(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    return not normalized or normalized in SKIP_TEXTS


def _direct_query(service: Any, text: str, session_id: str) -> dict[str, Any]:
    result = service.handle_query(text=text, session_id=session_id, history=None)
    return {
        "display_text": result.display_text,
        "speaker_text": result.speaker_text,
        "meta": {
            "latency_ms": result.latency_ms,
            "used_rag": result.used_rag,
            "fallback_used": result.fallback_used,
            "limits_applied": result.limits_applied,
            "rag_hits": result.rag_hits,
            "rag_sources": result.rag_sources,
            "intent": result.intent,
            "evidence_coverage": result.evidence_coverage,
            "answer_mode": result.answer_mode,
        },
    }


def _api_query(client: httpx.Client, api_url: str, text: str, session_id: str) -> dict[str, Any]:
    response = client.post(api_url, json={"text": text, "session_id": session_id})
    response.raise_for_status()
    return response.json()


def _build_tts(args: argparse.Namespace) -> SileroTTS | None:
    if not args.tts_enabled:
        return None
    args.tts_output_dir.mkdir(parents=True, exist_ok=True)
    return SileroTTS(
        speaker=args.tts_speaker,
        sample_rate=args.tts_sample_rate,
        speed=args.tts_speed,
        device=args.tts_device,
    )


def _write_speaker_output(path: Path | None, text: str) -> None:
    if path is None or not text:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", buffering=1) as f:
        f.write(text.strip() + "\n")


def run_bridge(args: argparse.Namespace) -> None:
    args.input.parent.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.input.touch(exist_ok=True)

    print(f"[ASR->LLM] input={args.input.resolve()}")
    print(f"[ASR->LLM] out={args.out.resolve()}")
    print(f"[ASR->LLM] mode={args.mode}")
    print(f"[ASR->LLM] session_id={args.session_id}")
    if args.speaker_output:
        print(f"[ASR->LLM] speaker_out={args.speaker_output.resolve()}")

    tts = _build_tts(args)
    service = build_service() if args.mode == "direct" else None
    tts_counter = 0

    with args.input.open("r", encoding="utf-8") as in_f, args.out.open(
        "a", encoding="utf-8", buffering=1
    ) as out_f, httpx.Client(timeout=args.timeout_s) as client:
        if not args.from_start:
            in_f.seek(0, 2)

        while True:
            line = in_f.readline()
            if not line:
                time.sleep(args.poll_interval)
                continue

            text = line.strip()
            if _should_skip_text(text):
                continue

            ts = time.time()
            try:
                if args.mode == "direct":
                    result = _direct_query(service, text, args.session_id)
                else:
                    result = _api_query(client, args.api_url, text, args.session_id)

                record = {
                    "ts": ts,
                    "input_text": text,
                    "display_text": result.get("display_text", ""),
                    "speaker_text": result.get("speaker_text", ""),
                    "meta": result.get("meta", {}),
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

                display = str(record["display_text"]).replace("\n", " ")
                speaker = str(record["speaker_text"]).replace("\n", " ")
                print(f'[ASR->LLM] Q: "{text}"')
                print(f"[ASR->LLM] D: {display}")
                print(f"[ASR->LLM] S: {speaker}")

                _write_speaker_output(args.speaker_output, record["speaker_text"])

                if tts is not None and record["speaker_text"]:
                    tts_counter += 1
                    wav_path = args.tts_output_dir / f"tts_{tts_counter:04d}.wav"
                    tts.speak(
                        record["speaker_text"],
                        output_path=str(wav_path),
                        play_audio=args.tts_play,
                    )
            except Exception as exc:  # noqa: BLE001
                err = {"ts": ts, "input_text": text, "error": str(exc)}
                out_f.write(json.dumps(err, ensure_ascii=False) + "\n")
                print(f'[ASR->LLM] ERROR for "{text}": {exc}')


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_bridge(args)


if __name__ == "__main__":
    main()
