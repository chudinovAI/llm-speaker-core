from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from llm_speaker_core.voice import asr as asr_module
from llm_speaker_core.voice.bridge import run_bridge


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified voice stack: ASR -> LLM/RAG -> TTS."
    )
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=Path("runtime"),
        help="Directory for runtime text/jsonl/wav outputs.",
    )
    parser.add_argument("--session-id", type=str, default="voice-live-1")
    parser.add_argument("--list-devices", action="store_true", help="List ASR input devices and exit.")
    parser.add_argument("--asr-device", type=int, default=None, help="ASR microphone device index.")
    parser.add_argument("--asr-model", type=str, default="v3_e2e_ctc", help="GigaAM model name.")
    parser.add_argument("--wake-word", type=str, default=asr_module.DEFAULT_WAKE_WORD)
    parser.add_argument("--stop-words", type=str, default=None, help="Comma-separated stop words.")
    parser.add_argument("--no-speaker-verify", action="store_true")
    parser.add_argument("--tts-speaker", type=str, default="aidar")
    parser.add_argument("--tts-sample-rate", type=int, default=48000, choices=[8000, 24000, 48000])
    parser.add_argument("--tts-device", type=str, default="cpu")
    parser.add_argument("--tts-speed", type=float, default=1.0)
    parser.add_argument("--tts-play", action="store_true", help="Play TTS through the local speaker.")
    parser.add_argument("--mode", choices=("direct", "api"), default="direct")
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://127.0.0.1:8000/query",
        help="Used only when --mode api.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_devices:
        asr_module.list_audio_devices()
        return

    runtime_dir = args.runtime_dir
    runtime_dir.mkdir(parents=True, exist_ok=True)
    asr_output = runtime_dir / "asr_output.txt"
    llm_output = runtime_dir / "asr_llm_output.jsonl"
    speaker_output = runtime_dir / "speaker_output.txt"
    tts_dir = runtime_dir / "tts"

    asr_cmd = [
        sys.executable,
        "-m",
        "llm_speaker_core.voice.asr",
        "--model",
        args.asr_model,
        "--wake-word",
        args.wake_word,
        "--output",
        str(asr_output),
    ]
    if args.asr_device is not None:
        asr_cmd.extend(["--device", str(args.asr_device)])
    if args.stop_words:
        asr_cmd.extend(["--stop-words", args.stop_words])
    if args.no_speaker_verify:
        asr_cmd.append("--no-speaker-verify")

    print(f"[VOICE] runtime={runtime_dir.resolve()}")
    print(f"[VOICE] starting ASR: {' '.join(asr_cmd)}")

    asr_process = subprocess.Popen(asr_cmd)
    bridge_args = argparse.Namespace(
        input=asr_output,
        mode=args.mode,
        api_url=args.api_url,
        session_id=args.session_id,
        out=llm_output,
        speaker_output=speaker_output,
        poll_interval=0.15,
        timeout_s=35.0,
        from_start=False,
        tts_enabled=True,
        tts_output_dir=tts_dir,
        tts_play=args.tts_play,
        tts_speaker=args.tts_speaker,
        tts_sample_rate=args.tts_sample_rate,
        tts_device=args.tts_device,
        tts_speed=args.tts_speed,
    )

    try:
        run_bridge(bridge_args)
    except KeyboardInterrupt:
        print("\n[VOICE] stopping stack...")
    finally:
        asr_process.terminate()
        try:
            asr_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            asr_process.kill()
            asr_process.wait(timeout=5)


if __name__ == "__main__":
    main()
