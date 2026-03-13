from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import httpx


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bridge ASR text output to LLM /query endpoint."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("asr_output.txt"),
        help="ASR text file (one utterance per line).",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://127.0.0.1:8000/query",
        help="LLM API /query URL.",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default="asr-live-1",
        help="session_id sent to LLM API.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("asr_llm_output.jsonl"),
        help="Output JSONL path for LLM responses.",
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
        help="HTTP timeout seconds.",
    )
    parser.add_argument(
        "--from-start",
        action="store_true",
        help="Read existing file contents from beginning.",
    )
    return parser


def post_query(
    client: httpx.Client, api_url: str, text: str, session_id: str
) -> dict[str, Any]:
    payload = {"text": text, "session_id": session_id}
    response = client.post(api_url, json=payload)
    response.raise_for_status()
    return response.json()


def run_bridge(
    input_path: Path,
    api_url: str,
    session_id: str,
    out_path: Path,
    poll_interval: float,
    timeout_s: float,
    from_start: bool,
) -> None:
    input_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.touch(exist_ok=True)

    print(f"[ASR->LLM] input={input_path.resolve()}")
    print(f"[ASR->LLM] out={out_path.resolve()}")
    print(f"[ASR->LLM] api={api_url}")
    print(f"[ASR->LLM] session_id={session_id}")

    with input_path.open("r", encoding="utf-8") as in_f, out_path.open(
        "a", encoding="utf-8", buffering=1
    ) as out_f, httpx.Client(timeout=timeout_s) as client:
        if not from_start:
            in_f.seek(0, 2)

        while True:
            line = in_f.readline()
            if not line:
                time.sleep(poll_interval)
                continue

            text = line.strip()
            if not text:
                continue

            ts = time.time()
            try:
                result = post_query(client, api_url, text, session_id)
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
            except Exception as exc:  # noqa: BLE001
                err = {"ts": ts, "input_text": text, "error": str(exc)}
                out_f.write(json.dumps(err, ensure_ascii=False) + "\n")
                print(f'[ASR->LLM] ERROR for "{text}": {exc}')


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_bridge(
        input_path=args.input,
        api_url=args.api_url,
        session_id=args.session_id,
        out_path=args.out,
        poll_interval=args.poll_interval,
        timeout_s=args.timeout_s,
        from_start=args.from_start,
    )


if __name__ == "__main__":
    main()
