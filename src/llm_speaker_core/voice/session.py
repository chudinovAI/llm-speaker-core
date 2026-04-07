from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from llm_speaker_core.voice.events import VoiceEvent


class VoiceState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ERROR = "error"


@dataclass(frozen=True)
class VoiceRuntimePaths:
    runtime_dir: Path
    asr_output: Path
    llm_output: Path
    display_output: Path
    speaker_output: Path
    events_output: Path
    control_output: Path
    tts_dir: Path

    @classmethod
    def from_runtime_dir(cls, runtime_dir: Path) -> "VoiceRuntimePaths":
        return cls(
            runtime_dir=runtime_dir,
            asr_output=runtime_dir / "asr_output.txt",
            llm_output=runtime_dir / "asr_llm_output.jsonl",
            display_output=runtime_dir / "display_output.txt",
            speaker_output=runtime_dir / "speaker_output.txt",
            events_output=runtime_dir / "voice_events.jsonl",
            control_output=runtime_dir / "voice_control.json",
            tts_dir=runtime_dir / "tts",
        )


class VoiceSessionController:
    def __init__(self) -> None:
        self.state = VoiceState.IDLE

    def on_event(self, event: VoiceEvent) -> None:
        if event.kind == "transcript_final":
            self.state = VoiceState.THINKING
        elif event.kind in {"display_ready", "speaker_ready"}:
            if self.state != VoiceState.SPEAKING:
                self.state = VoiceState.THINKING
        elif event.kind == "tts_started":
            self.state = VoiceState.SPEAKING
        elif event.kind == "tts_finished":
            self.state = VoiceState.LISTENING
        elif event.kind == "error":
            self.state = VoiceState.ERROR
        elif event.kind == "session_started":
            self.state = VoiceState.LISTENING
