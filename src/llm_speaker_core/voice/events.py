from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass
class VoiceEvent:
    kind: str
    session_id: str
    text: str = ""
    meta: dict[str, object] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


class VoiceEventSink(Protocol):
    def emit(self, event: VoiceEvent) -> None: ...


class JsonlVoiceEventSink:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: VoiceEvent) -> None:
        with self.path.open("a", encoding="utf-8", buffering=1) as f:
            f.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")


class CompositeVoiceEventSink:
    def __init__(self, *sinks: VoiceEventSink) -> None:
        self.sinks = [sink for sink in sinks if sink is not None]

    def emit(self, event: VoiceEvent) -> None:
        for sink in self.sinks:
            sink.emit(event)

