# -*- coding: utf-8 -*-
"""全链路 trace:trace_id 贯穿,badcase 可完整回放。"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

from .models import new_id


@dataclass
class TraceEvent:
    ts_ms: int
    stage: str
    event: str
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Tracer:
    trace_id: str = field(default_factory=lambda: new_id("trace"))
    started_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    events: List[TraceEvent] = field(default_factory=list)

    def log(self, stage: str, event: str, **payload: Any) -> None:
        self.events.append(TraceEvent(int(time.time() * 1000), stage, event, payload))

    def elapsed_ms(self) -> int:
        return int(time.time() * 1000) - self.started_ms

    def export(self) -> str:
        return json.dumps({
            "trace_id": self.trace_id,
            "started_ms": self.started_ms,
            "elapsed_ms": self.elapsed_ms(),
            "events": [
                {"ts_ms": e.ts_ms, "stage": e.stage,
                 "event": e.event, "payload": e.payload}
                for e in self.events
            ],
        }, ensure_ascii=False, indent=2, default=str)
