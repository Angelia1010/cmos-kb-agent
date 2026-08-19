# -*- coding: utf-8 -*-
"""全链路 trace(方案 六):trace_id 贯穿,badcase 可完整回放。

每个关键节点写入一条事件:归一化 query、缓存判定、每轮检索参数/DSL/召回、
充分性判据判定值、数据处理路由与前后快照、生成素材 chunk_id、
锚定校验逐句结果、最终答案。export() 输出 JSON 供离线分析。

生产接入:把 export 的 JSON 落到日志管道/OLAP 即可;若使用 LangSmith,
LangGraph 节点级 trace 会自动上报,本模块负责业务语义层的补充记录。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

from .models import new_id


@dataclass
class TraceEvent:
    ts_ms: int
    stage: str          # normalize / cache / retrieval.round1 / processing / answer ...
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
                {"ts_ms": e.ts_ms, "stage": e.stage, "event": e.event, "payload": e.payload}
                for e in self.events
            ],
        }, ensure_ascii=False, indent=2, default=str)
