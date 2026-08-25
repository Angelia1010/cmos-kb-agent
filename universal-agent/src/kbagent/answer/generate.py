# -*- coding: utf-8 -*-
"""答案生成:识别目标片段 → 组织答案(内联引用) → 逐句锚定校验 → 渲染输出。

使用标准 model.invoke([SystemMessage, HumanMessage]) 调用 LLM，无需特殊接口。
锚定失败策略:硬事实直接删除;软性表述标注"建议核实"。
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from ..shared.config import Config
from ..shared.models import AnswerSentence, Chunk, FinalAnswer, SourceRef
from ..shared.tracing import Tracer

_ANSWER_SYSTEM = (
    "[TASK:answer] 你是10086坐席辅助助手。基于给定知识片段生成回复,输出 JSON:"
    '{"business_explanation": str, "handling_suggestion": str, '
    '"sentences": [{"text": str, "citations": [chunk_id], "hard_fact": bool}]}。'
    "每个事实性陈述必须携带其依据的 chunk_id。禁止使用片段之外的信息。只输出 JSON。"
)

_ANCHOR_SYSTEM = (
    "[TASK:anchor_check] 判断句子与知识片段是否语义一致,"
    '输出 JSON: {"consistent": bool}。只输出 JSON。'
)


def _parse_json(raw: str) -> Dict[str, Any]:
    raw = re.sub(r"^```(json)?|```$", "", str(raw).strip(), flags=re.M).strip()
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def _invoke_json(model: Any, system: str, user: str) -> Dict[str, Any]:
    """用标准 model.invoke 调用 LLM 并解析 JSON 响应。"""
    resp = model.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    raw = str(getattr(resp, "content", resp))
    return _parse_json(raw)


def select_fragments(query: str, chunks: List[Chunk], top_n: int = 4) -> List[Chunk]:
    """识别目标片段:取 top_n,同文档最多 2 个片段。"""
    selected: List[Chunk] = []
    for c in chunks:
        if len(selected) >= top_n:
            break
        if sum(1 for s in selected if s.doc_id == c.doc_id) >= 2:
            continue
        selected.append(c)
    return selected


def generate(model: Any, query: str, materials: List[Chunk],
             cfg: Config, tracer: Tracer, trace_id: str) -> FinalAnswer:
    """组织答案并执行逐句锚定校验。"""
    tracer.log("answer", "materials", chunk_ids=[c.chunk_id for c in materials])
    material_text = "\n".join(
        f'<chunk id="{c.chunk_id}">{c.content}</chunk>' for c in materials)
    data = _invoke_json(model, _ANSWER_SYSTEM,
                        f"用户问题:{query}\n知识片段:\n{material_text}")

    valid_ids = {c.chunk_id for c in materials}
    by_id = {c.chunk_id: c for c in materials}
    sentences: List[AnswerSentence] = []
    for s in data.get("sentences", []):
        sentences.append(AnswerSentence(
            text=str(s.get("text", "")),
            citations=[str(c) for c in s.get("citations", [])],
            hard_fact=bool(s.get("hard_fact", False)),
        ))

    # ---- 逐句锚定校验 ----
    for sent in sentences:
        real_cites = [c for c in sent.citations if c in valid_ids]
        if not real_cites:
            sent.anchored = False
        else:
            sent.citations = real_cites
            chunk_text = " ".join(by_id[c].content for c in real_cites)
            check = _invoke_json(model, _ANCHOR_SYSTEM,
                                 f"句子:{sent.text}\n片段:{chunk_text}")
            sent.anchored = bool(check.get("consistent", False))
        if not sent.anchored:
            if sent.hard_fact:
                sent.dropped = True
            else:
                sent.note = "建议核实"
        tracer.log("answer", "anchor_check", text=sent.text[:40],
                   citations=sent.citations, hard_fact=sent.hard_fact,
                   anchored=sent.anchored, dropped=sent.dropped)

    kept = [s for s in sentences if not s.dropped]
    expl = str(data.get("business_explanation", ""))
    sugg = str(data.get("handling_suggestion", ""))
    for s in sentences:
        if s.dropped:
            expl = expl.replace(s.text, "").strip()
            sugg = sugg.replace(s.text, "").strip()

    cited_ids: List[str] = []
    for s in kept:
        for c in s.citations:
            if c not in cited_ids:
                cited_ids.append(c)
    stale_before = datetime.now() - timedelta(days=cfg.stale_days)
    sources: List[SourceRef] = []
    for cid in cited_ids:
        c = by_id[cid]
        try:
            stale = datetime.strptime(c.updated_at, "%Y-%m-%d") < stale_before
        except ValueError:
            stale = True
        sources.append(SourceRef(cid, c.doc_title, c.content, c.updated_at, stale))

    return FinalAnswer(
        trace_id=trace_id, query=query,
        business_explanation=expl, handling_suggestion=sugg,
        sentences=kept, sources=sources,
    )
