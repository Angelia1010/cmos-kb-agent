# -*- coding: utf-8 -*-
"""答案生成(方案 五,主Agent 内部,不涉及子Agent)。

四步:识别目标片段 → 组织答案(内联引用) → 逐句锚定校验 → 渲染输出。
锚定失败策略:硬事实(资费/办理条件/生效规则)直接删除;软性表述标注"建议核实"。
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import List

from .config import Config
from .models import AnswerSentence, Chunk, FinalAnswer, SourceRef
from .tracing import Tracer

_ANSWER_SYSTEM = (
    "[TASK:answer] 你是10086坐席辅助助手。基于给定知识片段生成回复,输出 JSON:"
    '{"business_explanation": str, "handling_suggestion": str, '
    '"sentences": [{"text": str, "citations": [chunk_id], "hard_fact": bool}]}。'
    "每个事实性陈述必须携带其依据的 chunk_id。禁止使用片段之外的信息。只输出 JSON。"
)

_ANCHOR_SYSTEM = (
    '[TASK:anchor_check] 判断句子与知识片段是否语义一致,'
    '输出 JSON: {"consistent": bool}。只输出 JSON。'
)


def select_fragments(query: str, chunks: List[Chunk], top_n: int = 4) -> List[Chunk]:
    """第一步:识别目标片段(该职责已从数据处理层收归到此)。
    简化实现:处理层已按得分排序,取 top_n 并保证同文档不重复占满素材集。
    """
    selected: List[Chunk] = []
    doc_seen = set()
    for c in chunks:
        if len(selected) >= top_n:
            break
        # 同一文档最多 2 个片段,给其他文档留位
        if sum(1 for s in selected if s.doc_id == c.doc_id) >= 2:
            continue
        selected.append(c)
        doc_seen.add(c.doc_id)
    return selected


def generate(llm, query: str, materials: List[Chunk],
             cfg: Config, tracer: Tracer, trace_id: str) -> FinalAnswer:
    """llm 需提供 large_json(system, user) 与 small_json(system, user) 两个方法。"""
    tracer.log("answer", "materials", chunk_ids=[c.chunk_id for c in materials])
    material_text = "\n".join(
        f'<chunk id="{c.chunk_id}">{c.content}</chunk>' for c in materials)
    data = llm.large_json(_ANSWER_SYSTEM,
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

    # ---- 第三步:逐句锚定校验 ----
    for sent in sentences:
        # 3a. 引用必须指向真实存在的片段
        real_cites = [c for c in sent.citations if c in valid_ids]
        if not real_cites:
            sent.anchored = False
        else:
            sent.citations = real_cites
            # 3b. 语义一致性(小模型/NLI;Mock 用重叠率)
            chunk_text = " ".join(by_id[c].content for c in real_cites)
            check = llm.small_json(_ANCHOR_SYSTEM,
                                   f"句子:{sent.text}\n片段:{chunk_text}")
            sent.anchored = bool(check.get("consistent", False))
        if not sent.anchored:
            if sent.hard_fact:
                sent.dropped = True            # 硬事实零容忍:直接删除
            else:
                sent.note = "建议核实"          # 软性表述降级标注
        tracer.log("answer", "anchor_check", text=sent.text[:40],
                   citations=sent.citations, hard_fact=sent.hard_fact,
                   anchored=sent.anchored, dropped=sent.dropped)

    kept = [s for s in sentences if not s.dropped]

    # 重组两段式输出:删句后按剩余句子重建,保证与逐句结果一致
    expl = str(data.get("business_explanation", ""))
    sugg = str(data.get("handling_suggestion", ""))
    for s in sentences:
        if s.dropped:
            expl = expl.replace(s.text, "").strip()
            sugg = sugg.replace(s.text, "").strip()

    # ---- 第四步:来源列表 + 过旧提示 ----
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
