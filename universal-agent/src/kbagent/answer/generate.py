# -*- coding: utf-8 -*-
"""答案生成(方案 五,答案子智能体内部实现)。

四步:识别目标片段 → 组织答案(内联引用) → 逐句锚定校验 → 渲染输出。
使用标准 model.invoke([SystemMessage, HumanMessage]) 调用 LLM,无需特殊接口,
任意 BaseChatModel(如灵犀网关的 LingxiSSLChatOpenAI)可直接接入。
锚定失败策略:硬事实(资费/办理条件/生效规则)直接删除;软性表述标注"建议核实"。
"""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from ..shared.config import Config
from ..shared.models import AnswerSentence, Chunk, FinalAnswer, SourceRef
from ..shared.tracing import Tracer

logger = logging.getLogger("kbagent.answer")

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


def _parse_json(raw: str) -> Dict[str, Any]:
    """解析 LLM 输出的 JSON;剥离 markdown 代码围栏,失败返回空字典。"""
    raw = re.sub(r"^```(json)?|```$", "", str(raw).strip(), flags=re.M).strip()
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            logger.warning("answer LLM 输出不是 JSON 对象: %s", raw[:500])
            return {}
        return data
    except json.JSONDecodeError as exc:
        logger.warning("answer LLM 输出 JSON 解析失败 err=%s 原始输出=%s",
                       exc, raw[:500])
        return {}


def _invoke_json(model: Any, system: str, user: str) -> Dict[str, Any]:
    """用标准 model.invoke 调用 LLM 并解析 JSON 响应;记录耗时与原始输出。"""
    task = system.split("]")[0].lstrip("[") if system.startswith("[") else "llm"
    t0 = time.time()
    try:
        resp = model.invoke([SystemMessage(content=system),
                             HumanMessage(content=user)])
    except Exception as exc:  # noqa: BLE001
        logger.warning("answer LLM 调用异常(%s) 耗时%.1fs: %r",
                       task, time.time() - t0, exc)
        raise
    elapsed = time.time() - t0
    raw = str(getattr(resp, "content", resp))
    logger.info("answer LLM 返回(%s) 耗时%.1fs 长度=%d 内容=%s",
                task, elapsed, len(raw), raw[:500].replace("\n", " "))
    if not raw.strip():
        logger.warning("answer LLM(%s) 返回空内容! 耗时%.1fs——检查网关/模型配置",
                       task, elapsed)
    return _parse_json(raw)


def select_fragments(query: str, chunks: List[Chunk], top_n: int = 4) -> List[Chunk]:
    """第一步:识别目标片段(该职责已从数据处理层收归到此)。
    简化实现:处理层已按得分排序,取 top_n 并保证同文档不重复占满素材集。
    """
    selected: List[Chunk] = []
    for c in chunks:
        if len(selected) >= top_n:
            break
        # 同一文档最多 2 个片段,给其他文档留位
        if sum(1 for s in selected if s.doc_id == c.doc_id) >= 2:
            continue
        selected.append(c)
    logger.info("answer 素材精选: 输入 %d 条 → 选中 %d 条 %s",
                len(chunks), len(selected), [c.chunk_id for c in selected])
    if chunks and not selected:
        logger.warning("answer 有候选却选不出素材,请检查处理阶段输出")
    return selected


def generate(model: Any, query: str, materials: List[Chunk],
             cfg: Config, tracer: Tracer, trace_id: str) -> FinalAnswer:
    """组织答案并执行逐句锚定校验;model 为任意标准 BaseChatModel。"""
    t_start = time.time()
    logger.info("answer 开始 query=%r 素材=%d条 ids=%s",
                query, len(materials), [c.chunk_id for c in materials])
    tracer.log("answer", "materials", chunk_ids=[c.chunk_id for c in materials])
    material_text = "\n".join(
        f'<chunk id="{c.chunk_id}">{c.content}</chunk>' for c in materials)
    data = _invoke_json(model, _ANSWER_SYSTEM,
                        f"用户问题:{query}\n知识片段:\n{material_text}")
    if not data:
        logger.warning("answer LLM 未产出有效 JSON → 最终答案将为空 "
                       "(耗时%.1fs, 原始输出见上一条日志)",
                       time.time() - t_start)
    else:
        logger.info("answer LLM JSON 解析成功: sentences=%d "
                    "business_explanation长度=%d handling_suggestion长度=%d",
                    len(data.get("sentences", [])),
                    len(str(data.get("business_explanation", ""))),
                    len(str(data.get("handling_suggestion", ""))))

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
            logger.info("answer 锚定: 引用无效(不在素材内) text=%r citations=%s",
                        sent.text[:40], sent.citations)
        else:
            sent.citations = real_cites
            # 3b. 语义一致性(小模型判句-片段一致性)
            chunk_text = " ".join(by_id[c].content for c in real_cites)
            check = _invoke_json(model, _ANCHOR_SYSTEM,
                                 f"句子:{sent.text}\n片段:{chunk_text}")
            sent.anchored = bool(check.get("consistent", False))
        if not sent.anchored:
            if sent.hard_fact:
                sent.dropped = True            # 硬事实零容忍:直接删除
            else:
                sent.note = "建议核实"          # 软性表述降级标注
        logger.info("answer 锚定结果: anchored=%s dropped=%s hard_fact=%s text=%r",
                    sent.anchored, sent.dropped, sent.hard_fact, sent.text[:40])
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

    logger.info("answer 完成: 总耗时%.1fs 句子=%d(保留%d) 来源=%d "
                "业务说明长度=%d 办理建议长度=%d",
                time.time() - t_start, len(sentences), len(kept),
                len(sources), len(expl), len(sugg))
    if not expl and not sugg:
        logger.warning("answer 最终业务说明/办理建议均为空! 总耗时%.1fs "
                       "——按序回看上面日志:素材是否为0 / LLM是否空返回 / "
                       "JSON是否解析失败 / 句子是否全被锚定删除",
                       time.time() - t_start)
    return FinalAnswer(
        trace_id=trace_id, query=query,
        business_explanation=expl, handling_suggestion=sugg,
        sentences=kept, sources=sources,
    )
