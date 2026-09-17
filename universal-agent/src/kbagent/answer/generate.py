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
from ..shared.models import (
    USABILITY_DIRECT,
    USABILITY_NOT,
    USABILITY_VERIFY,
    AnswerSentence,
    Chunk,
    FinalAnswer,
    SourceRef,
    Usability,
)
from ..shared.tracing import Tracer

logger = logging.getLogger("kbagent.answer")

_VALID_LEVELS = (USABILITY_DIRECT, USABILITY_VERIFY, USABILITY_NOT)

_ANSWER_SYSTEM = (
    "[TASK:answer] 你是10086坐席辅助助手。基于给定知识片段生成坐席答复内容,输出 JSON:"
    '{"business_explanation": str(业务说明), "handling_suggestion": str(办理建议), '
    '"sentences": [{"text": str, "citations": [chunk_id], "hard_fact": bool}], '
    '"direct_conclusion": str(一句话直接结论:能不能办/多少钱/怎么办), '
    '"key_elements": {"渠道": str, "材料": str, "条件": str, "时限": str, "资费": str}'
    "(办理要素,仅保留片段中有依据的键,无信息的键整个省略), "
    '"script": str(可直接念给用户的口语化完整话术,以“您好”开头,自然流畅), '
    '"caveats": [str](坐席答复前需注意/主动说明的事项), '
    '"usability": {"level": "directly_usable|verify_first|not_usable", '
    '"reasons": [str](判定依据), "uncovered": [str](用户问题中知识片段未覆盖的方面)}}。'
    "每个事实性陈述必须携带其依据的 chunk_id。禁止使用片段之外的信息。只输出 JSON。"
    "usability 必须诚实自评:片段足以完整回答且信息明确→directly_usable;"
    "片段只覆盖部分问题、信息含糊或相互矛盾→verify_first 并在 uncovered 列出未覆盖方面;"
    "片段与问题基本无关→not_usable。禁止为了好看而拔高 level。"
    "字符串值内禁止出现未转义的英文双引号:引用词语请改用中文引号“”,"
    '或写成 \\" ;值内禁止换行。'
)

_ANCHOR_SYSTEM = (
    '[TASK:anchor_check] 判断句子与知识片段是否语义一致,'
    '输出 JSON: {"consistent": bool}。只输出 JSON。'
)


def _repair_json(raw: str) -> str:
    """尽力修复 LLM 输出的坏 JSON(仅当严格解析失败后调用)。

    针对三种最常见劣化:
    1. 字符串值内未转义的英文双引号(如 通过"中国移动APP"办理)——
       引号后若跟的不是 JSON 结构符(``,`` ``}`` ``]`` ``:``),视为内容并转义;
    2. 字符串值内的裸换行/制表符——转义为 \\n / \\t;
    3. 对象/数组收尾前的多余逗号——删除。
    修复是启发式的,不保证百分百还原,但可覆盖绝大多数引号类劣化。
    """
    out: List[str] = []
    in_str = False
    i, n = 0, len(raw)
    while i < n:
        ch = raw[i]
        if not in_str:
            out.append(ch)
            if ch == '"':
                in_str = True
            i += 1
            continue
        # —— 字符串内部 ——
        if ch == "\\":                       # 已有转义序列原样保留
            out.append(ch)
            if i + 1 < n:
                out.append(raw[i + 1])
                i += 2
            else:
                i += 1
            continue
        if ch == '"':
            j = i + 1                        # 向后看首个非空白字符
            while j < n and raw[j] in " \t\r\n":
                j += 1
            if j >= n or raw[j] in ",}]:":   # 是真正的收尾引号
                out.append(ch)
                in_str = False
            else:                            # 值内裸引号 → 转义
                out.append('\\"')
            i += 1
            continue
        if ch == "\n":
            out.append("\\n"); i += 1; continue
        if ch == "\r":
            out.append("\\r"); i += 1; continue
        if ch == "\t":
            out.append("\\t"); i += 1; continue
        out.append(ch)
        i += 1
    repaired = "".join(out)
    return re.sub(r",\s*([}\]])", r"\1", repaired)   # 去尾逗号


def _parse_json(raw: str) -> Dict[str, Any]:
    """解析 LLM 输出的 JSON;剥离 markdown 代码围栏,失败返回空字典。

    严格解析失败后按序尝试两级兜底:
    a) 截取首个 { 到末个 } 的子串(剥掉模型多嘴的前后说明文字);
    b) _repair_json 启发式修复(值内裸引号/裸换行/尾逗号)。
    """
    raw = re.sub(r"^```(json)?|```$", "", str(raw).strip(), flags=re.M).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as first_exc:
        data = None
        candidates: List[str] = []
        s, e = raw.find("{"), raw.rfind("}")
        if s != -1 and e > s:
            candidates.append(raw[s:e + 1])          # a) 截取 {...}
        candidates.append(_repair_json(raw))          # b) 启发式修复
        if s != -1 and e > s:
            candidates.append(_repair_json(raw[s:e + 1]))  # a+b 组合
        for cand in candidates:
            try:
                data = json.loads(cand)
                logger.info("answer LLM 输出 JSON 已修复解析成功(原错误: %s)",
                            first_exc)
                break
            except json.JSONDecodeError:
                continue
        if data is None:
            logger.warning("answer LLM 输出 JSON 解析失败(含修复尝试) err=%s "
                           "原始输出=%s", first_exc, raw[:500])
            return {}
    if not isinstance(data, dict):
        logger.warning("answer LLM 输出不是 JSON 对象: %s", raw[:500])
        return {}
    return data


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
        # 3a. 引用必须指向真实存在的片段;无效引用一律清除,
        #     防止幻觉 chunk_id 流入下游 sources 构建(by_id KeyError)
        real_cites = [c for c in sent.citations if c in valid_ids]
        if not real_cites:
            sent.anchored = False
            logger.info("answer 锚定: 引用无效(不在素材内) text=%r citations=%s",
                        sent.text[:40], sent.citations)
            sent.citations = []
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
    conclusion = str(data.get("direct_conclusion", "") or "")
    script = str(data.get("script", "") or "")
    for s in sentences:
        if s.dropped:
            expl = expl.replace(s.text, "").strip()
            sugg = sugg.replace(s.text, "").strip()
            conclusion = conclusion.replace(s.text, "").strip()
            script = script.replace(s.text, "").strip()

    # ---- 坐席向结构化内容(容忍模型缺字段:旧格式输出走默认值) ----
    raw_elements = data.get("key_elements")
    key_elements = ({str(k): str(v).strip() for k, v in raw_elements.items()
                     if str(v).strip()}
                    if isinstance(raw_elements, dict) else {})
    raw_caveats = data.get("caveats")
    caveats = ([str(x).strip() for x in raw_caveats if str(x).strip()]
               if isinstance(raw_caveats, list) else [])

    # ---- 第四步:来源列表 + 过旧提示 ----
    cited_ids: List[str] = []
    for s in kept:
        for c in s.citations:
            if c not in cited_ids:
                cited_ids.append(c)
    stale_before = datetime.now() - timedelta(days=cfg.stale_days)
    sources: List[SourceRef] = []
    for cid in cited_ids:
        c = by_id.get(cid)
        if c is None:          # 双保险:引用不在素材内直接跳过
            continue
        try:
            stale = datetime.strptime(c.updated_at, "%Y-%m-%d") < stale_before
        except ValueError:
            stale = True
        sources.append(SourceRef(cid, c.doc_title, c.content, c.updated_at, stale))

    # ---- 第五步:可用性判定(LLM 自评 + 确定性规则纠偏,规则只收紧) ----
    usability = _assess_usability(
        data, sentences, sources, materials,
        has_content=bool(expl or sugg or script or conclusion))

    logger.info("answer 完成: 总耗时%.1fs 句子=%d(保留%d) 来源=%d "
                "业务说明长度=%d 办理建议长度=%d 话术长度=%d 可用性=%s",
                time.time() - t_start, len(sentences), len(kept),
                len(sources), len(expl), len(sugg), len(script),
                usability.level)
    if usability.reasons:
        logger.info("answer 可用性依据: %s", usability.reasons)
    if usability.uncovered:
        logger.info("answer 未覆盖方面: %s", usability.uncovered)
    if not expl and not sugg:
        logger.warning("answer 最终业务说明/办理建议均为空! 总耗时%.1fs "
                       "——按序回看上面日志:素材是否为0 / LLM是否空返回 / "
                       "JSON是否解析失败 / 句子是否全被锚定删除",
                       time.time() - t_start)
    return FinalAnswer(
        trace_id=trace_id, query=query,
        business_explanation=expl, handling_suggestion=sugg,
        sentences=kept, sources=sources,
        direct_conclusion=conclusion, key_elements=key_elements,
        script=script, caveats=caveats, usability=usability,
    )


def _assess_usability(data: Dict[str, Any], sentences: List[AnswerSentence],
                      sources: List[SourceRef], materials: List[Chunk],
                      has_content: bool) -> Usability:
    """可用性判定:以 LLM 自评为基线,用确定性规则收紧(只降不升)。

    规则层覆盖 LLM 自评不可信/缺失的场景:
      素材为空、JSON 解析失败、内容为空          → not_usable
      硬事实句被锚定删除、全部句子未锚定、
      引用知识过旧、存在未覆盖方面                → verify_first
    """
    raw = data.get("usability") if isinstance(data.get("usability"), dict) else {}
    level = raw.get("level")
    u = Usability(
        level=level if level in _VALID_LEVELS else USABILITY_VERIFY,
        reasons=[str(r).strip() for r in raw.get("reasons", [])
                 if str(r).strip()] if isinstance(raw.get("reasons"), list) else [],
        uncovered=[str(x).strip() for x in raw.get("uncovered", [])
                   if str(x).strip()] if isinstance(raw.get("uncovered"), list) else [],
    )
    if data and level not in _VALID_LEVELS:
        u.reasons.append("模型未输出可用性自评,默认按核实后使用处理")

    # ---- 确定性规则(只收紧、不放宽)----
    if not materials:
        u.tighten(USABILITY_NOT, "无可用知识素材,无法支撑答复")
    if not data:
        u.tighten(USABILITY_NOT, "答案生成失败(模型输出无法解析)")
    if data and not has_content:
        u.tighten(USABILITY_NOT, "答案内容为空")
    dropped_hard = sum(1 for s in sentences if s.dropped)
    if dropped_hard:
        u.tighten(USABILITY_VERIFY,
                  f"{dropped_hard} 条硬事实句未通过锚定校验已删除,相关内容缺失")
    if sentences and all(not s.anchored for s in sentences):
        u.tighten(USABILITY_VERIFY, "所有句子均未通过锚定校验")
    stale_n = sum(1 for s in sources if s.stale)
    if stale_n:
        u.tighten(USABILITY_VERIFY,
                  f"{stale_n} 条引用知识已过旧,请核实最新政策")
    if u.uncovered:
        u.tighten(USABILITY_VERIFY, "问题存在知识库未覆盖的方面,见未覆盖列表")
    return u
