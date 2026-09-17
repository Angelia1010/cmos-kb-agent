# -*- coding: utf-8 -*-
"""答案生成(方案 五,答案子智能体内部实现)。

流程:识别目标片段 → 组织话术+办理建议 → 一次批量话术一致性校验 → 组装来源。
使用标准 model.invoke([SystemMessage, HumanMessage]) 调用 LLM,无需特殊接口,
任意 BaseChatModel(如灵犀网关的 LingxiSSLChatOpenAI)可直接接入。

LLM 调用次数 = 1(答案组织)+ 1(批量一致性校验);相关度与关键片段由
locate_fragments 阶段顺带产出(见 .locate),此处零额外调用。
一致性校验失败不删话术,改为收紧 usability(至少 verify_first)并列出问题,
由坐席对照原文核实 —— 保证坐席永远有话术可看。
"""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from ..shared.config import Config
from ..shared.models import (
    USABILITY_DIRECT,
    USABILITY_NOT,
    USABILITY_VERIFY,
    Chunk,
    DocFragments,
    FinalAnswer,
    SourceRef,
    Usability,
)
from ..shared.tracing import Tracer

logger = logging.getLogger("kbagent.answer")

_VALID_LEVELS = (USABILITY_DIRECT, USABILITY_VERIFY, USABILITY_NOT)

_ANSWER_SYSTEM = (
    "[TASK:answer] 你是10086坐席辅助助手。基于给定知识片段生成坐席答复内容,输出 JSON:"
    '{"script": str(可直接念给用户的口语化完整话术,以“您好”开头,自然流畅,'
    "把必要的注意事项/前提条件自然融入话术,不要遗漏资费、时限等关键数字), "
    '"handling_suggestion": str(给坐席的办理建议:怎么办、需要什么材料、'
    '需与客户确认什么、有什么限制条件), '
    '"usability": {"level": "directly_usable|verify_first|not_usable", '
    '"reasons": [str](判定依据), "uncovered": [str](用户问题中知识片段未覆盖的方面)}}。'
    "话术与建议中的每个事实必须能在知识片段中找到依据,禁止使用片段之外的信息。"
    "只输出 JSON,不要输出 sentences/citations 等其他字段。"
    "usability 必须诚实自评:片段足以完整回答且信息明确→directly_usable;"
    "片段只覆盖部分问题、信息含糊或相互矛盾→verify_first 并在 uncovered 列出未覆盖方面;"
    "片段与问题基本无关→not_usable。禁止为了好看而拔高 level。"
    "字符串值内禁止出现未转义的英文双引号:引用词语请改用中文引号“”,"
    '或写成 \\" ;值内禁止换行。'
)

# 批量话术一致性校验:整段话术 vs 全部素材,一次调用替代逐句锚定。
# 标记仍用 [TASK:anchor_check](离线 ScriptedChatModel 按此标记分发)。
_CONSISTENCY_SYSTEM = (
    '[TASK:anchor_check] 判断坐席话术与知识片段是否一致:话术中每个事实性表述'
    '(资费数字、办理条件、渠道、时限等)都能在片段中找到依据且不与片段矛盾。'
    '输出 JSON: {"consistent": bool, "issues": [str](话术中缺乏依据或与片段矛盾的'
    '表述,逐条简述,每条不超过40字;完全一致时为空数组)}。只输出 JSON。'
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
             cfg: Config, tracer: Tracer, trace_id: str,
             matched: Optional[List[DocFragments]] = None) -> FinalAnswer:
    """组织话术+办理建议,并做一次批量话术一致性校验。

    model 为任意标准 BaseChatModel;matched 为 locate 阶段对每篇文档的
    证据片段定位结果(相关度与关键片段的来源,此处零额外 LLM 调用)。
    """
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
    script = str(data.get("script", "") or "").strip()
    suggestion = str(data.get("handling_suggestion", "") or "").strip()
    if data:
        logger.info("answer LLM JSON 解析成功: 话术长度=%d 办理建议长度=%d",
                    len(script), len(suggestion))

    # ---- 批量话术一致性校验(一次调用替代逐句锚定)----
    consistent: Optional[bool] = None
    issues: List[str] = []
    if script and materials:
        check = _invoke_json(
            model, _CONSISTENCY_SYSTEM,
            f"坐席话术:{script}\n知识片段:\n{material_text}")
        consistent = bool(check.get("consistent", False)) if check else False
        raw_issues = check.get("issues") if check else None
        if isinstance(raw_issues, list):
            issues = [str(x).strip()[:80] for x in raw_issues
                      if str(x).strip()][:5]
        tracer.log("answer", "consistency_check", consistent=consistent,
                   issue_count=len(issues), issues=issues)
        logger.info("answer 一致性校验: consistent=%s issues=%s",
                    consistent, issues)

    # ---- 来源列表:相关度(最相关=100)+ 关键片段 + 全文 + 过旧提示 ----
    # sources = 全部精选素材:LLM 输出解析失败时坐席仍有原文可核对。
    sources = _build_sources(materials, matched, cfg)

    # ---- 可用性判定(LLM 自评 + 确定性规则纠偏,规则只收紧) ----
    usability = _assess_usability(
        data, sources, materials,
        has_content=bool(script or suggestion),
        consistent=consistent, issues=issues)

    logger.info("answer 完成: 总耗时%.1fs 话术长度=%d 办理建议长度=%d 来源=%d "
                "可用性=%s 一致性=%s",
                time.time() - t_start, len(script), len(suggestion),
                len(sources), usability.level, consistent)
    if usability.reasons:
        logger.info("answer 可用性依据: %s", usability.reasons)
    if usability.uncovered:
        logger.info("answer 未覆盖方面: %s", usability.uncovered)
    if not script and not suggestion:
        logger.warning("answer 最终话术/办理建议均为空! 总耗时%.1fs "
                       "——按序回看上面日志:素材是否为0 / LLM是否空返回 / "
                       "JSON是否解析失败",
                       time.time() - t_start)
    return FinalAnswer(
        trace_id=trace_id, query=query,
        script=script, handling_suggestion=suggestion,
        sources=sources, usability=usability,
    )


def _build_sources(materials: List[Chunk],
                   matched: Optional[List[DocFragments]],
                   cfg: Config) -> List[SourceRef]:
    """组装来源:相关度归一化(最相关一篇=100)+ 关键片段 + 整篇原文。

    相关度优先取 locate 阶段模型自评(DocFragments.relevance);
    缺失时兜底:有可验证片段的文档给中位分,其余按 0;
    全部无信号时按素材顺序兜底(处理阶段已按重排得分排序),
    保证"最相关 = 100%"不变式恒成立。
    """
    matched_by_id = {d.chunk_id: d for d in (matched or [])}
    stale_before = datetime.now() - timedelta(days=cfg.stale_days)

    raws: List[float] = []
    for c in materials:
        df = matched_by_id.get(c.chunk_id)
        if df is not None and df.relevance > 0:
            raws.append(float(df.relevance))
        elif df is not None and df.answerable:
            raws.append(40.0)     # 有可验证片段但模型未给相关度
        else:
            raws.append(0.0)
    if materials and max(raws) <= 0:
        raws = [float(len(materials) - i) for i in range(len(materials))]
    peak = max(raws) if raws else 0.0

    rows: List[SourceRef] = []
    for i, c in enumerate(materials):
        df = matched_by_id.get(c.chunk_id)
        key_fragment = df.fragments[0].text if df and df.fragments else ""
        relevance = int(round(raws[i] / peak * 100)) if peak > 0 else 0
        try:
            stale = datetime.strptime(c.updated_at, "%Y-%m-%d") < stale_before
        except ValueError:
            stale = True
        rows.append(SourceRef(chunk_id=c.chunk_id, doc_id=c.doc_id,
                              doc_title=c.doc_title, relevance=relevance,
                              key_fragment=key_fragment, content=c.content,
                              updated_at=c.updated_at, stale=stale))
    # 按相关度降序(稳定排序,同分保持重排原序)
    rows.sort(key=lambda s: -s.relevance)
    return rows


def _assess_usability(data: Dict[str, Any], sources: List[SourceRef],
                      materials: List[Chunk], has_content: bool,
                      consistent: Optional[bool] = None,
                      issues: Optional[List[str]] = None) -> Usability:
    """可用性判定:以 LLM 自评为基线,用确定性规则收紧(只降不升)。

    规则层覆盖 LLM 自评不可信/缺失的场景:
      素材为空、JSON 解析失败、内容为空      → not_usable
      一致性校验不过、知识过旧、存在未覆盖    → verify_first
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
    if consistent is False:
        u.tighten(USABILITY_VERIFY,
                  f"话术一致性校验未通过({len(issues or [])} 处表述需核实),"
                  "请对照原文后再答复")
        for it in (issues or [])[:3]:
            u.tighten(USABILITY_VERIFY, f"待核实:{it}")
    stale_n = sum(1 for s in sources if s.stale)
    if stale_n:
        u.tighten(USABILITY_VERIFY,
                  f"{stale_n} 条引用知识已过旧,请核实最新政策")
    if u.uncovered:
        u.tighten(USABILITY_VERIFY, "问题存在知识库未覆盖的方面,见未覆盖列表")
    return u
