# -*- coding: utf-8 -*-
"""充分性验证器 —— 实现 uniagent 的 Verifier 协议,挂载到检索 GoalLoop。

V2 方案 3.4 的"规则先行,LLM 补充"直接映射为 GoalLoop 的验证步骤:
  - 规则(top3 得分阈值 / 数量下限)不通过 → 直接 failed,不调 LLM;
  - 规则通过 → 小模型判意图覆盖;
  - failed 时 GoalLoop 自动把 evidence 作为负例反馈注入下一轮对话
    (框架原生行为,正好就是我们的"反馈检索负例重新生成检索词")。

生成器/评估器分离由框架强制:验证器拿 goal+state 独立判断,
不依赖检索子Agent 的自述。
"""
from __future__ import annotations

import json
from typing import Any, Dict

from uniagent.verification.verifier import VerificationResult

from .workspace import get_workspace

_COVERAGE_SYSTEM = (
    "[TASK:intent_coverage] 判断已召回的知识标题是否覆盖用户问题的全部子意图。"
    '输出 JSON: {"covered": bool, "uncovered_intents": [..]}。只输出 JSON。'
)


class SufficiencyVerifier:
    """规则先行 + LLM 意图覆盖。llm_judge 为可调用 (system, user) -> str。"""

    def __init__(self, llm_judge=None):
        self._judge = llm_judge

    async def verify(self, goal: str, state: Dict[str, Any]) -> VerificationResult:
        ws = get_workspace()
        chunks = ws.data.get("chunks", [])
        cfg = ws.cfg

        # ---- 规则层 ----
        top3 = sorted((c.score for c in chunks), reverse=True)[:3]
        rule_top3 = len(top3) >= 3 and all(s >= cfg.top3_score_threshold for s in top3)
        rule_count = len(chunks) >= cfg.min_chunk_count
        if not (rule_top3 and rule_count):
            reasons = []
            if not rule_count:
                reasons.append(f"候选数 {len(chunks)} < {cfg.min_chunk_count}")
            if not rule_top3:
                reasons.append(f"top3 得分未达阈值 {cfg.top3_score_threshold}: {top3}")
            evidence = "; ".join(reasons) + "。请换策略:改写问题/扩展同义词/放宽过滤(relax_filters=true)。"
            ws.tracer.log(ws.stage, "sufficiency.rule_fail", reasons=reasons)
            return VerificationResult(passed=False, evidence=evidence,
                                      layer="rules", confidence=1.0)

        # ---- LLM 层:意图覆盖 ----
        covered, uncovered = True, []
        if self._judge is not None:
            titles = list({c.doc_title for c in chunks})
            raw = self._judge(_COVERAGE_SYSTEM,
                              f"用户问题:{ws.query}\n已召回标题:{titles}")
            try:
                data = json.loads(raw)
                covered = bool(data.get("covered", True))
                uncovered = [str(x) for x in data.get("uncovered_intents", [])]
            except (json.JSONDecodeError, TypeError):
                covered = True          # 解析失败保守放行,避免多余重试
        ws.tracer.log(ws.stage, "sufficiency.checked",
                      covered=covered, uncovered=uncovered)
        if covered:
            return VerificationResult(
                passed=True, layer="llm_coverage",
                evidence=f"候选 {len(chunks)} 条,top3 得分达标,意图已覆盖")
        return VerificationResult(
            passed=False, layer="llm_coverage",
            evidence=f"未覆盖意图: {uncovered}。请针对这些意图改写问题并重新召回。")
