# -*- coding: utf-8 -*-
"""充分性验证器 — 实现 uniagent Verifier 协议,挂载到检索 GoalLoop。

纯规则层:top3 得分阈值 + 候选数量下限。
验证失败时 GoalLoop 自动把 evidence 作为负例反馈注入下一轮对话。
"""
from __future__ import annotations

from typing import Any, Dict

from uniagent.verification.verifier import VerificationResult

from ..shared.workspace import get_workspace


class SufficiencyVerifier:
    """规则层充分性检验:top3 得分阈值 + 候选数量下限。"""

    async def verify(self, goal: str, state: Dict[str, Any]) -> VerificationResult:
        ws = get_workspace()
        chunks = ws.data.get("chunks", [])
        cfg = ws.cfg

        top3 = sorted((c.score for c in chunks), reverse=True)[:3]
        rule_top3 = len(top3) >= 3 and all(s >= cfg.top3_score_threshold for s in top3)
        rule_count = len(chunks) >= cfg.min_chunk_count

        if not (rule_top3 and rule_count):
            reasons = []
            if not rule_count:
                reasons.append(f"候选数 {len(chunks)} < {cfg.min_chunk_count}")
            if not rule_top3:
                reasons.append(f"top3 得分未达阈值 {cfg.top3_score_threshold}: {top3}")
            evidence = (
                "; ".join(reasons)
                + "。请换策略:改写问题/扩展同义词/放宽过滤(relax_filters=true)。"
            )
            ws.tracer.log(ws.stage, "sufficiency.rule_fail", reasons=reasons)
            return VerificationResult(passed=False, evidence=evidence,
                                      layer="rules", confidence=1.0)

        ws.tracer.log(ws.stage, "sufficiency.passed",
                      count=len(chunks), top3=top3)
        return VerificationResult(
            passed=True, layer="rules",
            evidence=f"候选 {len(chunks)} 条,top3 得分达标")
