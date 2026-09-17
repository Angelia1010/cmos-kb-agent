# -*- coding: utf-8 -*-
"""充分性验证器 —— 实现 uniagent 的 Verifier 协议,挂载到检索 GoalLoop。

当前为"固定三轮"模式:验证器不发挥原规则/LLM 判定作用,仅用计数器
控制 GoalLoop 固定执行三轮工具调用,对应预期流程:
  intergrate_all → query_rewrite → intergrate_all
  - 第 1 次调用 verify → passed=False(首轮召回后,驱动进入重写轮);
  - 第 2 次调用 verify → passed=False(重写完成,驱动进入二次召回轮);
  - 第 3 次调用 verify → passed=True(二次召回完成,loop 通过结束)。
生成器/评估器分离由框架强制:验证器拿 goal+state 独立判断。
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from uniagent.verification.verifier import VerificationResult

from ..shared.workspace import get_workspace

logger = logging.getLogger("kbagent.retrieval")

# _COVERAGE_SYSTEM = (
#     "[TASK:intent_coverage] 判断已召回的知识标题是否覆盖用户问题的全部子意图。"
#     '输出 JSON: {"covered": bool, "uncovered_intents": [..]}。只输出 JSON。'
# )


class SufficiencyVerifier:
    """固定三轮模式:第 1、2 次 verify 返回不通过,第 3 次返回通过。
    对应流程:intergrate_all → query_rewrite → intergrate_all。
    原 llm_judge 参数保留仅为调用方兼容,当前模式下不参与判定。"""

    def __init__(self, llm_judge=None):
        self._judge = llm_judge
        self._verify_count = 0

    async def verify(self, goal: str, state: Dict[str, Any]) -> VerificationResult:
        self._verify_count += 1
        ws = get_workspace()
        if self._verify_count < 3:
            ws.tracer.log(ws.stage, "sufficiency.fixed_round_fail",
                          round=self._verify_count,
                          reason=f"固定三轮模式:第{self._verify_count}轮强制不通过")
            passed = False
            evidence = f"固定三轮模式:第{self._verify_count}轮强制不通过,请继续下一轮检索。"
            logger.info("[VERIFY_ROUND] round=%d passed=%s evidence=%s",
                        self._verify_count, passed, evidence)
            return VerificationResult(
                passed=passed,
                evidence=evidence,
                layer="fixed_rounds", confidence=1.0)
        ws.tracer.log(ws.stage, "sufficiency.fixed_round_pass",
                      round=self._verify_count,
                      reason="固定三轮模式:第3轮强制通过")
        passed = True
        evidence = "固定三轮模式:第3轮强制通过,检索结束。"
        logger.info("[VERIFY_ROUND] round=%d passed=%s evidence=%s",
                    self._verify_count, passed, evidence)
        return VerificationResult(
            passed=passed,
            evidence=evidence,
            layer="fixed_rounds", confidence=1.0)

# class SufficiencyVerifier:
#     """规则先行 + LLM 意图覆盖。llm_judge 为可调用 (system, user) -> str;
#     不传则退化为纯规则层(离线/无判据模型场景)。"""
#
#     def __init__(self, llm_judge: Optional[Callable[[str, str], str]] = None):
#         self._judge = llm_judge
#
#     async def verify(self, goal: str, state: Dict[str, Any]) -> VerificationResult:
#         ws = get_workspace()
#         chunks = ws.data.get("chunks", [])
#         cfg = ws.cfg
#
#         # ---- 规则层 ----
#         top3 = sorted((c.score for c in chunks), reverse=True)[:3]
#         rule_top3 = len(top3) >= 3 and all(s >= cfg.top3_score_threshold for s in top3)
#         rule_count = len(chunks) >= cfg.min_chunk_count
#         if not (rule_top3 and rule_count):
#             reasons = []
#             if not rule_count:
#                 reasons.append(f"候选数 {len(chunks)} < {cfg.min_chunk_count}")
#             if not rule_top3:
#                 reasons.append(f"top3 得分未达阈值 {cfg.top3_score_threshold}: {top3}")
#             evidence = "; ".join(reasons) + "。请换策略:改写问题/扩展同义词/放宽过滤(relax_filters=true)。"
#             ws.tracer.log(ws.stage, "sufficiency.rule_fail", reasons=reasons)
#             return VerificationResult(passed=False, evidence=evidence,
#                                       layer="rules", confidence=1.0)
#
#         # ---- LLM 层:意图覆盖(未配判据模型则跳过) ----
#         covered, uncovered = True, []
#         if self._judge is not None:
#             titles = list({c.doc_title for c in chunks})
#             raw = self._judge(_COVERAGE_SYSTEM,
#                               f"用户问题:{ws.query}\n已召回标题:{titles}")
#             try:
#                 data = json.loads(raw)
#                 covered = bool(data.get("covered", True))
#                 uncovered = [str(x) for x in data.get("uncovered_intents", [])]
#             except (json.JSONDecodeError, TypeError):
#                 covered = True          # 解析失败保守放行,避免多余重试
#             ws.tracer.log(ws.stage, "sufficiency.llm_coverage",
#                           covered=covered, uncovered=uncovered)
#         ws.tracer.log(ws.stage, "sufficiency.passed" if covered
#                       else "sufficiency.coverage_fail",
#                       count=len(chunks), top3=top3)
#         if covered:
#             layer = "llm_coverage" if self._judge is not None else "rules"
#             return VerificationResult(
#                 passed=True, layer=layer, confidence=1.0,
#                 evidence=f"候选 {len(chunks)} 条,top3 得分达标,意图已覆盖")
#         return VerificationResult(
#             passed=False, layer="llm_coverage",
#             evidence=f"未覆盖意图: {uncovered}。请针对这些意图改写问题并重新召回。")
