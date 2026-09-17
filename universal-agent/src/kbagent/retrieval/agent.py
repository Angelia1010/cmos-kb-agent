# -*- coding: utf-8 -*-
"""检索子智能体 — GoalLoop 三轮形态(intergrate_all→query_rewrite→intergrate_all)。

RetrievalSubAgent 通过 create_agent 创建 GoalLoop,固定三轮迭代:
  轮1:intergrate_all(原始 query 召回)
  轮2:query_rewrite(LLM 重写查询)
  轮3:intergrate_all(改写 query 二次召回)
SufficiencyVerifier 固定三轮判定,第3轮返回 passed=True 结束循环。
"""
from __future__ import annotations

import json
import logging
from typing import Any, List

from uniagent import AgentFeatures, Budget, BudgetConfig, create_agent

from ..shared.config import Config
from ..shared.models import Chunk
from ..shared.tracing import Tracer
from ..shared.workspace import get_workspace
from .sufficiency import SufficiencyVerifier
from .tools import RETRIEVAL_TOOLS
from .prompt import RETRIEVAL_GOAL, RETRIEVAL_SYSTEM_PROMPT

logger = logging.getLogger("kbagent.retrieval")

# 单条召回率人工核对用:写死的正确知识ID(Postman 发单条请求时与实际检索 kid 比对)。
# 切换测试问题时,直接改这里为测试集对应条目的 knowledge_ids。
EXPECTED_KIDS: List[str] = [
    "1812111727130255978",
    "f31970797e99455dbc9c4698f7edac79",
]


class RetrievalSubAgent:
    """检索候选知识子智能体 — GoalLoop 三轮形态。"""

    def __init__(self, model: Any, cfg: Config, tracer: Tracer):
        self.model = model
        self.cfg = cfg
        self.tracer = tracer
        self._verifier = SufficiencyVerifier()

    async def run(self, query: str, region_code: str = "000") -> List[Chunk]:
        ws = get_workspace()
        ws.stage = "retrieval"
        ws.data["region_code"] = region_code
        loop = create_agent(
            model=self.model,
            tools=RETRIEVAL_TOOLS,
            features=AgentFeatures(skill=False),
            system_prompt=RETRIEVAL_SYSTEM_PROMPT,
            goal=RETRIEVAL_GOAL,
            verifier=self._verifier,
            budget=Budget(config=BudgetConfig(
                max_iterations=self.cfg.max_retrieval_rounds,
                max_time_seconds=self.cfg.budget["retrieval_total"] / 1000.0,
            )),
            name="retrieval_subagent",
        )
        input_messages = [{"role": "user",
                           "content": f"用户问题:{query}\nregion_code:{region_code}"}]
        logger.info("[AGENT_INPUT] input_messages=%s",
                    json.dumps(input_messages, ensure_ascii=False))
        result = await loop.run(
            input_messages=input_messages,
            thread_id=self.tracer.trace_id,
        )
        logger.info("[AGENT_RESULT] success=%s iterations=%s reason=%s",
                    result.success, result.iterations, result.reason)
        self.tracer.log("retrieval", "loop_result",
                        success=result.success, iterations=result.iterations,
                        reason=result.reason)
        chunks: List[Chunk] = ws.data.get("chunks", [])
        if not result.success:
            if not chunks and str(result.reason).startswith("错误"):
                raise RuntimeError(f"检索子智能体失败: {result.reason}")
            self.tracer.log("retrieval", "exit_with_best",
                            reason=result.reason, count=len(chunks))
        # logger.info("[AGENT_OUTPUT] chunks=%d titles=%s",
        #             len(chunks), [c.doc_title for c in chunks])
        logger.info("[AGENT_OUTPUT] chunks=%d titles=%s",
                    len(chunks), [c.doc_title for c in chunks])

        # 检索出的 kid 列表:直接复用 intergrate_all 已按 kid 去重排序的 ranked_kids
        # (get_kid_score 合并 keyword/vector 两路 kid 后按得分排序),不再从 chunks 重复推导。
        # 注意:chunks 的去重键是 chunk_id(片段级),同一 kid 可对应多个片段。
        # retrieved_kids: List[str] = []
        # for c in chunks:
        #     if c.doc_id and c.doc_id != "unknown" and c.doc_id not in retrieved_kids:
        #         retrieved_kids.append(c.doc_id)
        retrieved_kids: List[str] = list(ws.data.get("ranked_kids") or [])
        expected_set = set(EXPECTED_KIDS)
        hit_kids = [k for k in retrieved_kids if k in expected_set]
        missed_kids = [k for k in EXPECTED_KIDS if k not in set(retrieved_kids)]
        logger.info("[AGENT_KIDS] retrieved_kids(%d)=%s", len(retrieved_kids), retrieved_kids)
        logger.info("[AGENT_KIDS] expected_kids(%d)=%s", len(EXPECTED_KIDS), EXPECTED_KIDS)
        logger.info("[AGENT_KIDS] 命中 hit_kids(%d/%d)=%s 未命中 missed_kids=%s",
                    len(hit_kids), len(EXPECTED_KIDS), hit_kids, missed_kids)

        # 打印每个命中知识的全部信息(同一 kid 的所有 chunk 全字段输出)
        for idx, c in enumerate([c for c in chunks if c.doc_id in set(hit_kids)], 1):
            logger.info("[AGENT_HIT_DETAIL] #%d %s",
                        idx, json.dumps(c.to_dict(), ensure_ascii=False))
        return chunks
