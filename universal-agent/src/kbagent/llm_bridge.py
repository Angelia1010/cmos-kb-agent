# -*- coding: utf-8 -*-
"""LLMBridge —— 把任意 langchain BaseChatModel 适配出本项目需要的判据/生成接口。

审查发现的接口缺口:answer.generate 依赖 llm.large_json/small_json,
SufficiencyVerifier 依赖 judge,但普通 ChatOpenAI 等模型没有这些方法,
生产接入会在答案阶段崩溃(被降级兜底吞掉,表现为"每个请求都降级")。
本适配器补齐该缺口:MainAgent 检测到模型缺少这些方法时自动包装。
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage


def _parse_json(raw: str) -> Dict[str, Any]:
    raw = re.sub(r"^```(json)?|```$", "", str(raw).strip(), flags=re.M).strip()
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


class LLMBridge:
    """用 model.invoke([system, user]) 实现 judge/small_json/large_json。

    生产建议:judge/small_json 传小模型实例、large_json 传大模型实例
    (对应方案的模型分级);只传一个模型时两者共用。
    """

    def __init__(self, model: Any, large_model: Any = None):
        self._small = model
        self._large = large_model or model

    def _chat(self, model: Any, system: str, user: str) -> str:
        resp = model.invoke([SystemMessage(content=system),
                             HumanMessage(content=user)])
        return str(getattr(resp, "content", resp))

    def judge(self, system: str, user: str) -> str:
        return self._chat(self._small, system, user)

    def small_json(self, system: str, user: str) -> Dict[str, Any]:
        return _parse_json(self._chat(self._small, system, user))

    def large_json(self, system: str, user: str) -> Dict[str, Any]:
        return _parse_json(self._chat(self._large, system, user))


def ensure_judge_interface(model: Any, large_model: Any = None) -> Any:
    """模型已具备三个方法则原样返回,否则包一层 LLMBridge。"""
    if all(hasattr(model, m) for m in ("judge", "small_json", "large_json")):
        return model
    return LLMBridge(model, large_model)
