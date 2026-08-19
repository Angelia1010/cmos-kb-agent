# -*- coding: utf-8 -*-
"""ScriptedChatModel —— 离线可跑的 BaseChatModel 实现。

用规则模拟两个子Agent 的工具调用决策,让整套 uniagent/LangGraph ReAct
机制(工具绑定、ToolMessage 回灌、GoalLoop 反馈注入)真实跑通。
生产接入:换成 langchain_openai.ChatOpenAI / 内部网关模型即可,
config.yaml 的 models[0].use 一行切换。

同时提供 judge()/small_json()/large_json(),供充分性验证器与答案生成复用
(对应方案的模型分级:决策/判据用小模型,答案组织用大模型)。
"""
from __future__ import annotations

import json
import re
import uuid
from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from . import lexicon


def _called_tools(messages: List[BaseMessage],
                  since_last_feedback: bool = False) -> List[str]:
    """收集已发起的工具调用;since_last_feedback=True 时只统计最近一次
    [验证失败] 反馈之后的调用 —— 让重试轮重新走一遍检索工具链。"""
    start = 0
    if since_last_feedback:
        for i, m in enumerate(messages):
            if "[验证失败]" in str(getattr(m, "content", "")):
                start = i + 1
    names: List[str] = []
    for m in messages[start:]:
        if isinstance(m, AIMessage):
            for tc in (m.tool_calls or []):
                names.append(tc["name"])
    return names


def _tool_call(name: str, args: dict) -> dict:
    return {"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "tool_call"}


class ScriptedChatModel(BaseChatModel):
    """按消息历史决定下一个工具调用;工具全部执行完则输出最终答复。"""

    model: str = "scripted-mock"
    temperature: float = 0.0

    @property
    def _llm_type(self) -> str:
        return "kb-scripted-mock"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        """把工具名列表绑进调用 kwargs,供 _generate 判断当前扮演的子Agent。"""
        names = []
        for t in tools:
            name = getattr(t, "name", None) or getattr(t, "__name__", "")
            if name:
                names.append(name)
        return self.bind(bound_tool_names=names, **kwargs)

    def _generate(self, messages: List[BaseMessage],
                  stop: Optional[List[str]] = None,
                  run_manager: Optional[CallbackManagerForLLMRun] = None,
                  **kwargs: Any) -> ChatResult:
        text = "\n".join(str(getattr(m, "content", "")) for m in messages)
        retry = "[验证失败]" in text
        called = _called_tools(messages, since_last_feedback=retry)
        tool_names = set(kwargs.get("bound_tool_names") or [])

        is_processing = "清洗候选知识" in text or (
            "apply_business_skill" in tool_names and "coarse_recall" not in tool_names)
        is_retrieval = not is_processing and (
            "coarse_recall" in tool_names or "候选知识" in text)

        ai: AIMessage
        if is_retrieval and "coarse_recall" not in called:
            ai = self._next_retrieval(text, called)
        elif is_processing and not self._processing_done(called, text):
            ai = self._next_processing(text, called)
        else:
            ai = AIMessage(content="已完成当前阶段任务,结果写入工作区。")
        return ChatResult(generations=[ChatGeneration(message=ai)])

    # ---- 检索子Agent 决策:理解→(重试轮:改写)→关键词→召回 ----
    def _next_retrieval(self, text: str, called: List[str]) -> AIMessage:
        retry = "[验证失败]" in text
        if not retry and "query_understanding" not in called:
            return AIMessage(content="先理解问题意图。",
                             tool_calls=[_tool_call("query_understanding", {})])
        if retry and "question_rewrite" not in called:
            return AIMessage(content="上一轮验证失败,改写检索问题。",
                             tool_calls=[_tool_call("question_rewrite", {})])
        if "keyword_extraction" not in called:
            return AIMessage(content="提取关键词并做同义扩展。",
                             tool_calls=[_tool_call("keyword_extraction",
                                                    {"expand": True})])
        return AIMessage(content="执行混合召回。",
                         tool_calls=[_tool_call("coarse_recall",
                                                {"relax_filters": retry})])

    # ---- 数据处理子Agent 决策:分析→通用工具→业务skill(技能提示触发) ----
    _ORDER = ["analyze_data", "clean_data", "denoise_data",
              "dedupe_data", "structure_data", "sort_data"]

    def _processing_done(self, called: List[str], text: str) -> bool:
        base_done = all(t in called for t in self._ORDER)
        need_skill = "SKILL:" in text          # 技能中间件已注入业务skill提示
        return base_done and (not need_skill or "apply_business_skill" in called)

    def _next_processing(self, text: str, called: List[str]) -> AIMessage:
        for t in self._ORDER:
            if t not in called:
                return AIMessage(content=f"执行 {t}。",
                                 tool_calls=[_tool_call(t, {})])
        m = re.search(r"业务类目[::]\s*(套餐|宽带|账单|投诉)", text)
        cat = m.group(1) if m else next(
            (c for c in ("套餐", "宽带", "账单", "投诉") if c in text), "套餐")
        return AIMessage(content="套用业务skill做字段归一。",
                         tool_calls=[_tool_call("apply_business_skill",
                                                {"category": cat})])

    # ------------------------------------------------------------------
    # 非 Agent 场景的直调接口(判据/答案组织),供验证器与 answer.py 复用
    # ------------------------------------------------------------------
    def judge(self, system: str, user: str) -> str:
        if "[TASK:intent_coverage]" in system:
            query = user.split("用户问题:")[-1].split("\n")[0]
            titles = user.split("已召回标题:")[-1]
            intents = lexicon.detect_categories(query) or [query[:4]]
            uncovered = [i for i in intents if i not in titles]
            return json.dumps({"covered": not uncovered,
                               "uncovered_intents": uncovered}, ensure_ascii=False)
        if "[TASK:anchor_check]" in system:
            sent = user.split("句子:")[-1].split("\n")[0]
            chunk = user.split("片段:")[-1]
            overlap = sum(1 for ch in set(sent) if ch in chunk and not ch.isspace())
            return json.dumps(
                {"consistent": overlap >= max(3, int(len(set(sent)) * 0.3))},
                ensure_ascii=False)
        return "{}"

    def small_json(self, system: str, user: str) -> dict:
        try:
            return json.loads(self.judge(system, user))
        except json.JSONDecodeError:
            return {}

    def large_json(self, system: str, user: str) -> dict:
        if "[TASK:answer]" not in system:
            return {}
        chunks = re.findall(r'<chunk id="(.+?)">(.+?)</chunk>', user, re.S)
        if not chunks:
            return {"business_explanation": "", "handling_suggestion": "",
                    "sentences": []}
        expl, sentences = [], []
        for cid, content in chunks[:3]:
            first = content.strip().split("。")[0][:60] + "。"
            expl.append(first)
            sentences.append({"text": first, "citations": [cid],
                              "hard_fact": any(w in first for w in
                                               ("元", "资费", "条件", "生效"))})
        sugg = "可为客户办理上述业务,办理前请与客户确认需求与资费。"
        sentences.append({"text": sugg, "citations": [chunks[0][0]],
                          "hard_fact": False})
        return {"business_explanation": " ".join(expl),
                "handling_suggestion": sugg, "sentences": sentences}
