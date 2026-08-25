# -*- coding: utf-8 -*-
"""ScriptedChatModel — 离线可跑的 BaseChatModel 实现。

用规则模拟子智能体的工具调用决策,让整套 uniagent/LangGraph ReAct
机制(工具绑定、ToolMessage 回灌、GoalLoop 反馈注入)真实跑通。
同时处理答案生成和锚定校验的直调请求([TASK:answer] / [TASK:anchor_check])。

生产接入:换成 langchain_openai.ChatOpenAI 即可。
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

from .shared import lexicon


def _called_tools(messages: List[BaseMessage],
                  since_last_feedback: bool = False) -> List[str]:
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
    """按消息历史决定下一个工具调用或生成 JSON 响应。"""

    model: str = "scripted-mock"
    temperature: float = 0.0

    @property
    def _llm_type(self) -> str:
        return "kb-scripted-mock"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
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
        all_text = "\n".join(str(getattr(m, "content", "")) for m in messages)

        # ---- 答案生成直调 ----
        if "[TASK:answer]" in all_text:
            return ChatResult(generations=[ChatGeneration(
                message=AIMessage(content=self._scripted_answer(all_text)))])

        # ---- 锚定校验直调 ----
        if "[TASK:anchor_check]" in all_text:
            return ChatResult(generations=[ChatGeneration(
                message=AIMessage(content=self._scripted_anchor(all_text)))])

        # ---- ReAct 工具调用模式 ----
        retry = "[验证失败]" in all_text
        called = _called_tools(messages, since_last_feedback=retry)
        tool_names = set(kwargs.get("bound_tool_names") or [])

        is_processing = "清洗候选知识" in all_text or (
            "apply_business_skill" in tool_names and "coarse_recall" not in tool_names)
        is_retrieval = not is_processing and (
            "coarse_recall" in tool_names or "候选知识" in all_text)

        if is_retrieval and "coarse_recall" not in called:
            ai = self._next_retrieval(all_text, called)
        elif is_processing and not self._processing_done(called, all_text):
            ai = self._next_processing(all_text, called)
        else:
            ai = AIMessage(content="已完成当前阶段任务,结果写入工作区。")
        return ChatResult(generations=[ChatGeneration(message=ai)])

    # ---- 检索子智能体决策 ----
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

    # ---- 数据处理子智能体决策 ----
    _ORDER = ["analyze_data", "clean_data", "denoise_data",
              "dedupe_data", "structure_data", "sort_data"]

    def _processing_done(self, called: List[str], text: str) -> bool:
        base_done = all(t in called for t in self._ORDER)
        need_skill = "SKILL:" in text
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

    # ---- 答案生成脚本 ----
    def _scripted_answer(self, text: str) -> str:
        chunks = re.findall(r'<chunk id="(.+?)">(.+?)</chunk>', text, re.S)
        if not chunks:
            return json.dumps({"business_explanation": "", "handling_suggestion": "",
                               "sentences": []}, ensure_ascii=False)
        expl, sentences = [], []
        for cid, content in chunks[:3]:
            first = content.strip().split("。")[0][:60] + "。"
            expl.append(first)
            sentences.append({"text": first, "citations": [cid],
                              "hard_fact": any(w in first for w in
                                               ("元", "资费", "条件", "生效"))})
        sugg = "可为客户办理上述业务,办理前请与客户确认需求与资费。"
        sentences.append({"text": sugg, "citations": [chunks[0][0]], "hard_fact": False})
        return json.dumps({"business_explanation": " ".join(expl),
                           "handling_suggestion": sugg, "sentences": sentences},
                          ensure_ascii=False)

    # ---- 锚定校验脚本 ----
    def _scripted_anchor(self, text: str) -> str:
        sent = text.split("句子:")[-1].split("\n")[0]
        chunk = text.split("片段:")[-1]
        overlap = sum(1 for ch in set(sent) if ch in chunk and not ch.isspace())
        consistent = overlap >= max(3, int(len(set(sent)) * 0.3))
        return json.dumps({"consistent": consistent}, ensure_ascii=False)
