# -*- coding: utf-8 -*-
"""ScriptedChatModel — 离线可跑的 BaseChatModel 实现。

用规则模拟子智能体的工具调用决策,让整套 uniagent/LangGraph ReAct
机制(工具绑定、ToolMessage 回灌、GoalLoop 反馈注入)真实跑通。
同时处理答案环节的直调请求([TASK:answer] / [TASK:anchor_check] /
[TASK:locate_fragments])。

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

        # ---- 知识候选两阶段重排直调 ----
        if "[TASK:rerank_batch]" in all_text or "[TASK:rerank_global]" in all_text:
            return ChatResult(generations=[ChatGeneration(
                message=AIMessage(content=self._scripted_rerank(all_text)))])

        # ---- Top3 回答充分性校验直调 ----
        if "[TASK:top3_answerability]" in all_text:
            return ChatResult(generations=[ChatGeneration(
                message=AIMessage(content=self._scripted_top3_answerability(all_text)))])

        # ---- 答案生成直调 ----
        if "[TASK:answer]" in all_text:
            return ChatResult(generations=[ChatGeneration(
                message=AIMessage(content=self._scripted_answer(all_text)))])

        # ---- 锚定校验直调 ----
        if "[TASK:anchor_check]" in all_text:
            return ChatResult(generations=[ChatGeneration(
                message=AIMessage(content=self._scripted_anchor(all_text)))])

        # ---- 文档内证据片段定位直调 ----
        if "[TASK:locate_fragments]" in all_text:
            return ChatResult(generations=[ChatGeneration(
                message=AIMessage(content=self._scripted_locate(all_text)))])

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
    @staticmethod
    def _clean_demo_text(s: str) -> str:
        """演示数据清洗:去 markdown 标题符与换行,拼出的话术才像人话。"""
        s = s.replace("\\n", " ").replace("\n", " ")
        s = re.sub(r"#+\s*", "", s)
        return re.sub(r"\s+", " ", s).strip()

    def _scripted_answer(self, text: str) -> str:
        chunks = re.findall(r'<chunk id="(.+?)">(.+?)</chunk>', text, re.S)
        if not chunks:
            return json.dumps({"script": "", "handling_suggestion": "",
                               "usability": {"level": "not_usable",
                                             "reasons": ["无可用知识素材"],
                                             "uncovered": []}},
                              ensure_ascii=False)
        expl = []
        for _cid, content in chunks[:3]:
            content = self._clean_demo_text(content)
            expl.append(content.split("。")[0][:60] + "。")
        sugg = "办理前请与客户确认需求与资费,离线演示话术,生产以真实模型输出为准。"
        script = "您好," + "".join(expl) + sugg
        return json.dumps({"script": script,
                           "handling_suggestion": sugg,
                           "usability": {"level": "verify_first",
                                         "reasons": ["离线脚本模型生成,仅演示"],
                                         "uncovered": []}},
                          ensure_ascii=False)

    # ---- 批量话术一致性校验脚本 ----
    def _scripted_anchor(self, text: str) -> str:
        """离线模拟批量校验:话术与素材字符重叠度过低则判不一致。"""
        script = text.split("坐席话术:")[-1].split("\n")[0]
        material = text.split("知识片段:")[-1]
        schars = {ch for ch in script if not ch.isspace()}
        overlap = sum(1 for ch in schars if ch in material)
        consistent = overlap >= max(3, int(len(schars) * 0.5))
        issues = [] if consistent else ["话术含素材中未出现的表述(离线脚本判定)"]
        return json.dumps({"consistent": consistent, "issues": issues},
                          ensure_ascii=False)

    # ---- 文档内证据片段定位脚本 ----
    def _scripted_locate(self, text: str) -> str:
        """离线模拟 [TASK:locate_fragments]:按句切分文档,挑与问题字符重叠的句子。

        返回的 text 必为文档原文连续子串,使 locate_fragments 的逐字校验通过。
        """
        m = re.search(r"用户问题[:：]\s*(.*?)\s*文档内容[:：]\s*(.*)$", text, re.S)
        if not m:
            return json.dumps({"answerable": False, "relevance": 0, "fragments": []},
                              ensure_ascii=False)
        query, content = m.group(1), m.group(2)
        qchars = {c for c in query if not c.isspace()}
        if not qchars or not content.strip():
            return json.dumps({"answerable": False, "relevance": 0, "fragments": []},
                              ensure_ascii=False)
        sentences = [s.strip() for s in
                     re.split(r"(?<=[。;!?；！?\n])", content) if s.strip()]
        threshold = max(2, int(len(qchars) * 0.3))
        fragments, best = [], 0.0
        for s in sentences:
            schars = {c for c in s if not c.isspace()}
            hit = len(qchars & schars)
            if hit >= threshold:
                fragments.append({"text": s, "reason": "离线脚本:与问题字符重叠"})
                best = max(best, hit / len(qchars))
            if len(fragments) >= 3:
                break
        # 相关度:命中覆盖率映射到 40-95(有片段);无片段给 0
        relevance = int(round(40 + best * 55)) if fragments else 0
        return json.dumps({"answerable": bool(fragments),
                           "relevance": min(95, relevance),
                           "fragments": fragments},
                          ensure_ascii=False)

    # ---- Top3 回答充分性校验脚本 ----
    def _scripted_top3_answerability(self, text: str) -> str:
        if "TOP3_VERIFICATION_INPUT_BEGIN" not in text:
            return "{}"
        serialized = text.rsplit("TOP3_VERIFICATION_INPUT_BEGIN", 1)[-1]
        serialized = serialized.split("TOP3_VERIFICATION_INPUT_END", 1)[0].strip()
        try:
            payload = json.loads(serialized)
        except json.JSONDecodeError:
            return "{}"

        query = str(payload.get("query") or "").strip()
        candidates = payload.get("candidates") or []
        suggested_keywords = [
            keyword for keyword in dict.fromkeys(lexicon.extract_keywords(query))
            if keyword not in {"信息", "内容", "知识", "问题", "相关", "业务"}
        ] or [f"{query}业务规则"]
        if not candidates:
            return json.dumps({
                "status": "failed",
                "reason_codes": ["no_valid_candidates"],
                "summary": "当前没有可用于验证的有效候选知识。",
                "evidence_ids": [],
                "retrieval_feedback": {
                    "suggested_query": query,
                    "missing_aspects": [f"缺少能够回答“{query}”的有效候选知识"],
                    "suggested_keywords": suggested_keywords,
                    "retry_strategy": "broaden_semantic_recall",
                },
            }, ensure_ascii=False)

        def terms(value: str) -> set[str]:
            value = value.casefold()
            found = set(re.findall(r"[a-z0-9]+", value))
            chinese = "".join(re.findall(r"[\u4e00-\u9fff]", value))
            ignored = set("的了呢吗啊呀和与及或请问如何怎么是否可以能否")
            found.update(ch for ch in chinese if ch not in ignored)
            found.update(
                chinese[index:index + 2]
                for index in range(max(0, len(chinese) - 1))
                if not set(chinese[index:index + 2]) <= ignored
            )
            return found

        query_terms = terms(query)
        candidate_terms: list[tuple[str, set[str]]] = []
        combined_terms: set[str] = set()
        for candidate in candidates:
            evidence_id = str(candidate.get("evidence_id") or "")
            value = f"{candidate.get('title') or ''}\n{candidate.get('content_md') or ''}"
            current_terms = terms(value)
            candidate_terms.append((evidence_id, current_terms))
            combined_terms.update(current_terms)
        overlap = query_terms & combined_terms
        coverage = len(overlap) / max(1, len(query_terms))
        evidence_ids = [
            evidence_id for evidence_id, current_terms in candidate_terms
            if evidence_id and query_terms & current_terms
        ]
        if coverage >= 0.65 and evidence_ids:
            return json.dumps({
                "status": "passed",
                "reason_codes": [],
                "summary": "当前候选已覆盖回答用户问题所需的主要信息。",
                "evidence_ids": evidence_ids,
                "retrieval_feedback": None,
            }, ensure_ascii=False)

        off_topic = not overlap
        reason = "off_topic" if off_topic else (
            "partial_intent_coverage"
            if re.search(r"[、，,]|(?:和|及|与)", query)
            else "missing_key_fact"
        )
        strategy = (
            "replace_off_topic_results" if off_topic else "supplement_missing_aspects"
        )
        return json.dumps({
            "status": "failed",
            "reason_codes": [reason],
            "summary": "当前候选与问题偏离。" if off_topic else "当前候选缺少回答所需的关键信息。",
            "evidence_ids": evidence_ids,
            "retrieval_feedback": {
                "suggested_query": query,
                "missing_aspects": [f"缺少关于“{query}”的完整回答依据"],
                "suggested_keywords": suggested_keywords,
                "retry_strategy": strategy,
            },
        }, ensure_ascii=False)

    # ---- 知识候选重排脚本 ----
    def _scripted_rerank(self, text: str) -> str:
        matches = re.findall(r"RERANK_INPUT_BEGIN\s*(\{.*?\})\s*RERANK_INPUT_END", text, re.S)
        if not matches:
            return json.dumps({"ranked_ids": []}, ensure_ascii=False)
        try:
            payload = json.loads(matches[-1])
        except json.JSONDecodeError:
            return json.dumps({"ranked_ids": []}, ensure_ascii=False)
        query = " ".join(filter(None, (
            str(payload.get("query") or ""), str(payload.get("retrieval_query") or "")
        )))

        def terms(value: str) -> set[str]:
            value = value.casefold()
            found = set(re.findall(r"[a-z0-9]+", value))
            chinese = "".join(re.findall(r"[\u4e00-\u9fff]", value))
            found.update(chinese[i:i + 2] for i in range(max(0, len(chinese) - 1)))
            found.update(ch for ch in chinese if ch.strip())
            return found

        query_terms = terms(query)
        scored = []
        for input_index, candidate in enumerate(payload.get("candidates") or []):
            evidence_id = str(candidate.get("evidence_id") or "")
            title = str(candidate.get("title") or "")
            content = str(candidate.get("content_md") or "")
            title_terms = terms(title)
            content_terms = terms(content)
            score = 4 * len(query_terms & title_terms) + len(query_terms & content_terms)
            if query and query.casefold() in (title + "\n" + content).casefold():
                score += 20
            scored.append((-score, input_index, evidence_id))
        scored.sort()
        top_k = max(0, int(payload.get("top_k") or 0))
        return json.dumps({"ranked_ids": [item[2] for item in scored[:top_k]]}, ensure_ascii=False)
