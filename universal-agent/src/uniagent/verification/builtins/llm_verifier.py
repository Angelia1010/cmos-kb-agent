"""使用独立 LLM 判断目标是否完成（生成器/评估器分离）。"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from uniagent.verification.verifier import VerificationResult

logger = logging.getLogger(__name__)


class LLMVerifier:
    """使用独立 LLM 判断目标是否完成。

    这是生成器/评估器分离的核心 — 评估器 LLM 是与
    生成解决方案的 LLM **不同的实例**（可能是不同的模型）。

    参数
    ----------
    model:
        用于评估的 LangChain 聊天模型实例。
    system_prompt:
        评估器的自定义系统提示。
    confidence_threshold:
        "通过"判断的最低置信度（0.0–1.0）。
    """

    _DEFAULT_SYSTEM = (
        "你是一个独立的评估者。你的任务是根据对话历史和状态，"
        "判断目标是否已经达成。\n\n"
        "请以 JSON 对象的格式回复：\n"
        '{"passed": true/false, "confidence": 0.0-1.0, "evidence": "说明理由"}\n\n'
        "请保持严格和客观。没有明确证据时，不要假设目标已完成。"
    )

    def __init__(
        self,
        model: BaseChatModel,
        *,
        system_prompt: str | None = None,
        confidence_threshold: float = 0.7,
    ) -> None:
        self._model = model
        self._system_prompt = system_prompt or self._DEFAULT_SYSTEM
        self._threshold = confidence_threshold

    async def verify(self, goal: str, state: dict[str, Any]) -> VerificationResult:
        """调用评估器 LLM 判断目标是否达成，返回结构化验证结果。"""
        messages_summary = _summarize_messages(state.get("messages", []))

        eval_messages = [
            SystemMessage(content=self._system_prompt),
            HumanMessage(
                content=(
                    f"## 目标\n{goal}\n\n"
                    f"## 对话历史（摘要）\n{messages_summary}\n\n"
                    f"## 产出物\n{state.get('artifacts', {})}\n\n"
                    "目标是否已达成？请以 JSON 格式回复。"
                )
            ),
        ]

        try:
            response = await self._model.ainvoke(eval_messages)
            content = str(response.content).strip()

            # M10: 健壮的 JSON 代码块提取，支持大写标记、多代码块等
            if "```" in content:
                # 匹配所有代码块，提取第一个包含 JSON 的
                blocks = re.findall(
                    r"```(?:[jJ][sS][oO][nN])?\s*\n?(.*?)```",
                    content,
                    re.DOTALL,
                )
                for block in blocks:
                    stripped = block.strip()
                    if stripped.startswith("{"):
                        content = stripped
                        break
                else:
                    # 回退：取第一个代码块
                    if blocks:
                        content = blocks[0].strip()

            result = json.loads(content)
            passed = bool(result.get("passed", False))
            confidence = float(result.get("confidence", 0.0))
            evidence = str(result.get("evidence", ""))

            # 置信度低于阈值时强制降级为不通过
            if passed and confidence < self._threshold:
                passed = False
                evidence = (
                    f"评估器判断通过，但置信度 ({confidence:.2f}) "
                    f"低于阈值 ({self._threshold:.2f})。"
                    f"原始理由：{evidence}"
                )

            return VerificationResult(
                passed=passed,
                evidence=evidence,
                confidence=confidence,
                layer="llm",
            )
        except Exception as exc:
            logger.warning("LLM 验证失败：%s", exc)
            return VerificationResult(
                passed=False,
                evidence=f"LLM 验证出错：{exc}",
                confidence=0.0,
                layer="llm",
            )


def _summarize_messages(messages: list[Any], max_chars: int = 3000) -> str:
    """为评估器构建对话消息的精简摘要（从最新消息开始，不超过 max_chars）。"""
    parts: list[str] = []
    total = 0
    for msg in reversed(messages):
        role = getattr(msg, "type", "unknown")
        content = str(getattr(msg, "content", ""))[:500]
        line = f"[{role}] {content}"
        if total + len(line) > max_chars:
            parts.append("... （更早的消息已截断）")
            break
        parts.append(line)
        total += len(line)
    parts.reverse()
    return "\n".join(parts)
