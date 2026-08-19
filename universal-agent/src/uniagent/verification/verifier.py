"""验证器协议与结果模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class VerificationResult:
    """验证检查的结果。"""

    passed: bool
    """验证标准是否已满足。"""

    evidence: str = ""
    """人类可读的证据（测试输出、LLM 推理等）。"""

    confidence: float = 1.0
    """0.0–1.0 置信度得分（对基于 LLM 的验证有意义）。"""

    layer: str = ""
    """产生此结果的验证层（例如 'lint'、'test'、'e2e'）。"""

    details: dict[str, Any] = field(default_factory=dict)
    """任意结构化的详细信息。"""


@runtime_checkable
class Verifier(Protocol):
    """目标验证协议。

    实现必须独立于生成器 — 接收目标和当前状态，
    并产生带有证据的通过/失败判断。
    这强制了生成器/评估器分离原则。
    """

    async def verify(
        self,
        goal: str,
        state: dict[str, Any],
    ) -> VerificationResult:
        """根据当前 *state* 检查 *goal* 是否已达成。"""
        ...
