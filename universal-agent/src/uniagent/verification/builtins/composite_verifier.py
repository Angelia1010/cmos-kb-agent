"""链式多验证器 — 在第一个失败层快速失败。"""

from __future__ import annotations

from typing import Any, Sequence

from uniagent.verification.verifier import VerificationResult


class CompositeVerifier:
    """链式多验证器 — 在第一个失败层快速失败。

    实现三层终止检查模式::

        lint → test → e2e

    每一层必须通过后才会尝试下一层。
    """

    def __init__(self, verifiers: Sequence[Any]) -> None:
        self._verifiers = list(verifiers)

    async def verify(self, goal: str, state: dict[str, Any]) -> VerificationResult:
        all_evidence: list[str] = []

        for verifier in self._verifiers:
            result = await verifier.verify(goal, state)
            layer_name = result.layer or type(verifier).__name__
            all_evidence.append(f"[{layer_name}] {'通过' if result.passed else '失败'}: {result.evidence}")

            if not result.passed:
                return VerificationResult(
                    passed=False,
                    evidence="\n".join(all_evidence),
                    confidence=result.confidence,
                    layer=f"composite(failed@{layer_name})",
                )

        return VerificationResult(
            passed=True,
            evidence="\n".join(all_evidence),
            confidence=1.0,
            layer="composite(all_passed)",
        )
