"""空操作验证器，始终通过。用于测试或轮次循环模式。"""

from __future__ import annotations

from typing import Any

from uniagent.verification.verifier import VerificationResult


class AlwaysPassVerifier:
    """空操作验证器，始终通过。"""

    async def verify(self, goal: str, state: dict[str, Any]) -> VerificationResult:
        return VerificationResult(passed=True, evidence="（无验证）", layer="noop")
