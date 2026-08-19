"""技能激活中间件 — 自动匹配用户输入并注入技能内容。"""

from __future__ import annotations

import logging
from typing import Any

from uniagent.middleware.base import Middleware

logger = logging.getLogger(__name__)


class SkillMiddleware(Middleware):
    """拦截用户消息并激活匹配的技能。

    在每次 ``before_agent`` 调用时，该中间件将：
    1. 提取最新的用户消息。
    2. 将其与技能注册表中的触发器进行匹配。
    3. 若找到匹配项，加载技能内容并通过 ``SkillInjector`` 扩充系统提示词。

    此中间件应置于链的早期位置（在摘要和循环检测之前），
    以确保 agent 从一开始就能看到技能指令。
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._injector: Any = None
        self._registry: Any = None

    def _ensure_initialized(self) -> bool:
        """延迟初始化：仅在需要时导入技能子系统。"""
        if self._injector is not None:
            return True

        try:
            from uniagent.agents.config_factory import get_skill_registry
            from uniagent.skills.injector import SkillInjector

            registry = get_skill_registry()
            if registry is None:
                return False

            self._registry = registry
            self._injector = SkillInjector()
            return True
        except Exception as exc:
            logger.debug("技能中间件初始化失败：%s", exc)
            return False

    async def before_agent(self, state: dict[str, Any]) -> dict[str, Any] | None:
        """检查用户消息是否触发技能，若匹配则注入。"""
        if not self._ensure_initialized():
            return None

        # 提取最新的用户消息
        messages = state.get("messages", [])
        if not messages:
            return None

        last_msg = messages[-1]
        # 仅对 HumanMessage 进行匹配
        msg_type = getattr(last_msg, "type", None)
        if msg_type != "human":
            return None

        content = getattr(last_msg, "content", "")
        if not content or not isinstance(content, str):
            return None

        # 与触发器进行匹配
        matches = self._registry.match(content, max_results=1)
        if not matches:
            return None

        match = matches[0]
        if match.score < 0.3:
            return None

        # M5: 激活新技能前先注入上一次的技能内容（如有），避免状态积累
        skill_content = self._registry.activate(match)

        logger.info(
            "技能 %r 已激活（score=%.2f，trigger=%s）。",
            match.manifest.name,
            match.score,
            match.matched_trigger.type if match.matched_trigger else "direct",
        )

        # (KB 适配) 原实现依赖 AgentMiddleware 的提示词管道注入技能内容,
        # 该管道在当前 langgraph 版本不存在;改为把技能内容作为 SystemMessage
        # 直接追加进消息流,agent 在下一次推理即可看到技能指令。
        from langchain_core.messages import SystemMessage

        body = getattr(skill_content, "instruction", "") or str(skill_content)
        skill_msg = SystemMessage(
            content=f"<!-- SKILL: {match.manifest.name} -->\n{body}")
        return {"messages": messages + [skill_msg]}
