"""技能激活中间件 — 自动匹配用户输入并注入技能内容。"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import SystemMessage

# NOTE: config_factory 依赖链为 config_factory → factory → features → middleware.base，
#       不会回绕至 skill_middleware，因此此处可安全地在文件头导入（无循环依赖）。
#       （features.py 中对 middleware.builtins 的导入保持延迟，是整条链的断点。）
from uniagent.agents.config_factory import get_skill_registry
from uniagent.middleware.base import Middleware
from uniagent.skills.injector import SkillInjector

logger = logging.getLogger(__name__)


class SkillMiddleware(Middleware):
    """拦截用户消息并激活匹配的技能。

    在每次 ``before_agent`` 调用时，该中间件将：
    1. 提取最新的用户消息。
    2. 将其与技能注册表中的触发器进行匹配。
    3. 若找到匹配项，加载技能内容并以 SystemMessage 追加到消息流。

    幂等保护：若消息流中已存在 ``<!-- SKILL: ... -->`` 标记，则跳过注入，
    防止 GoalLoop 多轮迭代重复注入同一技能。

    此中间件应置于链的早期位置（在摘要和循环检测之前），
    以确保 agent 从一开始就能看到技能指令。
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # 延迟初始化：仅在第一次 before_agent 触发时完成实例化
        self._injector: SkillInjector | None = None
        self._initialized: bool = False

    def _ensure_initialized(self) -> bool:
        """延迟初始化技能注入器。

        技能注册表必须在此之前已由 _setup_skills() 或
        register_skill_directory() 完成初始化；否则返回 False 跳过。
        """
        if self._initialized:
            return self._injector is not None

        self._initialized = True
        registry = get_skill_registry()
        if registry is None:
            logger.debug("技能中间件：注册表未初始化，跳过。")
            return False

        self._injector = SkillInjector()
        return True

    async def before_agent(self, state: dict[str, Any]) -> dict[str, Any] | None:
        """检查用户消息是否触发技能，若匹配则注入 SystemMessage。"""
        if not self._ensure_initialized():
            return None

        # ── 1. 提取消息列表 ──
        messages = state.get("messages", [])
        if not messages:
            return None

        # ── 2. 幂等检查：已有技能注入标记时跳过，防止 GoalLoop 多轮重复注入 ──
        for msg in messages:
            c = getattr(msg, "content", "")
            if isinstance(c, str) and "<!-- SKILL:" in c:
                return None

        # ── 3. 从后往前找最后一条有效 HumanMessage ──
        # （GoalLoop 会在末尾追加 SystemMessage，因此不能直接取 messages[-1]）
        human_msg = None
        for msg in reversed(messages):
            if getattr(msg, "type", None) == "human":
                text = getattr(msg, "content", "")
                # 跳过 GoalLoop 注入的验证反馈消息
                if isinstance(text, str) and not text.startswith("[验证失败]"):
                    human_msg = msg
                    break
        if human_msg is None:
            return None

        content = getattr(human_msg, "content", "")
        if not content or not isinstance(content, str):
            return None

        # ── 4. 触发器匹配：取得分最高的技能 ──
        registry = get_skill_registry()
        matches = registry.match(content, max_results=1)
        if not matches:
            return None

        match = matches[0]
        if match.score < 0.3:
            return None

        # ── 5. 激活技能：加载 SKILL.md + 参考文档 + 脚本工具 ──
        skill_content = registry.activate(match)
        self._injector.activate(skill_content)

        logger.info(
            "技能 %r 已激活（score=%.2f，trigger=%s）。",
            match.manifest.name,
            match.score,
            match.matched_trigger.type if match.matched_trigger else "direct",
        )

        # ── 6. 将技能内容作为 SystemMessage 追加到消息流 ──
        # （替代原先依赖 AgentMiddleware 提示词管道的实现，该管道在
        #  langgraph 当前版本不存在；直接写入 state 可审计、可追溯）
        body = getattr(skill_content, "instruction", "") or str(skill_content)

        # 已激活的脚本工具：告知 LLM 本技能提供了哪些可直接调用的工具
        # （脚本工具已在 create_agent 时通过 factory.py 静态绑定到模型，
        #  此处仅在 SystemMessage 中显式列出，帮助 LLM 知道它们的存在和用途）
        script_tools = getattr(skill_content, "script_tools", [])
        if script_tools:
            body += "\n\n## 本技能已激活的业务工具（可直接调用）\n"
            for t in script_tools:
                tool_desc = getattr(t, "description", "") or ""
                # description 可能多行，取第一行作摘要
                tool_desc_summary = tool_desc.splitlines()[0] if tool_desc else ""
                body += f"- `{t.name}`: {tool_desc_summary}\n"

        # 推荐优先使用的通用工具（来自 metadata.promoted_tools）
        if match.manifest.promoted_tools:
            body += "\n## 推荐优先使用的工具\n"
            for tool_name in match.manifest.promoted_tools:
                body += f"- `{tool_name}`\n"

        # 列出可按需加载的参考文档，提示 LLM 可调用 load_skill_reference
        on_demand_refs = [
            ref for ref in match.manifest.references if ref.when == "on_demand"
        ]
        if on_demand_refs:
            body += "\n## 可按需加载的参考文档（调用 load_skill_reference 获取）\n"
            for ref in on_demand_refs:
                desc = f": {ref.description}" if ref.description else ""
                body += f"- `{ref.filename}`{desc}\n"

        skill_msg = SystemMessage(
            content=f"<!-- SKILL: {match.manifest.name} -->\n{body}"
        )

        # ── 7. 将 skill_msg 插入到第一条非 SystemMessage 之前 ──
        # Anthropic 等 API 要求 SystemMessage 必须连续出现在消息流开头，
        # 不能在 HumanMessage / AIMessage 之后再出现 SystemMessage。
        # GoalLoop 已在 [0] 插入了 SystemMessage([目标])，SkillMiddleware
        # 若直接 append 到末尾，就会触发 "multiple non-consecutive system messages"。
        # 正确做法：找到最后一条连续 SystemMessage 的位置，将 skill_msg 紧随其后插入。
        insert_idx = 0
        for idx, m in enumerate(messages):
            if isinstance(m, SystemMessage):
                insert_idx = idx + 1  # 跟在最后一条 SystemMessage 后面
            else:
                break  # 第一条非 SystemMessage，插入点固定

        patched_messages = messages[:insert_idx] + [skill_msg] + messages[insert_idx:]
        patch: dict[str, Any] = {"messages": patched_messages}

        # 将 promoted_tools 写入 state，供下游中间件或工具路由使用
        promoted = self._injector.get_promoted_tools()
        if promoted:
            patch["promoted_tools"] = promoted

        return patch
