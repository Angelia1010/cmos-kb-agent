"""SDK 层 Agent 工厂 — ``create_agent()``。"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from langchain_core.language_models import BaseChatModel
from langgraph.prebuilt import create_react_agent

from uniagent.agents.features import AgentFeatures
from uniagent.middleware.base import Middleware
from uniagent.middleware.chain import assemble_middleware_chain
from uniagent.runtime.budget import Budget, BudgetConfig
from uniagent.runtime.hooks import LoopHook
from uniagent.runtime.loop import GoalLoop, TurnLoop
from uniagent.state.thread_state import ThreadState
from uniagent.verification.builtins import AlwaysPassVerifier

logger = logging.getLogger(__name__)


def create_agent(
    model: BaseChatModel,
    tools: Sequence[Any],
    *,
    # 模式一：完全控制中间件
    middleware: list[Middleware] | None = None,
    # 模式二：基于特性自动组装
    features: AgentFeatures | None = None,
    extra_middleware: Sequence[Middleware] | None = None,
    # 循环选项
    goal: str | None = None,
    verifier: Any | None = None,
    loop_hooks: Sequence[LoopHook] | None = None,
    budget: Budget | BudgetConfig | None = None,
    # 通用选项
    system_prompt: str = "",
    state_schema: type = ThreadState,
    checkpointer: Any = None,
    name: str = "universal_agent",
    **langgraph_kwargs: Any,
) -> Any:
    """创建带有 uniagent 中间件和可选循环的 LangGraph ReAct Agent。

    三种使用模式：

    1. **裸 Agent**：不设 ``goal`` → 返回 ``CompiledGraph``（与之前相同）。
    2. **TurnLoop**：设置 ``budget`` 但不设 ``goal`` → 返回 ``TurnLoop`` 包装器。
    3. **GoalLoop**：设置 ``goal`` + ``verifier`` → 返回 ``GoalLoop`` 包装器，
       用于自主的、验证驱动的执行。

    中间件组装是正交的 — 始终应用于内部 Agent。

    返回
    -------
    CompiledGraph | TurnLoop | GoalLoop
        可调用 ``.invoke()`` / ``.astream()``（裸 Agent）或
        ``.run()``（循环包装器）。
    """
    if middleware is not None and features is not None:
        raise ValueError(
            "不能同时指定 'middleware' 和 'features'。"
            "使用 'middleware' 进行完全控制，或使用 'features' + 'extra_middleware' "
            "进行自动组装。"
        )

    # ── 解析中间件链 ──
    feat = features or AgentFeatures()
    if middleware is not None:
        chain = middleware
    else:
        built_in = feat.resolve_middleware()
        chain = assemble_middleware_chain(
            built_in,
            extras=list(extra_middleware) if extra_middleware else None,
        )

    # ── 预加载技能工具 ──
    # NOTE: skills.* 的导入保持延迟（不在文件头），原因：
    #   factory.py 被 config_factory.py 在文件头导入；
    #   skills.tools 被 skills/__init__.py 在文件头导入；
    #   skills/__init__.py 在加载时触发 tools.py，tools.py 文件头导入 config_factory，
    #   config_factory 文件头导入 factory → 形成循环。
    #   因此 factory.py 对 uniagent.skills 的所有导入必须保持延迟。
    final_tools = list(tools)
    if feat.skill:
        try:
            from uniagent.agents.config_factory import get_skill_registry  # 循环依赖，保持延迟
            from uniagent.skills.script_loader import load_skill_scripts    # 循环依赖，保持延迟
            from uniagent.skills.tools import load_skill_reference          # 循环依赖，保持延迟

            # load_skill_reference：供 LLM 按需加载参考文档
            final_tools.append(load_skill_reference)
            # 各技能 scripts/ 目录下的 @tool 脚本工具
            registry = get_skill_registry()
            if registry:
                for sid, manifest in registry.skills.items():
                    skill_dir = registry.get_skill_dir(sid)
                    if skill_dir:
                        final_tools.extend(load_skill_scripts(manifest, skill_dir))
        except Exception as exc:
            logger.debug("预加载技能工具失败：%s", exc)

    logger.info(
        "正在创建 Agent %r，共 %d 个中间件，%d 个工具。",
        name,
        len(chain),
        len(final_tools),
    )

    # ── 构建内部 ReAct Agent ──
    agent = create_react_agent(
        model=model,
        tools=final_tools,
        state_schema=state_schema,
        prompt=system_prompt or None,
        checkpointer=checkpointer,
        name=name,
        **langgraph_kwargs,
    )
    # M12: 猴子补丁挂载中间件链。
    # 注意：序列化/deepcopy 可能丢失此属性；若需序列化，应将链存储在外部注册表中。
    agent._uniagent_middleware = chain  # type: ignore[attr-defined]

    # ── 解析预算 ──
    resolved_budget: Budget
    if isinstance(budget, Budget):
        resolved_budget = budget
    elif isinstance(budget, BudgetConfig):
        resolved_budget = Budget(config=budget)
    else:
        resolved_budget = Budget()

    # ── 解析循环钩子 ──
    all_hooks = list(loop_hooks) if loop_hooks else feat.default_loop_hooks()

    # ── 按需包装为循环（三种模式）──

    if goal is not None:
        # 模式 3：GoalLoop — 验证驱动的自主执行
        if verifier is None:
            logger.warning(
                "GoalLoop 在没有验证器的情况下创建 — 使用 AlwaysPassVerifier。"
                "这使目标驱动执行失去了意义。"
            )
            verifier = AlwaysPassVerifier()

        return GoalLoop(
            agent=agent,
            goal=goal,
            verifier=verifier,
            hooks=all_hooks,
            budget=resolved_budget,
        )

    if feat.goal_loop:
        # 模式 2：TurnLoop — 通过特性启用了目标循环，但未设置目标
        # 调用 .run() 并传入 goal 时将动态创建 GoalLoop
        return TurnLoop(
            agent=agent,
            hooks=all_hooks,
            budget=resolved_budget,
        )

    # 模式 1：裸 Agent — 直接返回 CompiledGraph
    return agent
