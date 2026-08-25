"""配置驱动的 Agent 工厂 — ``create_agent_from_config()``。"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, TYPE_CHECKING

from uniagent.agents.factory import create_agent
from uniagent.agents.features import AgentFeatures
from uniagent.config.app_config import AppConfig, get_app_config
from uniagent.middleware.base import Middleware
from uniagent.imports.resolvers import resolve_class
from uniagent.models.factory import ModelFactory as _ModelFactory
from uniagent.runtime.budget import Budget, BudgetConfig
from uniagent.runtime.hooks import LoopHook
from uniagent.tools.registry import get_available_tools

# 仅用于类型标注，运行时不导入（避免循环依赖：config_factory → skills → tools → config_factory）
if TYPE_CHECKING:
    from uniagent.skills.registry import SkillRegistry

logger = logging.getLogger(__name__)

# ── 技能注册表单例 ──
# H10: 使用 Lock 保护延迟初始化，确保多线程并发安全
_skill_registry: "SkillRegistry | None" = None
_skill_registry_lock = threading.Lock()

# ── 模型工厂单例 ──
_model_factory = _ModelFactory()


def create_agent_from_config(
    config: AppConfig | None = None,
    *,
    extra_tools: list[Any] | None = None,
    mcp_tools: list[Any] | None = None,
    features: AgentFeatures | None = None,
    system_prompt: str = "",
    checkpointer: Any = None,
    goal: str | None = None,
    verifier: Any | None = None,
    loop_hooks: list[LoopHook] | None = None,
    skill_dirs: list[str] | None = None,
    **kwargs: Any,
) -> Any:
    """完全由 ``AppConfig`` 驱动创建 Agent。

    步骤：
    1. 从配置中解析 LLM 模型。
    2. 从配置、额外工具及 MCP 加载工具。
    3. (KB 适配) 延迟工具发现已移除 — 工具集固定且少量，直接全部可用。
    4. 从配置中解析额外中间件。
    5. 解析循环配置（预算、钩子、验证器）。
    6. 初始化技能子系统（若启用）。
    7. 通过 ``create_agent()`` 构建 Agent。

    参数
    ----------
    config:
        为 None 时使用 ``get_app_config()``（支持热重载）。
    goal:
        若提供，则将 Agent 包装在 GoalLoop 中。
    verifier:
        目标驱动执行的验证器。
    loop_hooks:
        额外的循环级钩子。

    返回
    -------
    CompiledGraph | TurnLoop | GoalLoop
    """
    cfg = config or get_app_config()

    # ── 1. 解析模型 ──
    model = _resolve_model(cfg)

    # ── 2. 加载工具 ──
    all_tools = get_available_tools(
        cfg,
        extra_tools=extra_tools,
        mcp_tools=mcp_tools,
    )

    # ── 3. 工具集固定，直接使用 ──
    final_tools = all_tools

    # ── 4. 从配置中解析额外中间件 ──
    extra_mw = _resolve_extra_middleware(cfg)

    # ── 5. 特性标志 ──
    feat = features or AgentFeatures()

    # ── 6. 从配置中解析循环预算 ──
    budget = _resolve_budget(cfg)
    all_loop_hooks = list(loop_hooks) if loop_hooks else None

    # ── 7. 技能子系统初始化（按需） ──
    final_prompt = system_prompt
    if cfg.skills.enabled:
        final_prompt = _setup_skills(cfg, system_prompt, skill_dirs)

    # ── 8. 构建 Agent ──
    return create_agent(
        model=model,
        tools=final_tools,
        features=feat,
        extra_middleware=extra_mw or None,
        system_prompt=final_prompt,
        checkpointer=checkpointer,
        goal=goal,
        verifier=verifier,
        loop_hooks=all_loop_hooks,
        budget=budget,
        **kwargs,
    )


def _resolve_model(cfg: AppConfig, *, name: str = "default") -> Any:
    """根据配置实例化聊天模型。

    H11: 优先按 name 查找模型配置，找不到时回退到第一个模型。
    委托给 ModelFactory.build() 处理参数注入（api_key/base_url/timeout/…）。
    """
    if not cfg.models:
        raise ValueError(
            "未配置任何模型。请在配置的 'models' 中至少添加一个条目。"
        )
    # H11: 优先按 name 字段查找
    mc = None
    for m in cfg.models:
        model_name = getattr(m, "name", None) or ""
        if model_name == name:
            mc = m
            break
    if mc is None:
        mc = cfg.models[0]  # 回退到第一个模型
    return _model_factory.build(mc)


def _resolve_extra_middleware(cfg: AppConfig) -> list[Middleware]:
    """从配置路径加载额外的中间件类并实例化。"""
    result: list[Middleware] = []
    for path in cfg.extra_middleware:
        cls = resolve_class(path, base_class=Middleware)
        result.append(cls())
    return result


def _resolve_budget(cfg: AppConfig) -> Budget:
    """根据循环配置段构建 Budget。"""
    loop_cfg = cfg.loop
    return Budget(
        config=BudgetConfig(
            max_iterations=loop_cfg.max_iterations,
            max_tokens=loop_cfg.max_tokens,
            max_time_seconds=loop_cfg.max_time_seconds,
        )
    )


def _setup_skills(
    cfg: AppConfig, base_prompt: str, extra_dirs: list[str] | None = None
) -> str:
    """初始化技能注册表并返回增强后的系统提示。

    H10: 使用 threading.Lock 保护单例初始化（线程安全）。
    H12: 将技能列表摘要注入系统提示，让 LLM 知晓可用技能。

    NOTE: SkillRegistry 的导入保持延迟（不在文件头），原因：
      config_factory 被 skills/tools.py 在文件头导入；
      若在文件头导入 skills.registry，则触发 skills/__init__.py，
      skills/__init__.py 导入 skills/tools.py，tools.py 导入 config_factory，
      形成循环：config_factory → skills.registry → skills/__init__ → tools → config_factory。
    """
    global _skill_registry
    # 循环依赖，保持延迟导入（见上方注释）
    from uniagent.skills.registry import SkillRegistry

    # H10: 线程安全的单例初始化
    with _skill_registry_lock:
        if _skill_registry is None:
            _skill_registry = SkillRegistry()

    # 扫描配置目录及额外目录
    dirs = list(cfg.skills.directories)
    if extra_dirs:
        dirs.extend(extra_dirs)
    _skill_registry.scan(*dirs)

    logger.info(
        "技能子系统已初始化：共注册 %d 个技能。",
        len(_skill_registry.skills),
    )

    # H12: 将技能列表注入系统提示（让 LLM 了解可用技能）
    skills_info = _skill_registry.list_skills()
    if not skills_info:
        return base_prompt

    skill_lines = []
    for s in skills_info:
        skill_lines.append(f"  - {s['name']}: {s['description']}")

    skill_section = (
        "\n\n## 可用技能\n"
        "以下技能已加载，当用户消息匹配触发条件时会自动激活；\n"
        "若问题明确属于某技能领域但尚未激活，可主动调用 load_skill_reference 获取详细指导。\n"
        + "\n".join(skill_lines)
    )
    return base_prompt + skill_section


# ---------------------------------------------------------------------------
# 技能注册表公共 API
# ---------------------------------------------------------------------------

def get_skill_registry() -> "SkillRegistry | None":
    """返回全局技能注册表（如已初始化）。"""
    return _skill_registry


def register_skill_directory(directory: str | Path) -> int:
    """动态注册新的技能目录（渐进式热插拔）。

    在 Agent 运行期间动态追加技能包目录，无需重启。
    重复调用同一目录是幂等操作（由注册表内部 _scanned_dirs 去重保证）。

    返回本次新增的技能数量。
    """
    global _skill_registry
    # 循环依赖，保持延迟导入（见 _setup_skills 注释）
    from uniagent.skills.registry import SkillRegistry

    with _skill_registry_lock:
        if _skill_registry is None:
            _skill_registry = SkillRegistry()

    count = _skill_registry.scan(str(directory))
    logger.info(
        "register_skill_directory(%s): 新增 %d 个技能，注册表共 %d 个。",
        directory,
        count,
        len(_skill_registry.skills),
    )
    return count


def reset_skill_registry() -> None:
    """重置技能注册表（通常仅用于测试清理）。"""
    global _skill_registry
    with _skill_registry_lock:
        _skill_registry = None
