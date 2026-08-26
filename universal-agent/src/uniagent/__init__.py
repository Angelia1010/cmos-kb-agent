"""Universal Agent Harness — 基于 LangGraph 的中间件驱动 Agent 框架。"""

from uniagent.agents.factory import create_agent
from uniagent.agents.config_factory import create_agent_from_config
from uniagent.agents.features import AgentFeatures
from uniagent.config.app_config import AppConfig, get_app_config
from uniagent.logging.trace import AgentTrace, agent_trace, get_current_trace, run_traced
from uniagent.logging.trace_middleware import TraceMiddleware
from uniagent.models.factory import ModelFactory, build_model, get_model
from uniagent.runtime.loop import GoalLoop, TurnLoop
from uniagent.runtime.budget import Budget, BudgetConfig
from uniagent.runtime.signals import LoopResult
from uniagent.verification.verifier import Verifier, VerificationResult

__all__ = [
    "create_agent",
    "create_agent_from_config",
    "AgentFeatures",
    "AppConfig",
    "get_app_config",
    # ── 结构化追踪 ──
    "AgentTrace",
    "TraceMiddleware",
    "agent_trace",
    "get_current_trace",
    "run_traced",
    # ── 模型 ──
    "ModelFactory",
    "build_model",
    "get_model",
    # ── 循环引擎 ──
    "Budget",
    "BudgetConfig",
    "GoalLoop",
    "TurnLoop",
    "LoopResult",
    # ── 验证 ──
    "Verifier",
    "VerificationResult",
]
