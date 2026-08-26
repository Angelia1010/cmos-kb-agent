# -*- coding: utf-8 -*-
"""uniagent 框架端到端演示测试 —— 使用真实大模型（model_config.yaml）。

本文件从零搭建一个最小可运行的 uniagent 智能体，不依赖 kbagent 业务代码。
覆盖框架全部核心层：

    ┌──────────────────────────────────────────────────────┐
    │  第1层：LLM + 工具                                    │
    │    真实大模型（model_config.yaml 配置）                │
    │    @tool 计算器/天气查询 (模拟工具)                      │
    ├──────────────────────────────────────────────────────┤
    │  第2层：中间件 (洋葱模型)                               │
    │    before_agent → [agent 推理] → after_agent          │
    │    LLMLoggingMiddleware（统一日志，三层覆盖）               │
    ├──────────────────────────────────────────────────────┤
    │  第3层：循环引擎                                       │
    │    TurnLoop  — 固定 N 轮迭代                           │
    │    GoalLoop  — 目标驱动 + 验证器 + 预算                  │
    ├──────────────────────────────────────────────────────┤
    │  第4层：钩子 & 预算                                    │
    │    LoopHook (生命周期事件)                              │
    │    Budget   (迭代/时间硬限制)                           │
    └──────────────────────────────────────────────────────┘

运行方式:
    PYTHONPATH=src python test_uniagent_e2e.py

按 Enter 逐步推进，Ctrl+C 随时中止。
"""
from __future__ import annotations

# ── 标准库 ──────────────────────────────────────────────────────────────────
import asyncio
import logging
import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, List, Optional

# 自动将 src/ 加入模块搜索路径，无需手动设置 PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

# ── 第三方 ───────────────────────────────────────────────────────────────────
import yaml

# ── LangChain 基础类型 ───────────────────────────────────────────────────────
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

# ── uniagent 公开 API ────────────────────────────────────────────────────────
from uniagent import (
    AgentFeatures,
    AgentTrace,
    Budget,
    BudgetConfig,
    GoalLoop,
    LoopResult,
    TraceMiddleware,
    TurnLoop,
    VerificationResult,
    agent_trace,
    create_agent,
    run_traced,
)
from uniagent.agents.config_factory import (
    get_skill_registry,
    register_skill_directory,
    reset_skill_registry,
)
from uniagent.config.sub_configs import ModelConfig
from uniagent.middleware.base import Middleware
from uniagent.middleware.builtins import LLMLoggingMiddleware
from uniagent.models.factory import build_model
from uniagent.runtime.hooks import LoopHook
from uniagent.runtime.signals import HookResponse, LoopSignal
from uniagent.skills import SkillManifest

# ── 日志配置 ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,          # 调试时关掉框架 INFO 日志，只看断点输出
    format="%(name)s | %(message)s",
    stream=sys.stdout,
)


# =====================================================================
# 真实大模型初始化 —— 从 model_config.yaml 加载
# =====================================================================

_MODEL_CONFIG_FILE = Path(__file__).parent / "model_config.yaml"


def _init_model() -> BaseChatModel:
    if not _MODEL_CONFIG_FILE.exists():
        raise FileNotFoundError(f"未找到：{_MODEL_CONFIG_FILE}")
    raw = yaml.safe_load(_MODEL_CONFIG_FILE.read_text(encoding="utf-8")) or {}
    if not raw.get("use") or not raw.get("model"):
        raise ValueError("model_config.yaml 缺少 'use' 或 'model' 字段。")
    mc = ModelConfig(**raw)
    model = build_model(mc)
    print(f"[模型] {mc.use}  model={mc.model}\n")
    return model


MODEL: BaseChatModel = _init_model()


# =====================================================================
# 工具定义（带调试输出）
# =====================================================================

def _dbg(label: str) -> None:
    print(f"\n{'━'*60}")
    print(f"  {label}")
    print(f"{'━'*60}")


def _pause(hint: str = "") -> None:
    """按 Enter 继续。"""
    if hint:
        print(f"  {hint}")
    input("  ↵ 按 Enter 继续...\n")


@tool
def calculator(expression: str) -> str:
    """计算数学表达式并返回结果。"""
    _pause("↑ 确认参数后，按 Enter 执行工具")
    try:
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算出错：{e}"


@tool
def weather(city: str) -> str:
    """查询指定城市的天气。"""
    _pause("↑ 确认参数后，按 Enter 执行工具")
    fake_data = {"北京": "晴 32°C", "上海": "多云 28°C", "深圳": "雷阵雨 30°C"}
    return f"{city}天气：{fake_data.get(city, '未知城市')}"


# =====================================================================
# 日志说明
# =====================================================================
# 旧方式（已废弃，仅保留注释说明）：
#   - VerboseCallback(BaseCallbackHandler)：散落在测试文件中，只能手动注入裸 Agent
#   - DebugMiddleware(Middleware)：混入 _pause() 交互逻辑，不可复用
#   - LoggingMiddleware(Middleware)：仅打印消息数，信息量不足
#
# 新方式（统一使用 LLMLoggingMiddleware）：
#   - 三层覆盖：LLM调用级（完整prompt/response）+ state级 + 循环层
#   - TurnLoop/GoalLoop：通过 get_invoke_config 自动注入 callback，无需手动配置
#   - 裸 Agent：调用 mw.as_langchain_callback() 手动传入
#   - 通过 verbose=True 控制详细程度，通过 log_level 控制输出级别
# =====================================================================


class KeywordVerifier:
    def __init__(self, keyword: str):
        self.keyword = keyword
        self.call_count = 0

    async def verify(self, goal: str, state: dict) -> VerificationResult:
        self.call_count += 1
        messages = state.get("messages", [])
        all_text = " ".join(str(getattr(m, "content", "")) for m in messages)
        found = self.keyword in all_text
        print(f"  [验证器] 第{self.call_count}次，查找 '{self.keyword}': {'✓' if found else '✗'}")
        return VerificationResult(
            passed=found,
            evidence=f"关键词 '{self.keyword}' {'已' if found else '未'}出现",
        )


class DebugHook(LoopHook):
    """自定义 LoopHook 示例，用于演示钩子体系的用法。

    注意：LLMLoggingMiddleware 通过 loop_hooks() 已内置了类似的循环层日志，
    生产环境优先使用 LLMLoggingMiddleware 而非手动定义此类。
    """

    name = "debug_hook"

    async def on_iteration_start(self, iteration: int, state: dict) -> HookResponse:
        print(f"\n{'='*60}\n  [钩子] 第 {iteration + 1} 轮迭代开始\n{'='*60}")
        return HookResponse()

    async def on_iteration_end(self, iteration: int, state: dict, agent_output: dict | None) -> HookResponse:
        print(f"  [钩子] 第 {iteration + 1} 轮迭代结束")
        return HookResponse()

    async def on_goal_achieved(self, state: dict, evidence: str) -> None:
        print(f"\n  [钩子] 目标达成！证据: {evidence}")

    async def on_budget_exhausted(self, state: dict, reason: str) -> None:
        print(f"\n  [钩子] 预算耗尽: {reason}")

    async def on_error(self, state: dict, error: Exception) -> HookResponse:
        print(f"  [钩子] 出错: {error}")
        return HookResponse(signal=LoopSignal.BREAK, message=str(error))


# =====================================================================
# IterationTrackerHook —— 记录每轮迭代的消息增量，供事后逐轮回溯
# =====================================================================

class IterationTrackerHook(LoopHook):
    """记录 GoalLoop / TurnLoop 每轮迭代的消息增量，供事后逐轮回溯。

    用法::

        tracker = IterationTrackerHook()
        loop = create_agent(..., loop_hooks=[tracker])
        result = asyncio.run(loop.run(...))

        tracker.print_summary()   # 打印逐轮汇总表
        tracker.print_detail(n)   # 打印第 n 轮详细消息
    """

    name = "iteration_tracker"

    def __init__(self) -> None:
        self.records: list[dict] = []
        self._count_before: int = 0

    async def on_iteration_start(
        self, iteration: int, state: dict
    ) -> HookResponse:
        self._count_before = len(state.get("messages", []))
        return HookResponse()

    async def on_iteration_end(
        self, iteration: int, state: dict, agent_output: dict | None
    ) -> HookResponse:
        msgs = state.get("messages", [])
        new_msgs = msgs[self._count_before:]   # 本轮新增的消息

        # 统计工具调用
        tool_calls_made: list[str] = []
        for m in new_msgs:
            tc = getattr(m, "tool_calls", None)
            if tc:
                tool_calls_made.extend(x["name"] for x in tc)

        # 最后一条 AIMessage 内容
        last_ai = ""
        for m in reversed(new_msgs):
            if type(m).__name__ == "AIMessage":
                last_ai = str(getattr(m, "content", ""))
                break

        self.records.append({
            "iteration": iteration + 1,
            "new_msgs": new_msgs,
            "tool_calls": tool_calls_made,
            "last_ai_content": last_ai,
            "total_tokens": state.get("token_usage", {}).get("total_tokens", "?"),
            "goal_achieved": False,
            "budget_exhausted": False,
        })
        return HookResponse()

    async def on_goal_achieved(self, state: dict, evidence: str) -> None:
        if self.records:
            self.records[-1]["goal_achieved"] = True
            self.records[-1]["evidence"] = evidence

    async def on_budget_exhausted(self, state: dict, reason: str) -> None:
        if self.records:
            self.records[-1]["budget_exhausted"] = True

    # ── 打印方法 ──────────────────────────────────────────────────────

    def print_summary(self) -> None:
        """打印每轮迭代的单行汇总表。"""
        total = len(self.records)
        print(f"\n  {'─'*60}")
        print(f"  迭代汇总（共 {total} 轮）")
        print(f"  {'─'*60}")
        for rec in self.records:
            n        = rec["iteration"]
            new_n    = len(rec["new_msgs"])
            tools    = rec["tool_calls"]
            tokens   = rec["total_tokens"]
            status   = "✓ 目标达成" if rec.get("goal_achieved") else \
                       "✗ 预算耗尽" if rec.get("budget_exhausted") else "→ 继续"
            tools_str = f"  工具={tools}" if tools else ""
            print(f"    第{n}轮  新增消息={new_n}{tools_str}  累计tokens={tokens}  {status}")
        print(f"  {'─'*60}")

    def print_detail(self, iteration: int) -> None:
        """打印指定轮次（1-based）的完整消息增量。"""
        rec = next((r for r in self.records if r["iteration"] == iteration), None)
        if rec is None:
            print(f"  [Tracker] 第{iteration}轮记录不存在（共{len(self.records)}轮）")
            return

        new_msgs = rec["new_msgs"]
        print(f"\n  {'─'*60}")
        print(f"  第 {iteration} 轮迭代详细消息（新增 {len(new_msgs)} 条）")
        print(f"  {'─'*60}")
        for i, m in enumerate(new_msgs):
            role    = type(m).__name__
            content = str(getattr(m, "content", ""))
            tc      = getattr(m, "tool_calls", None)
            tm_id   = getattr(m, "tool_call_id", None)

            print(f"\n  [{i}] {role}")
            if tc:
                print(f"    tool_calls: {[x['name'] for x in tc]}")
                for x in tc:
                    print(f"      • {x['name']}  args={x.get('args', {})}")
            if tm_id:
                print(f"    tool_call_id: {tm_id}")
            if content:
                limit = 400
                display = content if len(content) <= limit else content[:limit] + "…(截断)"
                for line in display.splitlines():
                    print(f"    {line}")
        print(f"\n  累计 token={rec['total_tokens']}")
        if rec.get("goal_achieved"):
            print(f"  ✓ 本轮目标达成，证据: {rec.get('evidence', '')[:100]}")
        print(f"  {'─'*60}")

    def print_all_details(self) -> None:
        """依次打印所有轮次的详细消息。"""
        for rec in self.records:
            self.print_detail(rec["iteration"])
            _pause(f"第{rec['iteration']}轮详情查看完毕")


# =====================================================================
# 场景1：裸 Agent —— 逐步调试版
# =====================================================================
#
# 执行流程（含断点位置）:
#
#   [BP-1] 准备创建 Agent（查看传入参数）
#      ↓  create_agent() ← 组装中间件链 + 创建 LangGraph ReAct Graph
#   [BP-2] Agent 创建完成（查看 agent 对象）
#      ↓  构建初始 state
#   [BP-3] 即将调用 ainvoke（查看初始消息）
#      ↓  asyncio.run(agent.ainvoke(state))
#         ↓  LangGraph ReAct 循环 第1轮:
#            LLM 推理 → 决定调用工具
#            [BP-TOOL-WEATHER] weather(city="北京") 被调用
#            [BP-TOOL-CALC]   calculator(expression="100+200") 被调用
#         ↓  LangGraph ReAct 循环 第2轮:
#            LLM 推理 → 整合工具结果，生成最终回答（无 tool_calls）
#            循环结束
#   [BP-4] ainvoke 完成（查看全部消息）
#
# 说明：
#   • 裸 Agent（无 TurnLoop/GoalLoop）时 DebugMiddleware 不触发。
#     如需看中间件断点，请运行 demo_turn_loop（使用 TurnLoop 包装）。
#   • 工具断点（BP-TOOL-*）在 LLM 决定调用该工具后立即触发。

def demo_bare_agent() -> None:
    print("\n" + "="*70)
    print("  场景1：裸 Agent（LLMLoggingMiddleware 手动注入 callback）")
    print("="*70)
    print("""
  说明：
    裸 Agent 不经过 _invoke_agent，中间件 before_agent/after_agent 不触发。
    但 LLMLoggingMiddleware.as_langchain_callback() 可手动注入到 ainvoke config，
    从而捕获完整的 LLM prompt / response / 工具调用信息。

  断点导航:
    BP-1  → 创建 Agent 前（查看传入配置）
    BP-2  → Agent 创建后（查看 agent 对象结构）
    BP-3  → ainvoke 前（查看初始消息）
    BP-4  → ainvoke 完成（查看全部输出消息）
    （LLM prompt/response/工具 日志由 LLMLoggingMiddleware callback 自动打印）
""")

    # ──────────────────────────────────────────────────────────────────
    # BP-1：创建 Agent 前
    # ──────────────────────────────────────────────────────────────────
    _dbg("【BP-1】即将调用 create_agent()")
    print(f"  model   = {type(MODEL).__name__}  (model_id={MODEL.model})")
    print(f"  tools   = [calculator, weather]")
    print(f"  middleware = [LLMLoggingMiddleware(verbose=True)]")
    print(f"  注意：裸 Agent 的中间件钩子(before/after_agent)不触发，")
    print(f"        但 as_langchain_callback() 仍可捕获 LLM 调用细节。")
    print()
    _pause()

    logging_mw = LLMLoggingMiddleware(verbose=True, log_level=logging.INFO)
    agent = create_agent(
        model=MODEL,
        tools=[calculator, weather],
        middleware=[logging_mw],
        system_prompt="你是一个 helpful 助手。",
        name="demo_bare_agent",
    )

    # ──────────────────────────────────────────────────────────────────
    # BP-2：Agent 创建完成
    # ──────────────────────────────────────────────────────────────────
    _dbg("【BP-2】create_agent() 返回，Agent 已组装完成")
    print(f"  agent 类型     = {type(agent).__name__}")
    mw_chain = getattr(agent, "_uniagent_middleware", [])
    print(f"  中间件链       = {[type(m).__name__ for m in mw_chain]}")
    print(f"  裸 Agent：before_agent/after_agent 不触发，")
    print(f"            但 as_langchain_callback() 注入后可捕获 LLM 调用。")
    print()
    _pause()

    # ──────────────────────────────────────────────────────────────────
    # 构建初始 state
    # ──────────────────────────────────────────────────────────────────
    state = {
        "messages": [HumanMessage(content="帮我查北京天气，再算100+200")],
    }

    # ──────────────────────────────────────────────────────────────────
    # BP-3：ainvoke 前
    # ──────────────────────────────────────────────────────────────────
    _dbg("【BP-3】即将调用 agent.ainvoke(state)")
    print(f"  初始 state['messages']:")
    for i, m in enumerate(state["messages"]):
        print(f"    [{i}] {type(m).__name__}: {m.content!r}")
    print()
    print("  LLM prompt/response/工具调用日志将由 LLMLoggingMiddleware 自动输出。")
    print()
    _pause()

    # ── 运行 Agent（通过 as_langchain_callback() 注入日志 callback）──
    result = asyncio.run(
        agent.ainvoke(
            state,
            config={"callbacks": [logging_mw.as_langchain_callback()]},
        )
    )

    # ──────────────────────────────────────────────────────────────────
    # BP-4：ainvoke 完成
    # ──────────────────────────────────────────────────────────────────
    messages = result["messages"]
    _dbg(f"【BP-4】agent.ainvoke() 完成，共产生 {len(messages)} 条消息")
    print(f"  完整消息序列:")
    for i, msg in enumerate(messages):
        role = type(msg).__name__
        content = str(getattr(msg, "content", ""))
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "") for b in content if isinstance(b, dict)
            )
        tc = getattr(msg, "tool_calls", None)
        tc_info = f"  →工具调用: {[x['name'] for x in tc]}" if tc else ""
        print(f"    [{i}] {role:15}: {content[:80]!r}{tc_info}")
    print()
    _pause()

    # ── 汇总 ─────────────────────────────────────────────────────────
    print(f"\n  [完成] 裸 Agent 模式：before/after_agent 未触发（预期），")
    print(f"         LLM 调用级日志通过 as_langchain_callback() 捕获。")


# =====================================================================
# 其他场景（暂不运行，保留供后续使用）
# =====================================================================

def demo_turn_loop() -> None:
    """场景2：TurnLoop —— LLMLoggingMiddleware 三层全部生效。"""
    print("\n" + "="*70)
    print("  场景2：TurnLoop（LLMLoggingMiddleware 在此场景三层全部生效）")
    print("="*70)
    print("""
  说明：
    TurnLoop 通过 _invoke_agent 调用中间件链：
    - before_agent / after_agent：每轮推理前后的 state 快照（state 级）
    - get_invoke_config callback：完整 LLM prompt/response（LLM调用级）
    - loop_hooks：迭代开始/结束（循环层）
""")

    logging_mw = LLMLoggingMiddleware(verbose=True, log_level=logging.INFO)
    loop = create_agent(
        model=MODEL,
        tools=[calculator, weather],
        middleware=[logging_mw],
        goal_loop=False,            # 无目标 → TurnLoop
        budget=Budget(config=BudgetConfig(max_iterations=3, max_time_seconds=60)),
        name="demo_turn_loop",
    )

    result: LoopResult = asyncio.run(loop.run(
        input_messages=[{"role": "user", "content": "查北京天气，再算100+200"}],
        thread_id="turn-loop-001",
    ))

    print(f"\n  success={result.success}  iterations={result.iterations}  reason={result.reason}")
    print(f"  LLMLoggingMiddleware before_agent 触发次数: {logging_mw._call_count}")


def demo_goal_loop_success() -> None:
    print("\n" + "="*70)
    print("  场景3a：GoalLoop 目标达成（含 LLMLoggingMiddleware）")
    print("="*70)

    logging_mw = LLMLoggingMiddleware(log_level=logging.INFO)
    verifier = KeywordVerifier(keyword="300")
    loop = create_agent(
        model=MODEL,
        tools=[calculator, weather],
        features=AgentFeatures(loop_detection=False, skill=False,
                               logging=logging_mw),
        goal="查询北京天气并计算100+200，确保输出中包含计算结果300",
        verifier=verifier,
        budget=Budget(config=BudgetConfig(max_iterations=5, max_time_seconds=60)),
        name="demo_goal_loop",
    )

    result: LoopResult = asyncio.run(loop.run(
        input_messages=[{"role": "user", "content": "请完成任务"}],
        thread_id="goal-debug-001",
    ))

    print(f"\n  success={result.success}  iterations={result.iterations}")
    print(f"  reason={result.reason}")
    print(f"  evidence={result.evidence}")


def demo_goal_loop_budget_exhausted() -> None:
    print("\n" + "="*70)
    print("  场景3b：GoalLoop 预算耗尽（含 LLMLoggingMiddleware）")
    print("="*70)

    logging_mw = LLMLoggingMiddleware(log_level=logging.INFO)
    verifier = KeywordVerifier(keyword="不可能出现的关键词XYZ_IMPOSSIBLE")
    loop = create_agent(
        model=MODEL,
        tools=[calculator, weather],
        features=AgentFeatures(loop_detection=False, skill=False,
                               logging=logging_mw),
        goal="完成一个不可能的任务",
        verifier=verifier,
        budget=Budget(config=BudgetConfig(max_iterations=2, max_time_seconds=60)),
        name="demo_exhausted",
    )

    result: LoopResult = asyncio.run(loop.run(
        input_messages=[{"role": "user", "content": "请完成任务"}],
        thread_id="goal-debug-002",
    ))

    print(f"\n  success={result.success}  iterations={result.iterations}  reason={result.reason}")


def demo_middleware_order() -> None:
    print("\n" + "="*70)
    print("  场景4：中间件链执行顺序")
    print("="*70)
    print("""
  演示：LLMLoggingMiddleware + 自定义中间件的洋葱模型执行顺序。
  LLMLoggingMiddleware 置于链首，before 正序、after 逆序。
""")

    execution_log: list[str] = []

    class MwA(Middleware):
        name = "mw_a"
        async def before_agent(self, state: dict) -> dict | None:
            execution_log.append("A.before")
            print("    [MwA] before_agent")
            return None
        async def after_agent(self, state: dict) -> dict | None:
            execution_log.append("A.after")
            print("    [MwA] after_agent")
            return None

    class MwB(Middleware):
        name = "mw_b"
        async def before_agent(self, state: dict) -> dict | None:
            execution_log.append("B.before")
            print("    [MwB] before_agent")
            return None
        async def after_agent(self, state: dict) -> dict | None:
            execution_log.append("B.after")
            print("    [MwB] after_agent")
            return None

    logging_mw = LLMLoggingMiddleware(log_level=logging.INFO)
    verifier = KeywordVerifier(keyword="2")
    loop = create_agent(
        model=MODEL,
        tools=[calculator],
        middleware=[logging_mw, MwA(), MwB()],
        goal="计算1+1，在回答中包含数字2",
        verifier=verifier,
        budget=Budget(config=BudgetConfig(max_iterations=3, max_time_seconds=60)),
        name="demo_mw_order",
    )

    result: LoopResult = asyncio.run(loop.run(
        input_messages=[{"role": "user", "content": "帮我算1+1"}],
        thread_id="mw-order-001",
    ))

    print(f"\n  执行日志(MwA/MwB): {execution_log}")
    if execution_log:
        print(f"  洋葱顺序: {'→'.join(execution_log)}")
    print(f"  （LLMLoggingMiddleware 日志输出在上方）")


def demo_full_pipeline() -> None:
    print("\n" + "="*70)
    print("  场景5：完整管线（全部组件串联，含 LLMLoggingMiddleware）")
    print("="*70)
    print("""
  演示：通过 AgentFeatures.logging=LLMLoggingMiddleware(...) 注入日志中间件，
  与内置中间件（dangling/tool_error/loop_detection/token_usage）共同工作。
  日志输出：完整 prompt/response + state 快照 + 迭代事件，三层覆盖。
""")

    verifier = KeywordVerifier(keyword="300")
    loop = create_agent(
        model=MODEL,
        tools=[calculator, weather],
        features=AgentFeatures(
            logging=LLMLoggingMiddleware(verbose=True, log_level=logging.INFO),
            dangling_tool_call=True,
            tool_error_handling=True,
            loop_detection=True,
            token_usage=True,
            skill=False,
        ),
        goal="查询北京天气并计算100+200，确保回答包含数字300",
        verifier=verifier,
        budget=Budget(config=BudgetConfig(max_iterations=3, max_time_seconds=120)),
        system_prompt="你是一个全能助手，请完成用户交代的任务。",
        name="demo_full_pipeline",
    )

    result = asyncio.run(loop.run(
        input_messages=[{"role": "user", "content": "查北京天气并算100+200"}],
        thread_id="full-pipeline-001",
    ))

    print(f"\n  success={result.success}  iterations={result.iterations}")
    print(f"  token用量={result.final_state.get('token_usage', {})}")


def demo_skill_system() -> None:
    print("\n" + "="*70)
    print("  场景6：Skill 技能系统（逐步调试版）")
    print("="*70)
    print("""
  执行流程（断点位置）：

    [BP-1] 扫描 skills/ 目录 → 查看注册的技能包
    [BP-2] 检查技能触发器 → 手动预演 SkillMiddleware 的匹配逻辑
    [BP-3] 创建 GoalLoop Agent → 查看中间件链（含 SkillMiddleware）
    [BP-4] 即将运行 loop.run() → 查看初始消息
       ↓  GoalLoop 第1轮迭代:
          SkillMiddleware.before_agent → 触发器命中 → 注入 SystemMessage（SKILL.md）
          LLM 推理 → 可能调用脚本工具 validate_taocan_price / calculator
          GoalLoop 验证 → 未通过则继续
       ↓  GoalLoop 后续轮次（如需）...
    [BP-5] loop.run() 完成 → 逐条查看消息序列，定位 SKILL 注入位置
    [BP-6] 最终汇总
""")

    project_dir = Path(__file__).parent
    skills_root = project_dir / "skills"

    # 用于测试的用户输入（触发 taocan-skill 的关键词："99元套餐"）
    USER_INPUT = "帮我查一下99元套餐的年费是多少"
    GOAL = "查询99元套餐的年费，在回答中给出具体数字（如1188元）"
    KEYWORD = "1188"

    try:
        # ──────────────────────────────────────────────────────────────
        # BP-1：扫描 skills/ 目录
        # ──────────────────────────────────────────────────────────────
        _dbg("【BP-1】扫描 skills/ 目录")
        print(f"  skills_root = {skills_root}")
        print(f"  目录存在: {skills_root.exists()}")
        if skills_root.exists():
            subdirs = [d.name for d in skills_root.iterdir() if d.is_dir()]
            print(f"  子目录: {subdirs}")
        print()
        _pause("接下来执行 reset_skill_registry() + register_skill_directory()")

        reset_skill_registry()
        count = register_skill_directory(str(skills_root))
        registry = get_skill_registry()

        print(f"\n  本次新增技能数: {count}")
        print(f"  注册表技能列表:")
        for sid, manifest in registry.skills.items():
            print(f"    [{sid}]  name={manifest.name}  tags={manifest.tags}")
            print(f"      description: {manifest.description}")
            print(f"      triggers({len(manifest.triggers)}):")
            for t in manifest.triggers:
                print(f"        - type={t.type}  value={t.value!r}")
            print(f"      references({len(manifest.references)}):")
            for r in manifest.references:
                print(f"        - {r.filename}  when={r.when}  desc={r.description}")
            print(f"      scripts: {manifest.scripts}")
            print(f"      promoted_tools: {manifest.promoted_tools}")
        _pause()

        # ──────────────────────────────────────────────────────────────
        # BP-2：手动预演触发器匹配（SkillMiddleware 内部逻辑）
        # ──────────────────────────────────────────────────────────────
        _dbg("【BP-2】手动预演触发器匹配")
        print(f"  用户输入: {USER_INPUT!r}")
        print()
        matches = registry.match(USER_INPUT, max_results=3)
        if matches:
            print(f"  ✓ 命中 {len(matches)} 个技能:")
            for m in matches:
                print(f"    技能: {m.manifest.name}  score={m.score:.2f}")
                if m.matched_trigger:
                    print(f"    匹配触发器: type={m.matched_trigger.type}  value={m.matched_trigger.value!r}")
            top = matches[0]
            print()
            print(f"  最高得分技能: {top.manifest.name}  score={top.score:.2f}")
            print(f"  SkillMiddleware 阈值=0.3，{'将注入' if top.score >= 0.3 else '不注入（低于阈值）'} SKILL.md")
        else:
            print(f"  ✗ 无技能命中（触发器未匹配）")
        print()
        print("  SkillMiddleware 注入后消息序列预览（注入发生在 before_agent 内）：")
        print("    [0] HumanMessage:  用户原始输入")
        print("    [1] SystemMessage: <!-- SKILL: taocan-skill -->\\n{SKILL.md内容}")
        print("         ↑ 这是 SkillMiddleware 追加的技能指令")
        _pause()

        # ──────────────────────────────────────────────────────────────
        # BP-3：创建 GoalLoop Agent
        # ──────────────────────────────────────────────────────────────
        _dbg("【BP-3】即将调用 create_agent()")
        print(f"  model   = {type(MODEL).__name__}  (model_id={MODEL.model})")
        print(f"  tools   = [calculator]  ← 脚本工具(validate_taocan_price)由 SkillMiddleware 预加载")
        print(f"  features:")
        print(f"    logging          = LLMLoggingMiddleware(verbose=True)  ← 三层日志")
        print(f"    skill            = True   ← 启用 SkillMiddleware")
        print(f"    dangling_tool_call = True")
        print(f"    tool_error_handling= True")
        print(f"    loop_detection   = False  ← 关闭（避免干扰调试）")
        print(f"  loop_hooks = [IterationTrackerHook()]  ← 逐轮消息增量记录")
        print(f"  goal    = {GOAL!r}")
        print(f"  verifier= KeywordVerifier(keyword={KEYWORD!r})")
        print(f"  budget  = max_iterations=5, max_time_seconds=120")
        _pause()

        logging_mw = LLMLoggingMiddleware(verbose=True, log_level=logging.INFO)
        tracker    = IterationTrackerHook()
        loop = create_agent(
            model=MODEL,
            tools=[calculator],
            features=AgentFeatures(
                logging=logging_mw,
                skill=True,
                dangling_tool_call=True,
                tool_error_handling=True,
                loop_detection=False,
            ),
            goal=GOAL,
            verifier=KeywordVerifier(keyword=KEYWORD),
            budget=Budget(config=BudgetConfig(max_iterations=5, max_time_seconds=120)),
            loop_hooks=[tracker],
            name="demo_skill_agent",
        )

        # 查看组装后的中间件链和注册的循环钩子
        agent_obj = loop._agent
        chain = getattr(agent_obj, "_uniagent_middleware", [])
        all_hooks = getattr(loop, "_hooks", [])
        _dbg("【BP-3 续】GoalLoop Agent 已创建")
        print(f"  loop 类型: {type(loop).__name__}")
        print(f"  内部 agent 类型: {type(agent_obj).__name__}")
        print(f"  中间件链 ({len(chain)} 个):")
        for i, mw in enumerate(chain):
            extra = "  (内置 VerboseCallback，get_invoke_config 自动注入)" \
                    if hasattr(mw, "_callback") else ""
            extra += f"  → loop_hooks={[type(h).__name__ for h in mw.loop_hooks()]}" \
                     if mw.loop_hooks() else ""
            print(f"    [{i}] {type(mw).__name__}{extra}")
        print()
        print(f"  循环层钩子 ({len(all_hooks)} 个):")
        for h in all_hooks:
            print(f"    • {h.name}")
        print()
        print("  洋葱顺序（before 正序 / after 逆序）：")
        print(f"    before: {' → '.join(type(m).__name__ for m in chain)}")
        print(f"    after : {' → '.join(type(m).__name__ for m in reversed(chain))}")
        _pause()

        # ──────────────────────────────────────────────────────────────
        # BP-4：即将运行 loop.run()
        # ──────────────────────────────────────────────────────────────
        _dbg("【BP-4】即将调用 loop.run()")
        print(f"  初始 input_messages:")
        print(f"    [0] HumanMessage: {USER_INPUT!r}")
        print()
        print("  GoalLoop 执行流程：")
        print("    ① 在消息列表前插入 SystemMessage([目标]...)")
        print("    ② 每轮迭代：")
        print("       [钩子] IterationTrackerHook.on_iteration_start  → 记录消息基线")
        print("       [中间件] SkillMiddleware.before_agent            → 触发器匹配→注入 SKILL.md")
        print("       [中间件] LLMLoggingMiddleware.before_agent       → 打印 state 快照")
        print("       [LLM]   agent.ainvoke（VerboseCallback 记录完整 prompt/response）")
        print("       [中间件] LLMLoggingMiddleware.after_agent        → 打印最新 AIMessage")
        print("       [钩子] IterationTrackerHook.on_iteration_end     → 记录消息增量")
        print("       [验证] KeywordVerifier 检查是否出现 '1188'")
        print("    ③ 验证通过 → on_goal_achieved → 退出 | 未通过 → 注入反馈 → 下一轮")
        _pause("接下来实际执行，日志自动输出...")

        # ── 实际运行 ──────────────────────────────────────────────────
        result = asyncio.run(loop.run(
            input_messages=[HumanMessage(content=USER_INPUT)],
            thread_id="skill-demo-001",
        ))

        # ──────────────────────────────────────────────────────────────
        # BP-5：逐条查看消息序列
        # ──────────────────────────────────────────────────────────────
        messages = result.final_state.get("messages", [])
        _dbg(f"【BP-5】loop.run() 完成，消息序列共 {len(messages)} 条")
        print(f"  逐条消息（带 SKILL 注入标记）：")
        for i, msg in enumerate(messages):
            role = type(msg).__name__
            content = str(getattr(msg, "content", ""))
            tc = getattr(msg, "tool_calls", None)

            if "<!-- SKILL:" in content:
                # SKILL 注入消息：展开全文
                skill_name = content.split("<!-- SKILL:")[1].split("-->")[0].strip()
                print(f"    [{i}] {role:20}:  ← ★ SKILL 注入（{skill_name}，共{len(content)}字符）")
                print(f"  {'─'*64}")
                for line in content.splitlines():
                    print(f"  │  {line}")
                print(f"  {'─'*64}")
            elif content.startswith("[目标]"):
                print(f"    [{i}] {role:20}: {content[:80]!r}…  ← GoalLoop 注入目标")
            elif content.startswith("[验证失败]"):
                print(f"    [{i}] {role:20}: {content[:80]!r}…  ← GoalLoop 验证反馈")
            elif tc:
                print(f"    [{i}] {role:20}: {content[:60]!r}  ← tool_calls: {[x['name'] for x in tc]}")
            else:
                print(f"    [{i}] {role:20}: {content[:80]!r}")
        _pause()

        # ──────────────────────────────────────────────────────────────
        # BP-5.5：逐轮迭代详细信息（IterationTrackerHook）
        # ──────────────────────────────────────────────────────────────
        _dbg(f"【BP-5.5】IterationTracker — 共执行 {len(tracker.records)} 轮迭代")
        tracker.print_summary()
        _pause("查看汇总完毕，按 Enter 查看每轮迭代的详细消息增量...")
        tracker.print_all_details()

        # ──────────────────────────────────────────────────────────────
        # BP-6：最终汇总
        # ──────────────────────────────────────────────────────────────
        skill_injected = any("<!-- SKILL:" in str(getattr(m, "content", "")) for m in messages)
        skill_names = []
        for m in messages:
            c = str(getattr(m, "content", ""))
            if "<!-- SKILL:" in c:
                skill_names.append(c.split("<!-- SKILL:")[1].split("-->")[0].strip())

        _dbg("【BP-6】最终汇总")
        print(f"  GoalLoop 结果:")
        print(f"    success    = {result.success}")
        print(f"    iterations = {result.iterations}")
        print(f"    reason     = {result.reason}")
        print(f"    evidence   = {result.evidence}")
        print()
        print(f"  Skill 系统:")
        print(f"    技能注入     = {'✓' if skill_injected else '✗'}")
        if skill_names:
            print(f"    注入技能名   = {skill_names}")
        print(f"    中间件触发   = {logging_mw._call_count} 次 before_agent")
        print()
        if result.success:
            print(f"  ✓ 关键词 {KEYWORD!r} 已出现在最终回答中")
        else:
            print(f"  ✗ 关键词 {KEYWORD!r} 未找到，GoalLoop 预算耗尽")
        _pause()

    finally:
        reset_skill_registry()


# =====================================================================
# 场景7：结构化追踪日志 —— AgentTrace + TraceMiddleware
# =====================================================================

def demo_trace_logging() -> None:
    """场景7：结构化 JSON 追踪日志，全链路一个 AgentTrace 对象。"""
    print("\n" + "="*70)
    print("  场景7：结构化追踪日志（AgentTrace + TraceMiddleware）")
    print("="*70)
    print("""
  说明：
    TraceMiddleware 内置两种机制：
    - _TraceCallback (BaseCallbackHandler) → LLM 调用 + 工具调用写入 JSON
    - _TraceLoopHook (LoopHook)          → 迭代生命周期管理

    loop.py _invoke_agent 为每个中间件的 before/after 写入耗时记录。
    GoalLoop.run() 将每轮验证结果（passed/evidence）写入当前迭代记录。

    所有信息通过 ContextVar 在 async 上下文全局流转，无需参数穿透。

    最终输出：一个包含完整执行过程的 AgentTrace JSON 对象，
    可写入 ELK/Datadog/日志文件/数据库等监控系统。

  断点导航:
    BP-1  → 创建 Agent（含 TraceMiddleware）
    BP-2  → 即将 run_traced()（查看自动提取的 agent_config）
    BP-3  → 执行完成（查看完整 JSON 追踪日志）
""")

    # ──────────────────────────────────────────────────────────────────
    # BP-1：创建 Agent
    # ──────────────────────────────────────────────────────────────────
    _dbg("【BP-1】创建含 TraceMiddleware 的 GoalLoop Agent")
    print("  注意：TraceMiddleware 同时注册了 _TraceLoopHook，")
    print("        factory.py 会自动将其加入循环钩子列表。")
    _pause()

    verifier = KeywordVerifier(keyword="300")
    loop = create_agent(
        model=MODEL,
        tools=[calculator, weather],
        middleware=[TraceMiddleware()],          # ← 追踪中间件
        goal="查询北京天气并计算100+200，确保回答包含数字300",
        verifier=verifier,
        budget=Budget(config=BudgetConfig(max_iterations=3, max_time_seconds=60)),
        system_prompt="你是一个 helpful 助手，请完成用户交代的任务。",
        name="demo_trace_agent",
    )

    chain = getattr(loop._agent, "_uniagent_middleware", [])
    hooks = getattr(loop, "_hooks", [])
    print(f"\n  中间件链: {[type(m).__name__ for m in chain]}")
    print(f"  循环钩子: {[getattr(h, 'name', type(h).__name__) for h in hooks]}")
    _pause()

    # ──────────────────────────────────────────────────────────────────
    # BP-2：即将 run_traced()
    # ──────────────────────────────────────────────────────────────────
    _dbg("【BP-2】即将调用 run_traced()")
    print("  run_traced() 会自动：")
    print("  1. 从 loop 对象提取 agent_config（middleware / budget / hooks）")
    print("  2. 创建 AgentTrace 并设置到 ContextVar")
    print("  3. 执行 loop.run()（所有组件自动写入 AgentTrace）")
    print("  4. 调用 trace.finish(result) 补全顶层摘要")
    print("  5. 返回 (LoopResult, AgentTrace) 元组")
    _pause("接下来实际执行，LLM 日志输出已关闭（设置 TraceMiddleware 即可，无需 LLMLoggingMiddleware）...")

    # ── 实际运行（run_traced 一行完成） ──────────────────────────────
    result, trace = asyncio.run(run_traced(
        loop,
        [HumanMessage(content="帮我查北京天气，再算100+200")],
        thread_id="trace-demo-001",
        name="demo_trace_agent",
    ))

    # ──────────────────────────────────────────────────────────────────
    # BP-3：查看完整 JSON 追踪日志
    # ──────────────────────────────────────────────────────────────────
    td = trace.to_dict()
    _dbg(f"【BP-3】追踪完成 — 顶层摘要")
    print(f"  trace_id       = {td['trace_id']}")
    print(f"  agent_name     = {td['agent_name']}")
    print(f"  loop_type      = {td['loop_type']}")
    print(f"  success        = {td['success']}")
    print(f"  iterations_used= {td['iterations_used']}")
    print(f"  duration_ms    = {td['duration_ms']}")
    print(f"  reason         = {td['reason']}")
    print(f"  token_usage    = {td['token_usage']}")
    print(f"  agent_config   = {td['agent_config']}")
    _pause()

    # ── 逐轮迭代展开 ────────────────────────────────────────────────
    _dbg(f"  iterations（共 {len(td['iterations'])} 轮）")
    for it in td["iterations"]:
        print(f"\n  ┌─ 第 {it['iteration']} 轮 ────────────────────────────────")
        print(f"  │  duration_ms      = {it['duration_ms']}")
        print(f"  │  new_messages     = {it['new_messages_count']}")
        print(f"  │  verification     = {it['verification']}")
        # 中间件事件
        print(f"  │  middleware_events ({len(it['middleware_events'])} 个):")
        for ev in it["middleware_events"]:
            print(f"  │    [{ev['phase']:6}] {ev['middleware']:30} {ev['action']:8}  "
                  f"{ev['duration_ms']:.1f}ms  keys={ev['patch_keys']}")
        # LLM 调用
        print(f"  │  llm_calls ({len(it['llm_calls'])} 次):")
        for lc in it["llm_calls"]:
            tc_names = [t["name"] for t in lc["response"]["tool_calls"]]
            print(f"  │    #{lc['call_index']}  {lc['duration_ms']:.0f}ms  "
                  f"tokens={lc['tokens']}  tool_calls={tc_names}")
            print(f"  │       prompt_messages={lc['prompt']['message_count']}条")
            resp_preview = lc["response"]["content"][:80].replace("\n", "↵")
            print(f"  │       response: {resp_preview!r}")
        # 工具调用
        print(f"  │  tool_calls ({len(it['tool_calls'])} 次):")
        for tc in it["tool_calls"]:
            print(f"  │    {tc['tool']:25} {tc['duration_ms']:.0f}ms  "
                  f"in={tc['input'][:40]!r}  out={tc['output'][:40]!r}")
        print(f"  └──────────────────────────────────────────────")
    _pause("查看完整 JSON...")

    # ── 输出完整 JSON ──────────────────────────────────────────────
    _dbg("  完整 AgentTrace JSON")
    print(trace.to_json())
    _pause()

    print(f"\n  ✓ 可将 trace.to_dict() 直接推送到监控系统（ELK / Datadog / DB）")
    print(f"  ✓ 可用 trace.to_json() 写入日志文件")
    print(f"  ✓ trace_id 支持关联外部请求链路（X-Request-ID）")


# =====================================================================
# 主入口 —— 当前只运行场景1（调试完成后可取消注释其他场景）
# =====================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  uniagent 框架端到端调试                                          ║
║  按 Enter 逐步推进，Ctrl+C 随时中止                               ║
╚══════════════════════════════════════════════════════════════════╝
""")

    # ── 选择要运行的场景（取消对应注释）──────────────────────────────
    #
    # 场景1  裸 Agent          — LLMLoggingMiddleware callback 手动注入
    # 场景2  TurnLoop          — 中间件三层全部自动生效
    # 场景3a GoalLoop 成功      — 目标达成
    # 场景3b GoalLoop 超预算    — 验证故意不通过
    # 场景4  中间件顺序         — 洋葱模型执行顺序演示
    # 场景5  完整管线           — AgentFeatures 所有内置中间件 + LLMLogging
    # 场景6  Skill 技能系统     — 逐步调试触发器/注入/脚本工具全流程
    # 场景7  结构化追踪日志     — AgentTrace JSON 全链路一个对象
    # ────────────────────────────────────────────────────────────────
    DEMO = demo_bare_agent   # ← 修改此行切换场景

    try:
        DEMO()
        print(f"\n  >>> {DEMO.__name__} 执行完毕 <<<\n")
    except Exception as e:
        print(f"\n  >>> {DEMO.__name__} 出错: {e} <<<\n")
        import traceback
        traceback.print_exc()
