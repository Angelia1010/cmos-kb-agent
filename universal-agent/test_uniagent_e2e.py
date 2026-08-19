# -*- coding: utf-8 -*-
"""uniagent 框架端到端演示测试 —— 可逐行 debug 理解整个框架。

本文件从零搭建一个最小可运行的 uniagent 智能体，不依赖 kbagent 业务代码。
覆盖框架全部核心层：

    ┌──────────────────────────────────────────────────────┐
    │  第1层：LLM + 工具                                    │
    │    FakeChatModel (模拟 LLM)                          │
    │    @tool 计算器/天气查询 (模拟工具)                      │
    ├──────────────────────────────────────────────────────┤
    │  第2层：中间件 (洋葱模型)                               │
    │    before_agent → [agent 推理] → after_agent          │
    │    自定义 LoggingMiddleware + 内置中间件                 │
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
    # 直接运行（看输出）
    PYTHONPATH=src python test_uniagent_e2e.py

    # 用 IDE 断点调试（PyCharm / VSCode）
    # 在任意 "# ★ 断点" 注释处打断点，F5 启动调试

    # 用 unittest 跑
    PYTHONPATH=src python -m unittest test_uniagent_e2e -v
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import uuid
import unittest
from typing import Any, Dict, List, Optional

# ── LangChain 基础类型 ──
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

# ── uniagent 公开 API ──
from uniagent import (
    AgentFeatures,
    Budget,
    BudgetConfig,
    GoalLoop,
    LoopResult,
    TurnLoop,
    VerificationResult,
    create_agent,
)
from uniagent.middleware.base import Middleware
from uniagent.runtime.hooks import LoopHook
from uniagent.runtime.signals import HookResponse, LoopSignal


# =====================================================================
# 第0步：日志配置 —— 打开 uniagent 内部日志，debug 时能看到完整流程
# =====================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s | %(message)s",
    stream=sys.stdout,
)

# 想看更详细的中间件/循环日志可以打开 DEBUG：
# logging.getLogger("uniagent").setLevel(logging.DEBUG)


# =====================================================================
# 第1步：准备工具 —— LLM 能调用的 @tool 函数
# =====================================================================
# 说明：uniagent 基于 LangGraph 的 ReAct 模式，
# LLM 看到工具列表后自主决定调用哪个工具、传什么参数。

@tool
def calculator(expression: str) -> str:
    """计算数学表达式并返回结果。"""
    # ★ 断点：在这里打断点，可以看到 LLM 传来的 expression 参数
    try:
        result = eval(expression, {"__builtins__": {}})  # 简化演示用
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算出错：{e}"


@tool
def weather(city: str) -> str:
    """查询指定城市的天气。"""
    # ★ 断点：在这里打断点，可以看到工具被调用
    fake_data = {"北京": "晴 32°C", "上海": "多云 28°C", "深圳": "雷阵雨 30°C"}
    return f"{city}天气：{fake_data.get(city, '未知城市')}"


# =====================================================================
# 第2步：准备 Mock LLM —— 用规则模拟 LLM 的工具调用决策
# =====================================================================
# 说明：生产中换成 ChatOpenAI 等真实模型即可。
# Mock LLM 的核心逻辑：
#   1. 看消息历史中有没有已调用的工具
#   2. 决定下一步调用哪个工具（或直接返回文本回答）
#
# LangGraph ReAct 循环：
#   HumanMessage → LLM 决策 → AIMessage(tool_calls=[...])
#   → 框架自动执行工具 → ToolMessage(结果) → 回灌给 LLM → 再次决策...
#   → AIMessage(content="最终回答") (无 tool_calls) → 结束

def _make_tool_call(name: str, args: dict) -> dict:
    """构造一个 LangChain 格式的 tool_call 字典。"""
    return {
        "name": name,
        "args": args,
        "id": f"call_{uuid.uuid4().hex[:8]}",
        "type": "tool_call",
    }


class FakeChatModel(BaseChatModel):
    """可 debug 的 Mock LLM —— 按规则决定工具调用序列。

    调用序列：
      第1轮：调用 weather(city="北京")
      第2轮：调用 calculator(expression="100+200")
      第3轮：生成最终文本回答（无 tool_calls → ReAct 循环结束）
    """

    model: str = "fake-for-debug"

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # ★ 断点：这是 LLM 的核心决策点
        #   - messages 包含完整对话历史
        #   - 已有的 ToolMessage 说明哪些工具已执行过

        # 收集已调用过的工具名
        called = set()
        for msg in messages:
            if isinstance(msg, AIMessage):
                for tc in (msg.tool_calls or []):
                    called.add(tc["name"])

        # ★ 断点：在这里检查 called 集合，理解 ReAct 循环进度
        print(f"  [FakeLLM] 已调用过的工具: {called or '(无)'}")

        # 决策逻辑：按固定顺序调用工具
        if "weather" not in called:
            ai = AIMessage(
                content="让我查一下北京的天气。",
                tool_calls=[_make_tool_call("weather", {"city": "北京"})],
            )
        elif "calculator" not in called:
            ai = AIMessage(
                content="再帮你算一下。",
                tool_calls=[_make_tool_call("calculator", {"expression": "100+200"})],
            )
        else:
            # 所有工具调用完毕 → 输出最终回答（无 tool_calls → ReAct 结束）
            ai = AIMessage(
                content="北京今天天气晴朗，100+200=300。任务完成！"
            )

        print(f"  [FakeLLM] 本轮输出: {ai.content[:50]}...")
        return ChatResult(generations=[ChatGeneration(message=ai)])


# =====================================================================
# 第3步：自定义中间件 —— 理解洋葱模型
# =====================================================================
# 说明：中间件在每次 agent 推理前后执行
#   before_agent: 正序执行（A → B → C）
#   after_agent:  逆序执行（C → B → A）
#
# 内置中间件链（见 AgentFeatures.resolve_middleware）：
#   DanglingToolCallMiddleware → ToolErrorHandlingMiddleware
#   → LoopDetectionMiddleware → TokenUsageMiddleware

class LoggingMiddleware(Middleware):
    """自定义中间件：记录每次 agent 推理的输入/输出。"""

    name = "my_logging"

    def __init__(self) -> None:
        self.call_count = 0

    async def before_agent(self, state: dict) -> dict | None:
        # ★ 断点：在这里打断点，可以看到每次推理前的完整 state
        self.call_count += 1
        msg_count = len(state.get("messages", []))
        print(f"  [中间件·before] 第{self.call_count}次推理，当前消息数: {msg_count}")
        return None  # 返回 None = 不修改 state

    async def after_agent(self, state: dict) -> dict | None:
        # ★ 断点：在这里打断点，可以看到推理后的 state
        msg_count = len(state.get("messages", []))
        print(f"  [中间件·after ] 推理完毕，消息数: {msg_count}")
        return None


# =====================================================================
# 第4步：自定义验证器 —— 用于 GoalLoop 的目标达成判定
# =====================================================================
# 说明：GoalLoop 每轮迭代结束后调用验证器
#   passed=True  → 目标达成，循环结束
#   passed=False → 注入反馈消息，继续下一轮

class KeywordVerifier:
    """检查 state["messages"] 中是否包含指定关键词，有则判定目标达成。"""

    def __init__(self, keyword: str):
        self.keyword = keyword
        self.call_count = 0

    async def verify(self, goal: str, state: dict) -> VerificationResult:
        # ★ 断点：在这里打断点，观察验证器如何判定目标
        self.call_count += 1
        messages = state.get("messages", [])
        all_text = " ".join(str(getattr(m, "content", "")) for m in messages)

        found = self.keyword in all_text
        print(f"  [验证器] 第{self.call_count}次验证，查找 '{self.keyword}': "
              f"{'✓ 找到' if found else '✗ 未找到'}")

        return VerificationResult(
            passed=found,
            evidence=f"关键词 '{self.keyword}' {'已' if found else '未'}出现在对话中",
        )


# =====================================================================
# 第5步：自定义循环钩子 —— 理解循环引擎的生命周期
# =====================================================================
# 说明：钩子在循环的各个阶段被调用：
#   on_iteration_start → [agent 推理] → on_iteration_end
#   → [验证] → on_goal_achieved / 继续循环
#   → on_budget_exhausted (预算耗尽时)

class DebugHook(LoopHook):
    """打印循环引擎的每个生命周期事件。"""

    name = "debug_hook"

    async def on_iteration_start(self, iteration: int, state: dict) -> HookResponse:
        # ★ 断点
        print(f"\n{'='*60}")
        print(f"  [钩子] >>>  第 {iteration + 1} 轮迭代开始")
        print(f"{'='*60}")
        return HookResponse()  # CONTINUE

    async def on_iteration_end(
        self, iteration: int, state: dict, agent_output: dict | None
    ) -> HookResponse:
        # ★ 断点
        print(f"  [钩子] <<<  第 {iteration + 1} 轮迭代结束")
        return HookResponse()

    async def on_goal_achieved(self, state: dict, evidence: str) -> None:
        print(f"\n  [钩子] 🎯 目标达成！证据: {evidence}")

    async def on_budget_exhausted(self, state: dict, reason: str) -> None:
        print(f"\n  [钩子] ⏰ 预算耗尽: {reason}")

    async def on_error(self, state: dict, error: Exception) -> HookResponse:
        print(f"  [钩子] ❌ 出错: {error}")
        return HookResponse(signal=LoopSignal.BREAK, message=str(error))


# =====================================================================
# 测试用例
# =====================================================================

class TestE2E_1_BareAgent(unittest.TestCase):
    """场景1：裸 Agent（不包循环） —— 理解 create_agent + 中间件。

    执行流程：
      create_agent(model, tools, middleware)
        → 返回 LangGraph CompiledGraph
        → agent.ainvoke(state) 启动 ReAct 循环
        → 循环: LLM决策 → 工具执行 → 结果回灌 → LLM再决策...
        → LLM 不再调用工具时循环结束

    重点观察：
      1. FakeChatModel._generate() 被调用了几次
      2. LoggingMiddleware 的 before/after 何时触发
      3. messages 列表如何增长
    """

    def test_bare_agent(self):
        print("\n" + "="*70)
        print("  场景1：裸 Agent（无循环引擎）")
        print("="*70)

        # ── 1. 创建 Agent ──
        # ★ 断点：进入 create_agent，观察中间件链的组装过程
        #   → factory.py:create_agent()
        #   → AgentFeatures.resolve_middleware()
        #   → assemble_middleware_chain()
        #   → create_react_agent()  (LangGraph)
        #   → agent._uniagent_middleware = chain  (猴子补丁挂载)
        my_logging = LoggingMiddleware()
        agent = create_agent(
            model=FakeChatModel(),
            tools=[calculator, weather],
            middleware=[my_logging],  # 只用自定义中间件，关闭内置的
            system_prompt="你是一个helpful助手。",
            name="demo_bare_agent",
        )

        # ── 2. 构建初始 state ──
        state = {
            "messages": [HumanMessage(content="帮我查北京天气，再算100+200")],
        }

        # ── 3. 运行 Agent ──
        # ★ 断点：进入 agent.ainvoke()
        #   → LangGraph 内部 ReAct 循环
        #   → 每轮循环会触发 _invoke_agent → 中间件 before/after
        result = asyncio.run(agent.ainvoke(state))

        # ── 4. 检查结果 ──
        # ★ 断点：观察 result["messages"] 的完整消息序列
        messages = result["messages"]
        print(f"\n  [结果] 共 {len(messages)} 条消息:")
        for i, msg in enumerate(messages):
            role = type(msg).__name__
            content = str(getattr(msg, "content", ""))[:60]
            tool_calls = getattr(msg, "tool_calls", None)
            extra = f" tool_calls={[tc['name'] for tc in tool_calls]}" if tool_calls else ""
            print(f"    [{i}] {role}: {content}{extra}")

        # 断言：最终消息应该包含结果
        last_msg = messages[-1]
        self.assertIsInstance(last_msg, AIMessage)
        self.assertIn("300", last_msg.content)
        self.assertGreaterEqual(my_logging.call_count, 1)
        print(f"\n  [断言通过] 中间件被调用了 {my_logging.call_count} 次")


class TestE2E_2_TurnLoop(unittest.TestCase):
    """场景2：TurnLoop（固定轮次循环） —— 理解循环引擎基础。

    执行流程：
      create_agent(model, tools, budget=...)
        → 返回 TurnLoop 包装器
        → loop.run() 启动迭代循环
        → 每轮: 预算检查 → on_iteration_start → _invoke_agent → on_iteration_end
        → 预算耗尽或钩子 BREAK 时停止

    重点观察：
      1. Budget 如何控制迭代次数
      2. LoopHook 的生命周期事件顺序
      3. TurnLoop.run() 的返回值 LoopResult
    """

    def test_turn_loop(self):
        print("\n" + "="*70)
        print("  场景2：TurnLoop（固定轮次循环）")
        print("="*70)

        # ── 1. 创建带 TurnLoop 的 Agent ──
        # ★ 断点：观察 create_agent 如何根据参数决定包装 TurnLoop
        #   → budget 有值且无 goal → 返回 TurnLoop
        loop = create_agent(
            model=FakeChatModel(),
            tools=[calculator, weather],
            features=AgentFeatures(
                dangling_tool_call=True,
                tool_error_handling=True,
                loop_detection=False,  # 关闭循环检测，避免干扰演示
                token_usage=True,
                skill=False,
            ),
            budget=Budget(config=BudgetConfig(
                max_iterations=5,      # 最多5轮
                max_time_seconds=10,   # 最多10秒
            )),
            loop_hooks=[DebugHook()],  # 加入调试钩子
            name="demo_turn_loop",
        )

        # ── 2. 验证返回的是 TurnLoop ──
        self.assertIsInstance(loop, TurnLoop)
        print(f"  [类型] 返回了 {type(loop).__name__}")

        # ── 3. 运行循环 ──
        # ★ 断点：进入 TurnLoop.run()
        #   → runtime/loop.py:TurnLoop.run()
        #   → 循环: budget.check() → _run_hooks("on_iteration_start")
        #       → _invoke_agent() → _run_hooks("on_iteration_end")
        result: LoopResult = asyncio.run(loop.run(
            input_messages=[{"role": "user", "content": "查天气算数字"}],
            thread_id="debug-session-001",
        ))

        # ── 4. 检查 LoopResult ──
        # ★ 断点：观察 LoopResult 的各个字段
        print(f"\n  [LoopResult]")
        print(f"    success    = {result.success}")
        print(f"    iterations = {result.iterations}")
        print(f"    reason     = {result.reason}")

        messages = result.final_state.get("messages", [])
        print(f"    消息总数   = {len(messages)}")

        # TurnLoop 会一直跑到 max_iterations 或 agent 内部 ReAct 结束
        self.assertGreater(result.iterations, 0)


class TestE2E_3_GoalLoop(unittest.TestCase):
    """场景3：GoalLoop（目标驱动循环） —— 理解验证驱动的自主执行。

    执行流程：
      create_agent(model, tools, goal=..., verifier=..., budget=...)
        → 返回 GoalLoop 包装器
        → loop.run() 启动目标驱动循环
        → 每轮: 预算检查 → on_iteration_start → 注入目标 → _invoke_agent
          → on_iteration_end → 验证器 verify()
          → passed=True → on_goal_achieved → 返回成功
          → passed=False → 注入反馈 → 继续循环

    重点观察：
      1. 目标如何作为 SystemMessage 注入
      2. 验证器如何判定目标达成
      3. 验证失败时反馈消息如何注入
      4. Budget 与验证器的交互
    """

    def test_goal_loop_success(self):
        """目标达成场景：LLM 调用完所有工具后，验证器检测到关键词 → 成功。"""
        print("\n" + "="*70)
        print("  场景3a：GoalLoop 目标达成")
        print("="*70)

        # ── 1. 创建 GoalLoop Agent ──
        # ★ 断点：观察 create_agent 如何组装 GoalLoop
        #   → goal + verifier → 返回 GoalLoop
        verifier = KeywordVerifier(keyword="300")  # 当 "300" 出现时目标达成

        loop = create_agent(
            model=FakeChatModel(),
            tools=[calculator, weather],
            features=AgentFeatures(
                loop_detection=False,
                skill=False,
            ),
            goal="查询北京天气并计算100+200",
            verifier=verifier,
            budget=Budget(config=BudgetConfig(
                max_iterations=5,
                max_time_seconds=10,
            )),
            loop_hooks=[DebugHook()],
            name="demo_goal_loop",
        )

        self.assertIsInstance(loop, GoalLoop)

        # ── 2. 运行目标驱动循环 ──
        # ★ 断点：进入 GoalLoop.run()
        #   → runtime/loop.py:GoalLoop.run()
        #   → 注意 self._inject_goal 如何添加 SystemMessage
        #   → 注意 self._verifier.verify() 的调用时机
        result: LoopResult = asyncio.run(loop.run(
            input_messages=[{"role": "user", "content": "请完成任务"}],
            thread_id="goal-debug-001",
        ))

        # ── 3. 验证结果 ──
        print(f"\n  [GoalLoop 结果]")
        print(f"    success    = {result.success}")
        print(f"    iterations = {result.iterations}")
        print(f"    reason     = {result.reason}")
        print(f"    evidence   = {result.evidence}")

        self.assertTrue(result.success)
        self.assertIn("300", result.evidence)
        print("  [断言通过] 目标在有限轮次内达成")

    def test_goal_loop_budget_exhausted(self):
        """预算耗尽场景：验证器永远不通过 → 预算耗尽退出。"""
        print("\n" + "="*70)
        print("  场景3b：GoalLoop 预算耗尽")
        print("="*70)

        # 关键词设为不可能出现的值 → 验证永远失败
        verifier = KeywordVerifier(keyword="不可能出现的关键词XYZ")

        loop = create_agent(
            model=FakeChatModel(),
            tools=[calculator, weather],
            features=AgentFeatures(loop_detection=False, skill=False),
            goal="完成一个不可能的任务",
            verifier=verifier,
            budget=Budget(config=BudgetConfig(
                max_iterations=2,   # 只给2轮 → 必然耗尽
                max_time_seconds=10,
            )),
            loop_hooks=[DebugHook()],
            name="demo_exhausted",
        )

        result: LoopResult = asyncio.run(loop.run(
            input_messages=[{"role": "user", "content": "请完成任务"}],
            thread_id="goal-debug-002",
        ))

        print(f"\n  [GoalLoop 结果]")
        print(f"    success    = {result.success}")
        print(f"    iterations = {result.iterations}")
        print(f"    reason     = {result.reason}")

        self.assertFalse(result.success)
        self.assertEqual(result.iterations, 2)
        print("  [断言通过] 预算耗尽后正确退出")


class TestE2E_4_MiddlewareChain(unittest.TestCase):
    """场景4：中间件链详解 —— 理解洋葱模型的执行顺序。

    执行顺序（洋葱模型）：
      → MiddlewareA.before_agent()     # 正序
      → MiddlewareB.before_agent()
        → [agent 推理]
      → MiddlewareB.after_agent()      # 逆序
      → MiddlewareA.after_agent()

    重点观察：
      1. before_agent 返回非 None 时 state 如何被修改
      2. after_agent 返回非 None 时 result 如何被修改
      3. 中间件的 state_patch 机制
    """

    def test_middleware_order(self):
        print("\n" + "="*70)
        print("  场景4：中间件链执行顺序")
        print("="*70)

        execution_log: list[str] = []

        class MwA(Middleware):
            name = "mw_a"
            async def before_agent(self, state):
                # ★ 断点
                execution_log.append("A.before")
                print(f"    [MwA] before_agent")
                return None
            async def after_agent(self, state):
                execution_log.append("A.after")
                print(f"    [MwA] after_agent")
                return None

        class MwB(Middleware):
            name = "mw_b"
            async def before_agent(self, state):
                execution_log.append("B.before")
                print(f"    [MwB] before_agent")
                return None
            async def after_agent(self, state):
                execution_log.append("B.after")
                print(f"    [MwB] after_agent")
                return None

        # 只调用1个工具的简单 LLM
        class OneShotLLM(BaseChatModel):
            model: str = "one-shot"
            @property
            def _llm_type(self) -> str:
                return "one-shot"
            def _generate(self, messages, **kwargs):
                called = {tc["name"] for m in messages
                          if isinstance(m, AIMessage) for tc in (m.tool_calls or [])}
                if "calculator" not in called:
                    ai = AIMessage(content="算一下",
                                   tool_calls=[_make_tool_call("calculator",
                                                               {"expression": "1+1"})])
                else:
                    ai = AIMessage(content="结果是2")
                return ChatResult(generations=[ChatGeneration(message=ai)])

        agent = create_agent(
            model=OneShotLLM(),
            tools=[calculator],
            middleware=[MwA(), MwB()],  # A在前，B在后
            name="demo_mw_order",
        )

        result = asyncio.run(agent.ainvoke({
            "messages": [HumanMessage(content="算1+1")],
        }))

        print(f"\n  [执行日志] {execution_log}")
        # before 正序: A.before → B.before
        # after 逆序: B.after → A.after
        # LangGraph 内部会多次调用 agent node，所以可能有多组
        # 但每组内顺序一定是 A.before → B.before → B.after → A.after

        # 验证顺序正确
        # 找第一组完整的 before-after 周期
        first_a_before = execution_log.index("A.before")
        first_b_before = execution_log.index("B.before")
        first_b_after = execution_log.index("B.after")
        first_a_after = execution_log.index("A.after")

        self.assertLess(first_a_before, first_b_before, "A.before 应在 B.before 之前")
        self.assertLess(first_b_before, first_b_after, "B.before 应在 B.after 之前")
        self.assertLess(first_b_after, first_a_after, "B.after 应在 A.after 之前")
        print("  [断言通过] 洋葱模型顺序正确：A.before → B.before → B.after → A.after")


class TestE2E_5_FullPipeline(unittest.TestCase):
    """场景5：完整管线 —— 内置中间件 + GoalLoop + 验证器 + 钩子 + 预算。

    这是最完整的端到端场景，串联所有组件。

    架构图：
      Budget(max_iterations=3)
        ↓
      GoalLoop
        ├── DebugHook (生命周期日志)
        ├── 内置钩子: ProgressLogHook, TokenBudgetHook
        └── 每轮迭代:
            ├── budget.check()
            ├── on_iteration_start
            ├── _invoke_agent:
            │   ├── DanglingToolCallMiddleware.before_agent
            │   ├── ToolErrorHandlingMiddleware.before_agent
            │   ├── TokenUsageMiddleware.before_agent
            │   ├── [LLM 推理 + 工具调用]
            │   ├── TokenUsageMiddleware.after_agent    (逆序)
            │   ├── ToolErrorHandlingMiddleware.after_agent
            │   └── DanglingToolCallMiddleware.after_agent
            ├── on_iteration_end
            └── verifier.verify()
                ├── passed → on_goal_achieved → return success
                └── failed → 注入反馈 → 继续
    """

    def test_full_pipeline(self):
        print("\n" + "="*70)
        print("  场景5：完整管线（全部组件串联）")
        print("="*70)

        verifier = KeywordVerifier(keyword="300")

        # ★ 断点：观察 create_agent 如何组装完整管线
        loop = create_agent(
            model=FakeChatModel(),
            tools=[calculator, weather],
            features=AgentFeatures(
                dangling_tool_call=True,     # 修补悬空工具调用
                tool_error_handling=True,    # 捕获工具异常
                loop_detection=True,         # 检测重复调用
                token_usage=True,            # 统计 token 用量
                skill=False,                 # 不启用技能系统
            ),
            goal="查询天气并完成计算",
            verifier=verifier,
            budget=Budget(config=BudgetConfig(
                max_iterations=3,
                max_time_seconds=30,
            )),
            loop_hooks=[DebugHook()],
            system_prompt="你是一个全能助手，请完成用户交代的任务。",
            name="demo_full_pipeline",
        )

        self.assertIsInstance(loop, GoalLoop)

        # ★ 断点：进入完整管线执行
        result = asyncio.run(loop.run(
            input_messages=[{"role": "user", "content": "查北京天气并算100+200"}],
            thread_id="full-pipeline-001",
        ))

        print(f"\n  [完整管线结果]")
        print(f"    success    = {result.success}")
        print(f"    iterations = {result.iterations}")
        print(f"    reason     = {result.reason}")
        print(f"    evidence   = {result.evidence}")

        # 检查 token 用量（由 TokenUsageMiddleware 统计）
        token_usage = result.final_state.get("token_usage", {})
        print(f"    token用量  = {token_usage}")

        self.assertTrue(result.success)
        print("  [断言通过] 完整管线端到端成功")


# =====================================================================
# 主入口：直接运行看效果，或用 IDE debug
# =====================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  uniagent 框架端到端调试测试                                      ║
║                                                                  ║
║  提示：在任何 "★ 断点" 注释处打断点，用 IDE 的 Debug 模式运行       ║
║  推荐断点位置：                                                    ║
║    1. FakeChatModel._generate()    — 观察 LLM 决策过程             ║
║    2. LoggingMiddleware.before/after — 观察中间件洋葱模型           ║
║    3. KeywordVerifier.verify()     — 观察验证器判定                ║
║    4. TurnLoop.run() / GoalLoop.run() — 观察循环引擎               ║
╚══════════════════════════════════════════════════════════════════╝
""")
    unittest.main(verbosity=2)
