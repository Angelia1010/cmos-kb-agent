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
import logging
import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, List, Optional

# 自动将 src/ 加入模块搜索路径，无需手动设置 PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

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
from uniagent.agents.config_factory import (
    get_skill_registry,
    register_skill_directory,
    reset_skill_registry,
)
from uniagent.middleware.base import Middleware
from uniagent.runtime.hooks import LoopHook
from uniagent.runtime.signals import HookResponse, LoopSignal
from uniagent.skills import SkillManifest


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

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        names = [getattr(t, "name", None) or getattr(t, "__name__", "") for t in tools]
        return self.bind(bound_tool_names=names, **kwargs)

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


class SkillDemoLLM(BaseChatModel):
    """用于 Skill 系统演示的脚本化 Mock LLM。

    按固定顺序调用技能工具：
      第1轮：load_skill_reference（加载参考文档）
      第2轮：validate_taocan_price（验证价格合规性）
      第3轮：生成最终回答
    """

    model: str = "skill-demo"

    def bind_tools(self, tools_list: Any, **kwargs: Any) -> Any:
        names = [getattr(t, "name", None) or getattr(t, "__name__", "") for t in tools_list]
        return self.bind(bound_tool_names=names, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "skill-demo"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        called = {
            tc["name"]
            for msg in messages
            if isinstance(msg, AIMessage)
            for tc in (msg.tool_calls or [])
        }
        if "load_skill_reference" not in called:
            ai = AIMessage(
                content="需要更详细的规范，让我加载参考文档。",
                tool_calls=[_make_tool_call("load_skill_reference", {
                    "skill_name": "taocan-skill",
                    "filename": "field_rules.md",
                })],
            )
        elif "validate_taocan_price" not in called:
            ai = AIMessage(
                content="参考文档已加载，验证价格合规性。",
                tool_calls=[_make_tool_call("validate_taocan_price", {
                    "price": "99元/月",
                })],
            )
        else:
            ai = AIMessage(content="套餐年费=99×12=1188元(每年)。价格格式已验证合规。")
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

def demo_bare_agent():
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

        print(f"\n  [结论] 中间件被调用了 {my_logging.call_count} 次")


def demo_turn_loop():
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
                goal_loop=True,  # 启用循环引擎，才会返回 TurnLoop
            ),
            budget=Budget(config=BudgetConfig(
                max_iterations=5,      # 最多5轮
                max_time_seconds=10,   # 最多10秒
            )),
            loop_hooks=[DebugHook()],  # 加入调试钩子
            name="demo_turn_loop",
        )

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



def demo_goal_loop_success():
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


def demo_goal_loop_budget_exhausted():
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



def demo_middleware_order():
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
            def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
                names = [getattr(t, "name", None) or getattr(t, "__name__", "") for t in tools]
                return self.bind(bound_tool_names=names, **kwargs)
            @property
            def _llm_type(self) -> str:
                return "one-shot"
            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                called = {tc["name"] for m in messages
                          if isinstance(m, AIMessage) for tc in (m.tool_calls or [])}
                if "calculator" not in called:
                    ai = AIMessage(content="算一下",
                                   tool_calls=[_make_tool_call("calculator",
                                                               {"expression": "1+1"})])
                else:
                    ai = AIMessage(content="结果是2")
                return ChatResult(generations=[ChatGeneration(message=ai)])

        # 用 GoalLoop 包装，中间件才会在循环引擎中被调用
        verifier = KeywordVerifier(keyword="2")
        loop = create_agent(
            model=OneShotLLM(),
            tools=[calculator],
            middleware=[MwA(), MwB()],  # A在前，B在后
            goal="计算1+1",
            verifier=verifier,
            budget=Budget(config=BudgetConfig(max_iterations=3, max_time_seconds=10)),
            name="demo_mw_order",
        )

        result: LoopResult = asyncio.run(loop.run(
            input_messages=[{"role": "user", "content": "算1+1"}],
            thread_id="mw-order-001",
        ))

        print(f"\n  [执行日志] {execution_log}")
        # before 正序: A.before → B.before
        # after 逆序: B.after → A.after
        if execution_log:
            print(f"  [结论] 洋葱模型顺序：{'→'.join(execution_log)}")
        else:
            print("  [结论] 中间件未被触发（预期外）")


def demo_full_pipeline():
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

        print(f"  [结论] 完整管线执行{'成功' if result.success else '失败'}")


# =====================================================================
# 第6步：Skill 技能系统 —— 理解触发器匹配 + 技能注入
# =====================================================================
# 说明：Skill 系统的完整链路：
#   1. 技能包（metadata.json + SKILL.md）定义触发条件和指令
#   2. SkillRegistry 扫描目录、注册技能、匹配用户输入
#   3. SkillMiddleware 在 before_agent 中自动匹配并注入 SystemMessage
#   4. LLM 在推理时能看到注入的技能指令
#
# 本演示使用项目 skills/taocan-skill/ 目录，通过框架完整链路：
#   create_agent(skill=True) → factory 预加载工具
#   → GoalLoop._invoke_agent() → SkillMiddleware.before_agent() 自动注入
#   → LLM 推理 + 工具调用 → 验证器判定
#
# 注册表管理函数（均来自 uniagent.agents.config_factory）：
#   register_skill_directory() — 渐进式扫描目录（幂等）
#   get_skill_registry()       — 获取全局注册表单例
#   reset_skill_registry()     — 重置注册表（测试清理用）

def demo_skill_system():
        print("\n" + "="*70)
        print("  场景6：Skill 技能系统（渐进式加载 + 触发匹配 + 技能注入 + 脚本工具）")
        print("="*70)

        project_dir = Path(__file__).parent
        skills_root = project_dir / "skills"

        try:
            # ═══════════════════════════════════════════════════════════
            # 阶段A：渐进式加载演示
            # ═══════════════════════════════════════════════════════════
            print(f"\n  ── 阶段A：渐进式加载演示 ──")

            # ── 步骤1：重置为空注册表 ──
            print(f"\n  [步骤1] 重置注册表（reset_skill_registry）")
            reset_skill_registry()
            assert get_skill_registry() is None
            print(f"    注册表状态: None  ✓ 已重置")

            # ── 步骤2：渐进式首次扫描 ──
            # register_skill_directory() 内部调用 registry.scan()，首次创建并扫描
            print(f"\n  [步骤2] 渐进式首次扫描（register_skill_directory）")
            count1 = register_skill_directory(str(skills_root))
            registry = get_skill_registry()
            print(f"    首次扫描新增: {count1} 个技能")
            print(f"    已扫描目录数: {len(registry.scanned_directories)}")
            for info in registry.list_skills():
                print(f"    - {info['name']}: {info['description']} "
                      f"(触发器={info['triggers']}, 标签={info['tags']})")
            assert count1 >= 1, f"应扫描到至少1个技能，实际: {count1}"
            assert "taocan-skill" in registry.skills

            # ── 步骤3：渐进式幂等性 —— 重复调用同目录应被跳过 ──
            print(f"\n  [步骤3] 重复注册同目录（渐进式幂等性验证）")
            count2 = register_skill_directory(str(skills_root))
            print(f"    第二次调用新增: {count2}  ✓ 已跳过（幂等）")
            print(f"    技能总数不变: {len(get_skill_registry().skills)}")
            assert count2 == 0, f"重复注册应返回 0，实际: {count2}"

            # ── 步骤4：force=True —— 强制刷新（热更新场景）──
            # 通过 get_skill_registry() 获取注册表后调用 scan(force=True)
            print(f"\n  [步骤4] force=True 强制刷新（模拟热更新场景）")
            count3 = get_skill_registry().scan(str(skills_root), force=True)
            print(f"    force 刷新重处理: {count3} 个技能（覆盖更新已有技能）")
            print(f"    技能总数不变: {len(get_skill_registry().skills)}（仅更新元数据）")

            # ── 步骤5：编程式注册 —— 动态追加无目录依赖的技能 ──
            # 适用于：运行时热插拔、从配置服务动态加载技能
            print(f"\n  [步骤5] 编程式注册内存技能（热插拔）")
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                (tmp_path / "SKILL.md").write_text(
                    "# 渐进式演示技能\n提供标准化演示场景处理流程。",
                    encoding="utf-8",
                )
                demo_manifest = SkillManifest.from_dict({
                    "name": "demo-progressive",
                    "description": "渐进式热插拔演示",
                    "triggers": [{"type": "keyword", "value": "演示场景"}],
                    "tags": ["demo"],
                })
                get_skill_registry().register(demo_manifest, tmp_path)
                total = len(get_skill_registry().skills)
                print(f"    编程式注册完成，注册表共 {total} 个技能")
                for info in get_skill_registry().list_skills():
                    print(f"    - {info['name']}: {info['description']}")
                assert "demo-progressive" in get_skill_registry().skills

                # ── 步骤6：热重载单个技能 ──
                # reload_skill() 重新解析 metadata.json，无需重扫整个目录
                print(f"\n  [步骤6] 热重载单个技能（reload_skill）")
                ok = get_skill_registry().reload_skill("taocan-skill")
                print(f"    taocan-skill 热重载: {'✓ 成功' if ok else '✗ 失败'}")
                assert ok, "taocan-skill 热重载应成功"

            # ── 步骤7：注销演示技能，还原至只有 taocan-skill ──
            print(f"\n  [步骤7] 注销演示技能")
            removed = get_skill_registry().unregister("demo-progressive")
            print(f"    注销: {'✓ 成功' if removed else '✗ 失败'}")
            print(f"    剩余技能: {list(get_skill_registry().skills.keys())}")
            assert "demo-progressive" not in get_skill_registry().skills

            # ═══════════════════════════════════════════════════════════
            # 阶段B：完整管线演示（技能注入 + 参考文档按需加载 + 脚本工具）
            # ═══════════════════════════════════════════════════════════
            # 框架全自动完成全部技能链路：
            #   SkillMiddleware:  before_agent() 匹配触发器、注入 SKILL.md
            #   factory.py:       feat.skill=True 时预加载 load_skill_reference + 脚本工具
            #   LLM ReAct 循环:   SkillDemoLLM 按序调用工具；验证器检查 "1188"
            print(f"\n  ── 阶段B：完整管线演示（技能注入 + 参考文档 + 脚本工具）──")

            print(f"\n  [步骤8] create_agent + SkillMiddleware + GoalLoop 全管线")
            loop = create_agent(
                model=SkillDemoLLM(),
                tools=[calculator],
                features=AgentFeatures(
                    skill=True,
                    dangling_tool_call=True,
                    tool_error_handling=True,
                    loop_detection=False,
                    token_usage=False,
                ),
                goal="查询99元套餐年费并验证价格合规性",
                verifier=KeywordVerifier(keyword="1188"),
                budget=Budget(config=BudgetConfig(
                    max_iterations=5,
                    max_time_seconds=30,
                )),
                name="demo_skill_agent",
            )

            print(f"    Agent 类型: {type(loop).__name__}（GoalLoop 包装）")

            result = asyncio.run(loop.run(
                input_messages=[HumanMessage(content="帮我查一下99元套餐的年费")],
                thread_id="skill-demo-001",
            ))

            # 从最终状态消息确认各环节执行情况
            # SkillMiddleware 将技能注入写入 state，消息流中可直接看到 <!-- SKILL: --> 标记
            messages = result.final_state.get("messages", [])
            print(f"\n  [消息流] 共 {len(messages)} 条:")
            for i, msg in enumerate(messages):
                role = type(msg).__name__
                c = str(getattr(msg, "content", ""))
                is_skill = "<!-- SKILL:" in c
                is_goal = "[目标]" in c
                tag = " [技能注入]" if is_skill else (" [目标注入]" if is_goal else "")
                display = c[:80] + ("..." if len(c) > 80 else "")
                print(f"    [{i}] {role}: {display}{tag}")

            tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
            skill_injected = any("<!-- SKILL:" in str(getattr(m, "content", "")) for m in messages)
            ref_loaded = any("字段归一规则" in str(getattr(m, "content", "")) for m in tool_msgs)
            script_called = any("合规" in str(getattr(m, "content", "")) for m in tool_msgs)

            print(f"\n  [验证结果]")
            print(f"    GoalLoop 成功:   {'通过' if result.success else '失败'}"
                  f" (迭代={result.iterations}, reason={result.reason})")
            print(f"    技能注入到 state: {'通过' if skill_injected else '失败'}"
                  f" (SkillMiddleware 在 before_agent 写入 state['messages'])")
            print(f"    参考文档加载:    {'通过' if ref_loaded else '失败'}"
                  f" (load_skill_reference 按需返回 field_rules.md)")
            print(f"    脚本工具调用:    {'通过' if script_called else '失败'}"
                  f" (validate_taocan_price 返回合规结果)")

        finally:
            reset_skill_registry()

        print(f"\n  [结论] 渐进式 Skill 系统完整演示完毕")


# =====================================================================
# 主入口：直接运行看效果，或用 IDE debug
# =====================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  uniagent 框架端到端演示                                          ║
║                                                                  ║
║  提示：在任何 "★ 断点" 注释处打断点，用 IDE 的 Debug 模式运行       ║
║  推荐断点位置：                                                    ║
║    1. FakeChatModel._generate()    — 观察 LLM 决策过程             ║
║    2. LoggingMiddleware.before/after — 观察中间件洋葱模型           ║
║    3. KeywordVerifier.verify()     — 观察验证器判定                ║
║    4. TurnLoop.run() / GoalLoop.run() — 观察循环引擎               ║
╚══════════════════════════════════════════════════════════════════╝
""")

    demos = [
        ("场景1：裸 Agent（无循环引擎）", demo_bare_agent),
        ("场景2：TurnLoop（固定轮次循环）", demo_turn_loop),
        ("场景3a：GoalLoop 目标达成", demo_goal_loop_success),
        ("场景3b：GoalLoop 预算耗尽", demo_goal_loop_budget_exhausted),
        ("场景4：中间件链执行顺序", demo_middleware_order),
        ("场景5：完整管线（全部组件串联）", demo_full_pipeline),
        ("场景6：Skill 技能系统", demo_skill_system),
    ]

    for name, fn in demos:
        try:
            fn()
            print(f"\n  >>> {name} 执行完毕 <<<\n")
        except Exception as e:
            print(f"\n  >>> {name} 出错: {e} <<<\n")
            import traceback
            traceback.print_exc()
