"""LLM 调用日志中间件 — 三层全覆盖的可复用日志组件。

覆盖三个层次：
- **LLM调用级**（内置 LangChain Callback）：完整 prompt / response / 工具调用
- **state 级**（before_agent / after_agent）：每轮推理前后的状态快照
- **循环层**（loop_hooks）：迭代开始/结束/目标达成/预算耗尽

适用模式：
- TurnLoop / GoalLoop：通过 `get_invoke_config` 自动注入 callback，三层全部生效。
- 裸 Agent（CompiledGraph）：中间件钩子不触发，但可调用
  ``mw.as_langchain_callback()`` 手动传入 ainvoke config。

典型用法::

    from uniagent.middleware.builtins import LLMLoggingMiddleware

    # TurnLoop / GoalLoop —— 自动注入，无需额外操作
    loop = create_agent(
        model=model,
        tools=[...],
        middleware=[LLMLoggingMiddleware(verbose=True)],
        budget=Budget(...),
    )

    # 裸 Agent —— 手动传入 callback
    logging_mw = LLMLoggingMiddleware(verbose=True)
    result = await agent.ainvoke(
        state,
        config={"callbacks": [logging_mw.as_langchain_callback()]},
    )
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage

from uniagent.middleware.base import Middleware
from uniagent.runtime.hooks import LoopHook
from uniagent.runtime.signals import HookResponse
from uniagent.state.thread_state import ThreadState

_DEFAULT_LOGGER = logging.getLogger("uniagent.llm_log")


# ---------------------------------------------------------------------------
# 内部 LangChain Callback —— 捕获 LLM 调用级的完整信息
# ---------------------------------------------------------------------------

class _LLMVerboseCallback(BaseCallbackHandler):
    """捕获 LangChain 运行时的 LLM prompt / response / 工具事件。

    该回调由 LLMLoggingMiddleware 通过 get_invoke_config 注入，
    无需用户手动传递。
    """

    def __init__(self, log: logging.Logger, level: int, verbose: bool) -> None:
        super().__init__()
        self._log = log
        self._level = level
        self._verbose = verbose
        self._llm_round = 0

    # ── LLM 推理前：打印完整 prompt ──────────────────────────────────────

    def on_chat_model_start(
        self, serialized: dict, messages: list, **kwargs: Any
    ) -> None:
        self._llm_round += 1
        self._log.log(
            self._level,
            "[LLM#%d] ── Prompt (%d 批次) ──────────────────────────────",
            self._llm_round,
            len(messages),
        )
        if not self._verbose:
            # 非详细模式：仅打印最后一条消息的类型和简要内容
            for batch in messages:
                if batch:
                    last = batch[-1]
                    role = type(last).__name__
                    content = _fmt_content(getattr(last, "content", ""))[:120]
                    self._log.log(self._level, "  最后消息 [%s]: %s", role, content)
            return
        for batch in messages:
            for msg in batch:
                role = type(msg).__name__
                content = _fmt_content(getattr(msg, "content", ""))
                tool_calls = getattr(msg, "tool_calls", None)
                tc_info = (
                    f"  tool_calls={[tc['name'] for tc in tool_calls]}"
                    if tool_calls
                    else ""
                )
                self._log.log(
                    self._level,
                    "  [%s] %s%s",
                    role,
                    content[:300] if content else "(空)",
                    tc_info,
                )

    # ── LLM 推理后：打印响应内容 ──────────────────────────────────────────

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        self._log.log(
            self._level,
            "[LLM#%d] ── Response ─────────────────────────────────────",
            self._llm_round,
        )
        for gen_list in response.generations:
            for gen in gen_list:
                msg = getattr(gen, "message", None) or gen
                content = _fmt_content(getattr(msg, "content", ""))
                tool_calls: list = getattr(msg, "tool_calls", []) or []
                limit = 400 if self._verbose else 200
                display = content if len(content) <= limit else content[:limit] + "…(截断)"
                self._log.log(self._level, "  content: %s", display or "(空)")
                if tool_calls:
                    for tc in tool_calls:
                        self._log.log(
                            self._level,
                            "  tool_call: %s  args=%s",
                            tc["name"],
                            tc.get("args", {}),
                        )
                    self._log.log(self._level, "  → 决定调用工具，ReAct 继续")
                else:
                    self._log.log(self._level, "  → 无 tool_calls，ReAct 结束")

    # ── 工具调用前：打印工具名 + 参数 ─────────────────────────────────────

    def on_tool_start(
        self, serialized: dict, input_str: str, **kwargs: Any
    ) -> None:
        tool_name = serialized.get("name", "?")
        self._log.log(
            self._level,
            "[工具] ▶ %s  input=%s",
            tool_name,
            input_str[:200],
        )

    # ── 工具调用后：打印返回值 ────────────────────────────────────────────

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        self._log.log(
            self._level,
            "[工具] ◀ output=%s",
            str(output)[:300],
        )

    def on_tool_error(self, error: Exception, **kwargs: Any) -> None:
        self._log.warning("[工具] ✗ error=%s", error)


# ---------------------------------------------------------------------------
# 内部 LoopHook —— 循环层事件日志
# ---------------------------------------------------------------------------

class _LLMLoggingHook(LoopHook):
    """记录循环层迭代开始/结束/目标达成/预算耗尽信息。"""

    name = "llm_logging_hook"

    def __init__(self, log: logging.Logger, level: int) -> None:
        self._log = log
        self._level = level

    async def on_iteration_start(
        self, iteration: int, state: dict[str, Any]
    ) -> HookResponse:
        msgs = state.get("messages", [])
        self._log.log(
            self._level,
            "[循环] ══ 第 %d 轮迭代开始（当前消息数: %d）",
            iteration + 1,
            len(msgs),
        )
        return HookResponse()

    async def on_iteration_end(
        self,
        iteration: int,
        state: dict[str, Any],
        agent_output: dict[str, Any] | None,
    ) -> HookResponse:
        msgs = state.get("messages", [])
        last = msgs[-1] if msgs else None
        has_tc = bool(getattr(last, "tool_calls", None)) if last else False
        token_usage = state.get("token_usage", {})
        total_tokens = token_usage.get("total_tokens", "?")
        self._log.log(
            self._level,
            "[循环] ── 第 %d 轮迭代结束（消息数: %d，tool_calls: %s，累计tokens: %s）",
            iteration + 1,
            len(msgs),
            has_tc,
            total_tokens,
        )
        return HookResponse()

    async def on_goal_achieved(
        self, state: dict[str, Any], evidence: str
    ) -> None:
        self._log.log(self._level, "[循环] ✓ 目标达成！证据: %s", evidence[:200])

    async def on_budget_exhausted(
        self, state: dict[str, Any], reason: str
    ) -> None:
        self._log.warning("[循环] ✗ 预算耗尽: %s", reason)

    async def on_error(
        self, state: dict[str, Any], error: Exception
    ) -> HookResponse:
        self._log.error("[循环] ✗ 出错: %s", error)
        return HookResponse()


# ---------------------------------------------------------------------------
# 公开中间件
# ---------------------------------------------------------------------------

class LLMLoggingMiddleware(Middleware):
    """三层全覆盖的 LLM 执行日志中间件。

    参数
    ----------
    verbose : bool
        True  → 打印完整 prompt（每条消息全文）和完整 response。
        False → 仅打印最后一条消息摘要和 response 摘要（默认）。
    log_level : int
        日志级别，默认 ``logging.DEBUG``。生产环境可设为 INFO。
    logger : logging.Logger | None
        使用指定 logger；默认使用 ``uniagent.llm_log``。
    include_loop_hooks : bool
        是否通过 loop_hooks() 注册循环层钩子（默认 True）。
    """

    name = "llm_logging"

    def __init__(
        self,
        *,
        verbose: bool = False,
        log_level: int = logging.DEBUG,
        logger: logging.Logger | None = None,
        include_loop_hooks: bool = True,
    ) -> None:
        self._verbose = verbose
        self._level = log_level
        self._log = logger or _DEFAULT_LOGGER
        # 显式设置 logger 自身的级别，避免继承根 logger 的更高级别
        # （如 basicConfig(level=WARNING) 时 INFO 消息会被静默丢弃）
        if self._log.level == logging.NOTSET or self._log.level > log_level:
            self._log.setLevel(log_level)
        self._include_loop_hooks = include_loop_hooks
        self._call_count = 0
        self._callback = _LLMVerboseCallback(self._log, self._level, verbose)

    # ── state 级：before_agent / after_agent ─────────────────────────────

    async def before_agent(self, state: ThreadState) -> ThreadState | None:
        """在每轮 LLM 推理前记录 state 快照（TurnLoop/GoalLoop 模式生效）。"""
        self._call_count += 1
        msgs = state.get("messages", [])
        self._log.log(
            self._level,
            "[MW·before #%d] 推理前快照：消息数=%d",
            self._call_count,
            len(msgs),
        )
        if self._verbose and msgs:
            for i, m in enumerate(msgs):
                role = type(m).__name__
                content = str(getattr(m, "content", ""))[:80]
                tc = getattr(m, "tool_calls", None)
                tc_info = f"  tc={[x['name'] for x in tc]}" if tc else ""
                self._log.log(
                    self._level,
                    "    [%d] %s: %r%s",
                    i, role, content, tc_info,
                )
        return None

    async def after_agent(self, state: ThreadState) -> ThreadState | None:
        """在每轮 LLM 推理后记录 state 变化（TurnLoop/GoalLoop 模式生效）。"""
        msgs = state.get("messages", [])
        last = msgs[-1] if msgs else None
        if last is None:
            return None
        role = type(last).__name__
        content = str(getattr(last, "content", ""))[:120]
        tc = getattr(last, "tool_calls", None)
        has_tc = bool(tc)
        self._log.log(
            self._level,
            "[MW·after  #%d] 推理后快照：[%s] %r  tool_calls=%s",
            self._call_count,
            role,
            content,
            [x["name"] for x in tc] if tc else False,
        )
        if not has_tc:
            self._log.log(self._level, "[MW·after  #%d] → 无工具调用，ReAct 结束", self._call_count)
        return None

    # ── LLM调用级：通过 get_invoke_config 自动注入 callback ──────────────

    def get_invoke_config(self) -> dict[str, Any]:
        """向 ``_invoke_agent`` 提供 callback，自动捕获完整 LLM prompt/response。

        在 TurnLoop / GoalLoop 模式下由 ``_invoke_agent`` 自动调用，
        无需用户手动配置。
        """
        return {"callbacks": [self._callback]}

    def as_langchain_callback(self) -> _LLMVerboseCallback:
        """返回内置的 LangChain callback，供裸 Agent 模式手动注入。

        裸 Agent 不经过 ``_invoke_agent``，中间件钩子不触发，
        但可以通过此方法获取 callback 并手动传入::

            cb = logging_mw.as_langchain_callback()
            result = await agent.ainvoke(state, config={"callbacks": [cb]})
        """
        return self._callback

    # ── 循环层：loop_hooks ────────────────────────────────────────────────

    def loop_hooks(self) -> list[Any]:
        """返回循环层钩子，记录迭代开始/结束/目标达成/预算耗尽。"""
        if not self._include_loop_hooks:
            return []
        return [_LLMLoggingHook(self._log, self._level)]


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _fmt_content(content: Any) -> str:
    """统一处理 str / list（thinking 模型返回 block 列表）两种格式。"""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            parts.append(str(block))
            continue
        t = block.get("type", "")
        if t == "thinking":
            snippet = block.get("thinking", "")[:80]
            parts.append(f"[thinking] {snippet}…")
        elif t == "text":
            parts.append(block.get("text", ""))
        elif t == "tool_use":
            parts.append(f"[tool_use] {block.get('name')}")
        else:
            parts.append(str(block))
    return "\n".join(parts)
