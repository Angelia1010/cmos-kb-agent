# 循环引擎实现详解 — 裸 Agent / TurnLoop / GoalLoop

---

## 总览：三种模式的代码路径

```
create_agent()                                          ← agents/factory.py:22-137
  │
  ├─ 共同部分：构建中间件链 + create_react_agent()        ← factory.py:67-95
  │
  ├─ goal != None? ──是──→ GoalLoop(agent, goal, verifier, ...)  ← factory.py:110-125
  │                          └→ GoalLoop.run()                    ← runtime/loop.py:211-323
  │
  ├─ feat.goal_loop? ──是──→ TurnLoop(agent, hooks, budget)      ← factory.py:127-134
  │                           └→ TurnLoop.run()                   ← runtime/loop.py:115-169
  │
  └─ 都不是 ──→ return agent（裸 CompiledGraph）                  ← factory.py:137
                  └→ agent.ainvoke() / agent.astream()            ← LangGraph 原生
```

判断逻辑在 `factory.py:110-137`：

```python
if goal is not None:       # → GoalLoop
    ...
if feat.goal_loop:         # → TurnLoop
    ...
return agent               # → 裸 Agent（走到这里说明什么都没设）
```

三者共享的基础设施在 `BaseLoop`（`runtime/loop.py:31-100`）：
- `_invoke_agent()` — 调用 LangGraph Agent 做一轮推理
- `_run_hooks()` — 执行钩子并收集控制信号
- `_notify_hooks()` — 即发即忘通知
- `Budget` — 三维预算管理（`runtime/budget.py`）
- `LoopSignal` / `LoopResult` / `HookResponse` — 控制信号和结果类型（`runtime/signals.py`）

---

## 一、裸 Agent

### 实现位置

`agents/factory.py:86-95` + `agents/factory.py:136-137`

### 实现方式

裸 Agent 就是直接调用 LangGraph 的 `create_react_agent()`，不做任何循环包装：

```python
# factory.py:86-94 — 构建 LangGraph 的 ReAct Agent
agent = create_react_agent(
    model=model,          # ChatOpenAI 等 LLM 实例
    tools=list(tools),    # 工具列表
    state_schema=state_schema,  # ThreadState
    prompt=system_prompt or None,
    checkpointer=checkpointer,
    name=name,
)
agent._uniagent_middleware = chain   # 挂载中间件（但不包装循环）

# factory.py:136-137 — 没有 goal 也没有 goal_loop 特性
return agent  # 直接返回 LangGraph 的 CompiledGraph
```

### 调用方式

和原生 LangGraph 完全一样：

```python
result = await agent.ainvoke({"messages": [...]})
# 或
async for chunk in agent.astream({"messages": [...]}):
    ...
```

### 本质

裸 Agent 就是 LangGraph 的 `CompiledGraph`，uniagent 只是帮你组装了中间件挂上去，没有做任何额外包装。调用一次就结束。

---

## 二、TurnLoop

### 实现位置

`runtime/loop.py:107-169`

### 触发条件

`factory.py:127-134`：

```python
if feat.goal_loop:   # 开启了 goal_loop 特性，但没传 goal
    return TurnLoop(agent=agent, hooks=all_hooks, budget=resolved_budget)
```

### 核心执行流程

`TurnLoop.run()` 就是一个简单的 for 循环：

```
factory.py:127-134                    loop.py:115-169
┌──────────────┐                 ┌──────────────────────────────┐
│ 判断条件：    │                 │ TurnLoop.run():              │
│ goal_loop=True│  ──创建──→     │                              │
│ goal=None     │                 │ for i in range(max_iter):    │
└──────────────┘                 │   ①  budget.check()          │
                                  │       → 超限? return 失败    │
                                  │                              │
                                  │   ②  hooks.on_iteration_start│
                                  │       → BREAK? return 失败   │
                                  │                              │
                                  │   ③  agent.ainvoke(state)    │
                                  │       → 异常? 问钩子怎么办   │
                                  │                              │
                                  │   ④  budget.record_iteration │
                                  │                              │
                                  │   ⑤  hooks.on_iteration_end  │
                                  │       → BREAK? return 成功   │
                                  │                              │
                                  │ return "已达最大迭代次数"      │
                                  └──────────────────────────────┘
```

### 逐行拆解关键代码

```python
# loop.py:122-124 — 初始化状态
state = initial_state or {}
if input_messages:
    state["messages"] = input_messages

# loop.py:126-154 — 主循环
for i in range(self._budget.config.max_iterations or 100):

    # ① 预算检查（loop.py:128-133）
    signal, reason = self._budget.check()      # 检查 次数/token/时间 三维限制
    if signal == LoopSignal.BREAK:
        await self._notify_hooks("on_budget_exhausted", state, reason)
        return LoopResult(success=False, ...)   # 预算耗尽，失败退出

    # ② 迭代前钩子（loop.py:136-140）
    resp = await self._run_hooks("on_iteration_start", i, state)
    if resp.signal == LoopSignal.BREAK:         # 钩子叫停
        return LoopResult(success=False, ...)

    # ③ 调用 Agent 做一轮推理（loop.py:143-152）
    try:
        result = await self._invoke_agent(state, thread_id=thread_id)
        state.update(result)                    # 把 Agent 输出合并回 state
    except Exception as exc:
        resp = await self._run_hooks("on_error", state, exc)  # 问钩子怎么办
        if resp.signal == LoopSignal.BREAK:
            return LoopResult(success=False, reason=f"错误：{exc}", ...)

    # ④ 记录迭代（loop.py:154）
    self._budget.record_iteration()             # iterations_used += 1

    # ⑤ 迭代后钩子（loop.py:157-162）
    resp = await self._run_hooks("on_iteration_end", i, state, result)
    if resp.signal == LoopSignal.BREAK:
        return LoopResult(success=True, ...)     # 钩子说够了，成功退出
```

### TurnLoop 的特点

- 没有验证器，不检查目标是否完成
- 没有检查点/回退机制
- 只处理 BREAK 信号，不处理 RETRY/ROLLBACK

---

## 三、GoalLoop

### 实现位置

`runtime/loop.py:176-323`

### 触发条件

`factory.py:110-125`：

```python
if goal is not None:   # 传了 goal
    if verifier is None:
        verifier = AlwaysPassVerifier()  # 没传验证器就用空操作的
    return GoalLoop(agent=agent, goal=goal, verifier=verifier, ...)
```

### GoalLoop 比 TurnLoop 多了什么

GoalLoop 继承 BaseLoop，和 TurnLoop 共享基础设施，但增加了 4 个关键能力：

| 能力 | TurnLoop | GoalLoop | 实现位置 |
|------|----------|----------|---------|
| 目标注入 | 无 | 首次迭代注入 SystemMessage | `loop.py:222-232` |
| 验证器 | 无 | 每 N 轮运行验证 | `loop.py:288-316` |
| 检查点/回退 | 无 | 每轮保存，支持 ROLLBACK | `loop.py:274-275, 251-254` |
| 验证反馈 | 无 | 失败时注入 HumanMessage | `loop.py:308-316` |

### 核心执行流程

```
factory.py:110-125                     loop.py:211-323
┌──────────────┐                  ┌─────────────────────────────────────┐
│ 判断条件：    │                  │ GoalLoop.run():                     │
│ goal="写游戏" │  ──创建──→      │                                     │
│ verifier=...  │                  │ ⓪ 注入目标 SystemMessage            │
└──────────────┘                  │    "[目标] 您的任务目标：写游戏..."    │
                                   │                                     │
                                   │ for i in range(max_iter):           │
                                   │   ① budget.check()                  │
                                   │   ② hooks.on_iteration_start()      │
                                   │      → ROLLBACK? 回退到检查点        │
                                   │   ③ agent.ainvoke(state)            │
                                   │      → 异常? RETRY 可重试            │
                                   │   ④ 保存检查点 {**state}             │
                                   │   ⑤ hooks.on_iteration_end()        │
                                   │      → RETRY? 跳到下一轮             │
                                   │   ⑥ verifier.verify(goal, state)    │
                                   │      → 通过? return 成功              │
                                   │      → 失败? 注入反馈继续             │
                                   │                                     │
                                   │ return "已达最大迭代次数，目标未完成"   │
                                   └─────────────────────────────────────┘
```

### 逐块拆解 GoalLoop 独有的代码

#### ⓪ 目标注入（loop.py:222-232）

```python
if self._inject_goal:
    goal_msg = SystemMessage(
        content=(
            f"[目标] 您的任务目标：\n{self._goal}\n\n"
            "请逐步朝该目标推进。"
            "每一步结束后，请评估当前进度。"
        )
    )
    msgs = state.get("messages", [])
    state["messages"] = [goal_msg] + msgs   # 插到消息最前面
```

这让 LLM 从第一轮开始就知道自己要干什么。

#### ② ROLLBACK 支持（loop.py:251-254）

```python
resp = await self._run_hooks("on_iteration_start", i, state)
if resp.signal == LoopSignal.ROLLBACK and last_checkpoint_state:
    logger.info("正在回退至第 %d 次迭代的检查点", i)
    state = {**last_checkpoint_state}   # 恢复到上一轮的状态
    continue                            # 重新开始这轮
```

TurnLoop 的 `on_iteration_start` 只处理 BREAK，GoalLoop 额外处理 ROLLBACK。

#### ③ 异常时支持 RETRY（loop.py:261-270）

```python
except Exception as exc:
    resp = await self._run_hooks("on_error", state, exc)
    if resp.signal == LoopSignal.BREAK:
        return LoopResult(success=False, ...)
    if resp.signal == LoopSignal.RETRY:     # GoalLoop 独有
        continue                            # 跳过本轮，重试
```

TurnLoop 遇到异常只有 BREAK 一个选择，GoalLoop 可以选择重试。

#### ④ 保存检查点（loop.py:274-275）

```python
last_checkpoint_state = {**state}   # 浅拷贝当前状态作为检查点
```

每轮成功执行后都保存一份快照，这样 ROLLBACK 时有东西可以回退到。

#### ⑥ 验证 + 反馈注入（loop.py:288-316）

```python
# 每 verify_every 轮运行一次验证
if (i + 1) % self._verify_every == 0:
    vr = await self._verifier.verify(self._goal, state)

    if vr.passed:   # 目标达成！
        await self._notify_hooks("on_goal_achieved", state, vr.evidence)
        return LoopResult(
            success=True,
            iterations=i + 1,
            reason="目标验证通过",
            evidence=vr.evidence,   # 比如 "6 tests passed"
        )
    else:           # 没达成，告诉 Agent 为什么
        feedback = HumanMessage(
            content=(
                f"[验证失败] 目标尚未达成。\n"
                f"依据：{vr.evidence}\n"           # 比如 "3 tests failed"
                f"请继续朝目标推进：{self._goal}"
            )
        )
        state["messages"] = msgs + [feedback]   # 注入反馈让 Agent 看到
```

这是 GoalLoop 最关键的设计——验证失败时不是直接退出，而是把失败原因作为 HumanMessage 注入回去，让 Agent 在下一轮根据反馈调整策略。

---

## 四、共享基础设施

### BaseLoop（loop.py:31-100）

三个循环共享的抽象基类：

```python
class BaseLoop(ABC):
    def __init__(self, agent, *, hooks, budget, config):
        self._agent = agent
        self._budget = budget or Budget()
        self._hooks = list(hooks) if hooks else [ProgressLogHook()]

        # 自动添加 token 预算同步钩子
        if not any(isinstance(h, TokenBudgetHook) for h in self._hooks):
            self._hooks.append(TokenBudgetHook(self._budget))
```

#### _invoke_agent()（loop.py:61-70）

调用 LangGraph Agent 做一轮推理：

```python
async def _invoke_agent(self, state, *, thread_id):
    invoke_config = {"configurable": {"thread_id": thread_id}}
    result = await self._agent.ainvoke(state, config=invoke_config)
    return result
```

#### _run_hooks()（loop.py:72-93）

依次执行所有钩子，遇到第一个非 CONTINUE 的信号就立即返回：

```python
async def _run_hooks(self, method, *args, **kwargs):
    for hook in self._hooks:
        handler = getattr(hook, method, None)
        if handler is None:
            continue
        resp = await handler(*args, **kwargs)
        if isinstance(resp, HookResponse) and resp.signal != LoopSignal.CONTINUE:
            return resp          # 某个钩子返回了 BREAK/RETRY/ROLLBACK
    return HookResponse()        # 全部通过，返回 CONTINUE
```

#### _notify_hooks()（loop.py:95-100）

即发即忘通知（不收集返回值）：

```python
async def _notify_hooks(self, method, *args, **kwargs):
    for hook in self._hooks:
        handler = getattr(hook, method, None)
        if handler:
            await handler(*args, **kwargs)
```

### Budget（budget.py:24-88）

三维预算管理器：

```python
@dataclass
class Budget:
    config: BudgetConfig          # max_iterations=25, max_tokens=0, max_time_seconds=0
    iterations_used: int = 0
    tokens_used: int = 0
    _start_time: float = field(default_factory=time.monotonic)

    def check(self) -> tuple[LoopSignal, str]:
        # 检查迭代次数
        if self.iterations_used >= self.config.max_iterations:
            return LoopSignal.BREAK, "迭代预算已耗尽"
        # 检查 token 数量
        if self.config.max_tokens > 0 and self.tokens_used >= self.config.max_tokens:
            return LoopSignal.BREAK, "Token 预算已耗尽"
        # 检查时间
        if self.config.max_time_seconds > 0 and self.elapsed_seconds() >= self.config.max_time_seconds:
            return LoopSignal.BREAK, "时间预算已耗尽"
        return LoopSignal.CONTINUE, ""
```

### LoopSignal / LoopResult / HookResponse（signals.py:1-57）

```python
class LoopSignal(Enum):
    CONTINUE = auto()    # 继续执行
    BREAK    = auto()    # 立即终止循环
    RETRY    = auto()    # 重新执行当前迭代（GoalLoop 独有）
    ROLLBACK = auto()    # 回退至上一个检查点（GoalLoop 独有）

@dataclass(frozen=True)
class LoopResult:
    success: bool           # 是否达成目标
    iterations: int         # 已执行的迭代次数
    reason: str = ""        # 终止原因
    final_state: dict = {}  # 终止时的状态快照
    evidence: str = ""      # 验证依据（如测试输出）

@dataclass(frozen=True)
class HookResponse:
    signal: LoopSignal = LoopSignal.CONTINUE
    message: str = ""
    state_patch: dict | None = None
```

---

## 五、三种模式对比总表

| | 裸 Agent | TurnLoop | GoalLoop |
|---|---|---|---|
| **实现位置** | `factory.py:86-95, 137` | `loop.py:107-169` | `loop.py:176-323` |
| **返回类型** | `CompiledGraph` | `TurnLoop` | `GoalLoop` |
| **调用方式** | `agent.ainvoke()` | `agent.run()` | `agent.run()` |
| **迭代次数** | 1次 | 最多N次 | 最多N次 |
| **预算控制** | 无 | 次数/token/时间 | 次数/token/时间 |
| **目标注入** | 无 | 无 | 首轮注入 SystemMessage |
| **验证器** | 无 | 无 | 每 N 轮运行验证 |
| **验证反馈** | 无 | 无 | 失败时注入 HumanMessage |
| **检查点** | 无 | 无 | 每轮保存 `{**state}` |
| **BREAK 信号** | 不适用 | ✅ | ✅ |
| **RETRY 信号** | 不适用 | ❌ | ✅ |
| **ROLLBACK 信号** | 不适用 | ❌ | ✅ |
| **典型场景** | 简单问答 | 多步任务 | 自主编程 |
