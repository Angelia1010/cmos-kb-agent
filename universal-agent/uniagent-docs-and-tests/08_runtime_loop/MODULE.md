# 08 - runtime/loop 模块：GoalLoop / TurnLoop 循环引擎

## 文件

- `uniagent/runtime/loop.py`

## 功能说明

循环引擎是框架的核心运行时抽象，封装 LangGraph 智能体并驱动迭代执行。

### BaseLoop（抽象基类）

所有循环的公共逻辑：

- **`_invoke_agent(state, thread_id)`**：执行一轮推理。**关键适配**：中间件链在此显式执行（before_agent 正序 → agent.ainvoke → after_agent 逆序），不依赖 langgraph 内部接口。
- **`_run_hooks(method, *args)`**：执行所有钩子的指定生命周期方法，返回第一个非 CONTINUE 的响应。
- **`_notify_hooks(method, *args)`**：即发即忘通知（不期待返回信号）。
- 自动添加 `TokenBudgetHook` 同步 token 用量到 Budget。

### TurnLoop（基于轮次的简单循环）

最多运行 N 次迭代，无强制验证：

```
for i in range(max_iterations):
    budget.check() → BREAK?
    hooks.on_iteration_start() → BREAK?
    agent.invoke()
    budget.record_iteration()
    hooks.on_iteration_end() → BREAK?
```

适用场景：对话型智能体、简单任务流水线。

### GoalLoop（目标驱动的自主循环）

包含强制验证和停止条件，是自主智能体工作流的主要抽象：

```
inject_goal → SystemMessage
for i in range(max_iterations):
    budget.check() → BREAK?
    hooks.on_iteration_start() → BREAK/ROLLBACK?
    agent.invoke()
    save checkpoint (棘轮模式)
    hooks.on_iteration_end() → BREAK/RETRY?

    if i % verify_every == 0:
        verifier.verify(goal, state)
        if passed → on_goal_achieved → return success
        else → inject feedback HumanMessage → continue
```

特有功能：
- **目标注入**：首次迭代时将目标作为 SystemMessage 注入。
- **验证反馈循环**：验证失败时将 evidence 注入为 HumanMessage，LLM 据此调整策略。
- **棘轮检查点**：每次迭代后保存状态快照，支持 ROLLBACK 回退。
- **可配置验证频率**：`verify_every` 控制每隔几次迭代验证一次。
