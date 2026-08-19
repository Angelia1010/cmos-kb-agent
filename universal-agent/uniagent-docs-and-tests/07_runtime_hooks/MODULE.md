# 07 - runtime/hooks 模块：循环级钩子系统

## 文件

- `uniagent/runtime/hooks.py`

## 功能说明

### LoopHook 基类

与中间件（Agent节点级）不同，循环钩子运行在 GoalLoop/TurnLoop 的**迭代级别**，可发出硬性控制信号。

| 生命周期方法 | 触发时机 | 返回类型 |
|-------------|---------|---------|
| `on_iteration_start(iteration, state)` | 每次迭代开始前 | `HookResponse`（可阻止迭代） |
| `on_iteration_end(iteration, state, output)` | 每次迭代结束后 | `HookResponse`（可 BREAK/RETRY/ROLLBACK） |
| `on_goal_achieved(state, evidence)` | 验证器确认目标达成 | `None`（通知性） |
| `on_budget_exhausted(state, reason)` | 预算限制触发 | `None`（通知性） |
| `on_error(state, error)` | 未处理异常 | `HookResponse`（默认 BREAK） |

### 内置钩子

| 钩子 | 功能 |
|------|------|
| `ProgressLogHook` | 记录迭代进度日志（开始/结束/目标达成/预算耗尽） |
| `TokenBudgetHook` | 将 `state["token_usage"]` 中的 token 数同步到 Budget 对象（绝对值同步，非增量） |

### 中间件 ↔ 钩子的关系

中间件可通过 `loop_hooks()` 方法返回钩子实例，同时参与 Agent 节点层和循环迭代层的生命周期。例如 `LoopDetectionMiddleware` 在 Agent 层注入警告消息，同时在循环层通过钩子发出 `BREAK`。
