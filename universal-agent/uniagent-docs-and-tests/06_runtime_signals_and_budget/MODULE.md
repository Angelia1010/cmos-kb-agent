# 06 - runtime/signals + budget 模块：循环信号与预算管理

## 文件

- `uniagent/runtime/signals.py` — LoopSignal 枚举、LoopResult、HookResponse
- `uniagent/runtime/budget.py` — BudgetConfig、Budget

## 功能说明

### LoopSignal（循环控制信号）

钩子和预算检查发出的信号，引导循环迭代流程：

| 信号 | 含义 |
|------|------|
| `CONTINUE` | 继续下一步/下一次迭代 |
| `BREAK` | 立即终止循环 |
| `RETRY` | 重新执行当前迭代 |
| `ROLLBACK` | 回退至上一个检查点并重试 |

### LoopResult（循环执行结果）

| 字段 | 说明 |
|------|------|
| `success: bool` | 是否达成目标 |
| `iterations: int` | 已执行迭代次数 |
| `reason: str` | 终止原因 |
| `final_state: dict` | 终止时的状态快照 |
| `evidence: str` | 验证通过时的依据 |

### HookResponse（钩子返回值）

包含 `signal`（控制信号）、`message`（注入对话的消息）、`state_patch`（可选状态补丁）。

### Budget（预算追踪器）

三维预算强制执行：

| 维度 | 配置字段 | 说明 |
|------|---------|------|
| 迭代次数 | `max_iterations` | 默认25，0=不限 |
| Token数量 | `max_tokens` | 默认0（不限） |
| 时间 | `max_time_seconds` | 默认0（不限） |

`check()` 在每次迭代开始时调用，任一限制超出返回 `(BREAK, reason)`。
`record_iteration()` / `record_tokens()` 记录消耗。
`summary()` 返回人类可读的使用摘要。
