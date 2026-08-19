# 13 - agents 模块：Agent 工厂

## 文件

- `uniagent/agents/factory.py` — `create_agent()` SDK层工厂
- `uniagent/agents/config_factory.py` — `create_agent_from_config()` 配置驱动工厂
- `uniagent/agents/features.py` — `AgentFeatures` 声明式特性标志

## 功能说明

### create_agent() — SDK层工厂

三种使用模式：

| 条件 | 返回类型 | 说明 |
|------|---------|------|
| 不设 `goal` | `CompiledGraph` | 裸 Agent，直接 `invoke()` |
| 设 `budget` 不设 `goal` | `TurnLoop` | 基于轮次的简单循环 |
| 设 `goal` + `verifier` | `GoalLoop` | 目标驱动的自主循环 |

中间件组装与循环选择正交——中间件始终应用于内部 Agent。

**互斥规则**：`middleware` 和 `features` 不能同时指定。

### create_agent_from_config() — 配置驱动工厂

完全由 `AppConfig` 驱动的创建流程：

1. 解析 LLM 模型（`models[0]` → 通过反射实例化）
2. 加载工具（配置 + extra + MCP）
3. 解析额外中间件（配置中的点分路径）
4. 构建预算
5. 初始化技能子系统（若 `skills.enabled`）
6. 调用 `create_agent()` 构建

### AgentFeatures — 声明式特性开关

用布尔值或中间件实例声明特性：

**Agent节点层（中间件）**：
- `dangling_tool_call` — 悬空工具调用修补（默认开）
- `tool_error_handling` — 工具异常捕获（默认开）
- `loop_detection` — 循环检测（默认开）
- `token_usage` — Token统计（默认开）
- `skill` — 技能自动匹配（默认关）

**循环层**：
- `goal_loop` — 启用 GoalLoop
- `verification` — 验证策略

`resolve_middleware()` 将布尔值解析为默认中间件实例，`Middleware` 实例直接使用，`False` 跳过。
