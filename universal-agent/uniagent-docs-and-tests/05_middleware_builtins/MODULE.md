# 05 - middleware/builtins 模块：5个内置中间件

## 文件

- `dangling_tool_call.py` — 悬空工具调用修补
- `tool_error_handling.py` — 工具异常捕获
- `loop_detection.py` — 重复调用循环检测
- `skill_middleware.py` — 技能自动匹配注入
- `token_usage.py` — Token用量统计

## 功能说明

### 1. DanglingToolCallMiddleware

**问题**：AIMessage 中有 `tool_calls` 但消息历史中缺少对应的 `ToolMessage`（中断/部分执行导致），会让 LLM API 报错。

**方案**：`before_agent` 时扫描消息列表，为孤立的 `tool_calls` 插入合成 `ToolMessage`（内容为"[工具调用未执行]"）。

### 2. ToolErrorHandlingMiddleware

**问题**：工具执行抛异常会导致整个 Agent 循环崩溃。

**方案**：提供 `wrap_tool_call()` 上下文管理器，将异常捕获并转为包含错误信息的 `ToolMessage`，让 LLM 自行决定如何继续。

### 3. LoopDetectionMiddleware

**双层检测**：

- **Agent节点层（软控制）**：跟踪最近 N 次工具调用签名（名称+参数），连续重复 ≥ `hard_limit` 次时注入警告 `HumanMessage`。
- **循环层（硬控制）**：通过 `loop_hooks()` 暴露 `LoopHook`，达到硬限制后发出 `LoopSignal.BREAK` 强制停止。

签名算法：将 `tool_calls` 中每个调用的名称和排序后的参数拼接为字符串，再排序拼接。

### 4. SkillMiddleware

**流程**：
1. `before_agent` 时提取最新 `HumanMessage`。
2. 与技能注册表的触发器匹配（score > 0.3）。
3. 匹配成功则加载技能内容（SKILL.md + 参考），作为 `SystemMessage` 追加进消息流。

**适配**：原实现依赖 `AgentMiddleware` 的提示词管道，不存在于当前 langgraph；改为直接追加 `SystemMessage`。

### 5. TokenUsageMiddleware

**功能**：在 `after_agent` 时扫描所有 `AIMessage` 的 `usage_metadata` 或 `response_metadata`，累加 prompt/completion tokens 到 `state["token_usage"]`。

可选：持有 Budget 引用，将用量同步到预算系统实现硬限制。
