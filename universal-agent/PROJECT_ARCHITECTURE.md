# Universal Agent (uniagent) 项目架构文档

---

## 1. 项目总览

### 项目定位

**uniagent** 是一个基于 LangGraph 构建的中间件驱动、配置热重载的通用 Agent 框架。它将 LLM 智能体的构建过程抽象为可配置、可组合、可验证的工程化流水线，支持从简单的单轮对话到复杂的目标驱动自主执行等多种场景。

### 技术栈

| 层面 | 技术选型 |
|------|---------|
| 运行时 | Python >= 3.12 |
| LLM 框架 | LangGraph >= 0.4, LangChain-Core >= 0.3 |
| 数据建模 | Pydantic >= 2.0 |
| 配置格式 | YAML (PyYAML >= 6.0) |
| 构建系统 | Hatchling |
| 可选 LLM 提供商 | langchain-openai, langchain-anthropic |

### 核心依赖关系

```
pyproject.toml:13-18 — 四个核心依赖：langgraph, langchain-core, pydantic, pyyaml
pyproject.toml:20-26 — 可选依赖：openai / anthropic 适配器，pytest 测试框架
```

---

## 2. 目录结构

```
universal-agent/
├── pyproject.toml                    # 包元数据与依赖声明
├── config.example.yaml               # 配置文件模板
├── skills/                           # 示例技能包
│   └── example-skill/
│       ├── metadata.json             # 技能清单
│       ├── SKILL.md                  # 主要指令文件
│       ├── references/               # 渐进式披露参考文档
│       │   ├── guide.md
│       │   └── advanced.md
│       └── templates/
│           └── report.md
└── src/uniagent/                     # 核心源码
    ├── __init__.py                   # 顶层公共 API 导出
    ├── imports/                      # 动态导入/反射工具
    │   ├── __init__.py
    │   └── resolvers.py
    ├── config/                       # 配置系统（热重载 + ContextVar 隔离）
    │   ├── __init__.py
    │   ├── app_config.py
    │   ├── sub_configs.py
    │   └── reload_boundary.py
    ├── agents/                       # Agent 工厂层（SDK + 配置驱动）
    │   ├── __init__.py
    │   ├── factory.py
    │   ├── config_factory.py
    │   └── features.py
    ├── middleware/                    # 中间件系统（洋葱模型）
    │   ├── __init__.py
    │   ├── base.py
    │   ├── chain.py
    │   ├── positioning.py
    │   └── builtins/                 # 7 个内置中间件
    │       ├── __init__.py
    │       ├── dangling_tool_call.py
    │       ├── tool_error_handling.py
    │       ├── loop_detection.py
    │       ├── sandbox_middleware.py
    │       ├── summarization.py
    │       ├── token_usage.py
    │       └── skill_middleware.py
    ├── runtime/                      # 循环引擎
    │   ├── __init__.py
    │   ├── loop.py                   # GoalLoop / TurnLoop / BaseLoop
    │   ├── hooks.py                  # 生命周期钩子
    │   ├── signals.py                # 控制信号（CONTINUE/BREAK/RETRY/ROLLBACK）
    │   ├── budget.py                 # 预算管理器
    │   ├── checkpointer.py           # LangGraph 检查点工厂
    │   ├── context.py                # 用户上下文 ContextVar
    │   ├── protocols.py              # 跨层类型协议
    │   └── wip_hook.py               # WIP=1 约束钩子
    ├── tools/                        # 工具系统
    │   ├── __init__.py
    │   ├── registry.py               # 工具注册表
    │   ├── mcp_metadata.py           # MCP 工具标记
    │   ├── builtins/
    │   │   ├── __init__.py
    │   │   └── clarification.py      # 内置澄清工具
    │   └── deferred/                 # 延迟工具发现
    │       ├── __init__.py
    │       ├── assembly.py
    │       ├── catalog.py
    │       └── tool_search.py
    ├── skills/                       # 技能子系统
    │   ├── __init__.py
    │   ├── manifest.py               # 技能清单数据模型
    │   ├── loader.py                 # 渐进式内容加载器
    │   ├── registry.py               # 触发器索引与匹配
    │   └── injector.py               # 系统提示注入
    ├── state/                        # 状态持久化
    │   ├── __init__.py
    │   ├── thread_state.py           # 扩展的 AgentState
    │   ├── reducers.py               # 状态归约器
    │   ├── backend.py                # 可插拔持久化后端
    │   ├── feature_list.py           # 带状态机的功能跟踪
    │   ├── progress_log.py           # 仅追加迭代日志
    │   ├── decision_log.py           # 决策及理由日志
    │   └── wip.py                    # WIP=1 约束检查器
    ├── verification/                 # 验证系统
    │   ├── __init__.py
    │   ├── verifier.py               # 验证器协议
    │   └── builtins/
    │       ├── __init__.py
    │       ├── always_pass.py
    │       ├── command_verifier.py
    │       ├── llm_verifier.py
    │       └── composite_verifier.py
    └── sandbox/                      # 沙箱隔离
        ├── __init__.py
        ├── base.py                   # 沙箱抽象接口
        ├── provider.py               # 提供者抽象 + 单例
        ├── exceptions.py
        └── local/
            ├── __init__.py
            └── local_provider.py     # 本地文件系统沙箱
```

---

## 3. 每个模块的详细分析

---

### 3.1 imports/ — 动态导入/反射工具

**模块职责：** 将 `"pkg.mod:name"` 格式的点分路径字符串在运行时解析为真实的 Python 对象，是整个框架配置驱动设计的基础设施。

**核心函数：**

- `resolve_variable(path, expected_type=None)` — `resolvers.py:52-73`：导入任意对象，可选类型验证
- `resolve_class(path, base_class=None)` — `resolvers.py:76-98`：导入类，可选子类关系验证
- `_split_path(path)` — `resolvers.py:15-26`：拆分 `"pkg.mod:name"` 为模块路径和属性名
- `_import_attr(path)` — `resolvers.py:29-43`：底层的模块导入+属性获取

**设计思路：** 使用 `importlib.import_module` 实现延迟导入，使得配置文件中可以用字符串指定任意 Python 类/对象。`ResolveError` 提供了友好的中文错误信息，包含安装提示。

**依赖关系：** 无内部依赖，被 config、agents、tools、sandbox 等几乎所有模块使用。

---

### 3.2 config/ — 配置系统

**模块职责：** 提供基于 YAML 的分层配置，支持环境变量替换、文件签名热重载、ContextVar 栈隔离和启动期字段冻结。

#### 3.2.1 sub_configs.py — 子配置模型

**位置：** `config/sub_configs.py:1-88`

8 个 Pydantic BaseModel 子配置类：

| 配置类 | 用途 | 关键字段 |
|--------|------|---------|
| `ModelConfig` | LLM 提供商 | name, use（点分路径）, model, temperature, kwargs |
| `SandboxConfig` | 沙箱提供商 | use（默认 LocalSandboxProvider）, kwargs |
| `ToolConfig` | 单个工具 | name, use, group, enabled, kwargs |
| `ToolSearchConfig` | 延迟发现 | enabled, max_results |
| `LoopConfig` | 循环引擎 | max_iterations(25), max_tokens(0), max_time_seconds(0), verify_every(1), checkpoint_every(1) |
| `StateConfig` | 外部状态 | backend("local"), state_dir |
| `VerificationConfig` | 验证策略 | strategy, command, confidence_threshold(0.7), evaluator_model |
| `SkillConfig` | 技能子系统 | enabled, directories, max_active(3), auto_match |

#### 3.2.2 app_config.py — 应用根配置

**位置：** `config/app_config.py:1-190`

**核心类：**
- `AppConfig(BaseModel)` — 聚合所有子配置的根模型，`from_yaml()` 类方法从 YAML 加载
- `_ConfigCache` — 线程安全的单例缓存，基于 `(mtime, size, sha256)` 三元组文件签名实现热重载检测

**关键机制：**
1. **环境变量替换** (`app_config.py:31-58`)：`_substitute_env` 支持 `${VAR}` 和 `${VAR:default}` 语法，`_walk_substitute` 递归遍历 dict/list 结构
2. **热重载** (`app_config.py:101-147`)：`_ConfigCache.get()` 每次调用时比较文件签名，变更时自动重新加载
3. **ContextVar 栈** (`app_config.py:170-190`)：`push_current_app_config` / `pop_current_app_config` 支持请求级配置覆盖，优先级高于文件配置

#### 3.2.3 reload_boundary.py — 热重载边界

**位置：** `config/reload_boundary.py:1-48`

注册启动期冻结字段（目前仅注册了 `"sandbox"`），热重载时检测这些字段的变更并忽略，防止运行时破坏沙箱句柄等关键资源。

---

### 3.3 agents/ — Agent 工厂层

**模块职责：** 提供两级 Agent 创建入口 — SDK 级别的 `create_agent()` 和配置驱动的 `create_agent_from_config()`。

#### 3.3.1 features.py — 声明式特性开关

**位置：** `agents/features.py:1-101`

`AgentFeatures` 是一个 `@dataclass`，分为两个层级的特性标志：

**Agent 节点层（中间件）：**

| 标志 | 默认值 | 对应中间件 |
|------|--------|-----------|
| `sandbox` | False | SandboxMiddleware |
| `dangling_tool_call` | True | DanglingToolCallMiddleware |
| `tool_error_handling` | True | ToolErrorHandlingMiddleware |
| `loop_detection` | True | LoopDetectionMiddleware |
| `summarization` | False | SummarizationMiddleware |
| `token_usage` | True | TokenUsageMiddleware |
| `skill` | False | SkillMiddleware |

每个标志可取 `True`（使用默认实例）、`False`（禁用）或 `Middleware` 实例（自定义配置）。

**循环层：**
- `goal_loop` — 启用 GoalLoop 包装
- `wip_constraint` — WIP=1 约束
- `external_state` — 磁盘持久化
- `verification` — 验证策略名称

`resolve_middleware()` 方法 (`features.py:62-93`) 按固定顺序将标志解析为有序中间件列表。

#### 3.3.2 factory.py — SDK 工厂

**位置：** `agents/factory.py:1-137`

`create_agent()` 是核心构建函数，支持三种模式：

1. **裸 Agent**：不设 `goal` -> 返回 LangGraph `CompiledGraph`
2. **TurnLoop**：启用 `goal_loop` 特性但不设 `goal` -> 返回 `TurnLoop` 包装器
3. **GoalLoop**：设置 `goal` + `verifier` -> 返回 `GoalLoop` 包装器

关键流程 (`factory.py:60-137`)：
1. 解析中间件（`middleware` 参数直接传入，或 `features` 自动组装）
2. 调用 `create_react_agent()` 创建 LangGraph 内部 Agent
3. 将中间件附加到 Agent 对象上 (`agent._uniagent_middleware = chain`)
4. 解析预算 → 解析钩子 → 按需包装为循环

#### 3.3.3 config_factory.py — 配置驱动工厂

**位置：** `agents/config_factory.py:1-195`

`create_agent_from_config()` 是全配置化入口，8 步流水线：
1. 解析 LLM 模型（`_resolve_model` 通过反射实例化）
2. 加载工具（`get_available_tools`）
3. 组装延迟工具（MCP 工具分离）
4. 解析额外中间件
5. 解析特性开关
6. 解析循环预算
7. 设置技能子系统
8. 调用 `create_agent()`

还管理全局 `_skill_registry` 单例和 `_setup_wip_hook` 辅助函数。

---

### 3.4 middleware/ — 中间件系统

**模块职责：** 提供基于 LangGraph `AgentMiddleware` 的中间件抽象，支持 `@after/@before` 装饰器定位和拓扑排序链组装。

#### 3.4.1 base.py — 中间件基类

**位置：** `middleware/base.py:1-57`

`Middleware` 继承自 `AgentMiddleware[ThreadState]`，在两个层级运行：

- **Agent 节点层**：`before_agent(state)` / `after_agent(state)` / `wrap_model_call` / `wrap_tool_call`
- **循环层**：`loop_hooks()` 返回 `LoopHook` 实例列表

`__init_subclass__` 自动将类名设为 `name` 属性。

#### 3.4.2 positioning.py — 排序装饰器

**位置：** `middleware/positioning.py:1-55`

- `@after(AnchorClass)` — 标记当前中间件必须排在锚点之后
- `@before(AnchorClass)` — 标记当前中间件必须排在锚点之前

通过类属性 `__uniagent_next__` 和 `__uniagent_prev__` 存储定位元数据。

#### 3.4.3 chain.py — 链组装

**位置：** `middleware/chain.py:1-108`

`assemble_middleware_chain(built_in, extras, tail_type)`:
1. 以 `built_in` 为基础
2. 按 `@after/@before` 约束插入 `extras`
3. 可选将指定类型移至末尾
4. `_detect_cycles` 验证最终链满足所有约束

#### 3.4.4 内置中间件（7 个）

| 中间件 | 文件 | 职责 |
|--------|------|------|
| **DanglingToolCallMiddleware** | `dangling_tool_call.py` | 检测无对应 ToolMessage 的 AIMessage.tool_calls，插入合成占位 ToolMessage 防止 API 错误 |
| **ToolErrorHandlingMiddleware** | `tool_error_handling.py` | 包装工具调用，捕获异常转为错误 ToolMessage，防止 Agent 崩溃 |
| **LoopDetectionMiddleware** | `loop_detection.py` | 双层循环检测：Agent 层注入警告 HumanMessage（软控制），循环层发出 BREAK 信号（硬控制，`hard_limit` 默认 3 次） |
| **SandboxMiddleware** | `sandbox_middleware.py` | 延迟初始化沙箱，首次 `before_agent` 时通过 `SandboxProvider` 获取 sandbox_id |
| **SummarizationMiddleware** | `summarization.py` | 消息数超阈值（默认 20）时触发抽取式摘要，保留最近 6 条消息 |
| **TokenUsageMiddleware** | `token_usage.py` | 收集 AIMessage 的 token 用量统计，同步到 Budget 预算系统 |
| **SkillMiddleware** | `skill_middleware.py` | 拦截 HumanMessage，匹配技能注册表触发器，激活匹配技能 |

**循环检测的实现细节** (`loop_detection.py:118-126`)：通过 `_signature()` 函数将 tool_calls 转为可比较的字符串签名（按名称和参数排序），使用滑动窗口（默认 6）检测重复模式。

---

### 3.5 runtime/ — 循环引擎

**模块职责：** 封装 Agent 的迭代执行，提供预算控制、生命周期钩子、检查点和控制信号。

#### 3.5.1 signals.py — 控制信号

**位置：** `runtime/signals.py:1-57`

三个核心数据类型：

- `LoopSignal` 枚举：`CONTINUE`(继续) / `BREAK`(终止) / `RETRY`(重试当前) / `ROLLBACK`(回退检查点)
- `LoopResult` — 循环执行结果，含 `success`, `iterations`, `reason`, `final_state`, `evidence`
- `HookResponse` — 钩子返回值，含 `signal`, `message`, `state_patch`

#### 3.5.2 budget.py — 预算管理

**位置：** `runtime/budget.py:1-88`

`Budget` 是可变预算追踪器，支持三维限制：
- **迭代次数**：`max_iterations`（默认 25）
- **Token 数量**：`max_tokens`（0=不限）
- **时间**：`max_time_seconds`（0=不限）

`check()` 方法在每次迭代开始时调用，超限时返回 `(LoopSignal.BREAK, reason)`。

#### 3.5.3 hooks.py — 生命周期钩子

**位置：** `runtime/hooks.py:1-123`

`LoopHook` 抽象基类定义 5 个生命周期方法：
- `on_iteration_start` — 可阻止迭代执行
- `on_iteration_end` — 可触发 BREAK/RETRY/ROLLBACK
- `on_goal_achieved` — 目标达成通知
- `on_budget_exhausted` — 预算耗尽通知
- `on_error` — 异常处理（默认 BREAK）

两个内置钩子：
- `ProgressLogHook` — 记录迭代进度日志（**仅输出到控制台，未持久化到 StateBackend**）
- `TokenBudgetHook` — 同步 token 用量到 Budget（绝对值同步，非增量）

#### 3.5.4 loop.py — 循环引擎核心

**位置：** `runtime/loop.py:1-323`

三个循环类：

**BaseLoop** (`loop.py:31-100`)：
- 管理 Agent 引用、Budget、钩子列表
- `_invoke_agent()` — 单轮 Agent 推理
- `_run_hooks()` — 依次执行钩子，返回第一个非 CONTINUE 响应
- `_notify_hooks()` — 即发即忘通知

**TurnLoop** (`loop.py:107-169`)：
- 最多运行 N 次迭代的简单循环
- 无强制验证，仅依赖预算和钩子控制
- 流程：预算检查 -> 迭代前钩子 -> 运行 Agent -> 记录迭代 -> 迭代后钩子

**GoalLoop** (`loop.py:176-323`)：
- 目标驱动的验证循环，是自主 Agent 的核心抽象
- 额外参数：`goal`, `verifier`, `verify_every`, `inject_goal`
- 流程：预算检查 -> 迭代前钩子 -> 注入目标 -> 运行 Agent -> 保存检查点 -> 迭代后钩子 -> 验证
- 支持 ROLLBACK 信号回退到上一个检查点状态
- 验证失败时注入反馈 HumanMessage 指导 Agent 调整

#### 3.5.5 checkpointer.py — 检查点工厂

**位置：** `runtime/checkpointer.py:1-44`

`create_checkpointer(backend)` 支持 `"memory"` / `"sqlite"` / 点分导入路径三种模式。SQLite 不可用时优雅降级到内存模式。

#### 3.5.6 context.py — 用户上下文

**位置：** `runtime/context.py:1-32`

基于 `ContextVar` 的 `CurrentUser` 协议（需实现 `user_id` 和 `display_name`），通过 `set_current_user` / `get_current_user` 实现请求级用户身份传递，无需参数透传。

#### 3.5.7 protocols.py — 跨层协议

**位置：** `runtime/protocols.py:1-42`

定义 `LoopSignalBase` 和 `LoopHookBase`，供底层模块（如中间件）依赖，避免导入完整的 runtime 模块。

#### 3.5.8 wip_hook.py — WIP=1 约束钩子

**位置：** `runtime/wip_hook.py:1-70`

`WorkInProgressHook` 实现基于 FeatureList 的单任务约束：
- `on_iteration_start`：若无活跃功能则自动激活下一个待处理功能；若全部完成则发出 BREAK
- 基于工程研究：WIP=1 相比无约束多任务提升 37% 完成率

---

### 3.6 tools/ — 工具系统

**模块职责：** 工具注册、MCP 元数据标记、延迟发现和内置工具。

#### 3.6.1 registry.py — 工具注册表

**位置：** `tools/registry.py:1-87`

`get_available_tools(config, extra_tools, mcp_tools, excluded_names)`:
1. 配置工具 — 通过 `resolve_variable` 反射加载，类则尝试实例化
2. 额外工具 — 代码直接传入
3. MCP 工具 — 从 MCP 服务器发现

按 `tool.name` 去重（先出现者优先）。`_tool_name()` 支持 `BaseTool`、`.name` 属性和 `__name__`。

#### 3.6.2 mcp_metadata.py — MCP 标记

**位置：** `tools/mcp_metadata.py:1-23`

三个工具函数：`tag_mcp_tool` / `is_mcp_tool` / `get_mcp_metadata`，通过 `__uniagent_mcp__` 属性标记工具来源。

#### 3.6.3 deferred/ — 延迟工具发现

**assembly.py** (`tools/deferred/assembly.py:1-75`)：
- `assemble_deferred_tools` 将 MCP 工具分离到延迟目录，其余保持立即可用
- 若启用且有延迟工具，自动生成 `tool_search` 工具并加入立即工具列表

**catalog.py** (`tools/deferred/catalog.py:1-90`)：
- `DeferredToolCatalog` — 不可变的延迟工具目录（frozen dataclass）
- `search(query, max_results)` — 支持精确名称匹配 + 关键词评分搜索
- 评分机制：名称精确 10 分，名称包含子串 10 分，描述包含 5 分，词级别 3/1 分

**tool_search.py** (`tools/deferred/tool_search.py:1-62`)：
- `build_tool_search(catalog)` 创建闭包工具
- 搜索后通过 `Command(update={"promoted_tools": names})` 将工具名称提升到状态中

#### 3.6.4 builtins/clarification.py — 澄清工具

**位置：** `tools/builtins/clarification.py:1-19`

简单的 `@tool` 装饰函数 `ask_for_clarification(question)`，用于 Agent 向用户请求更多信息。当前为桩实现，返回格式化字符串。

---

### 3.7 skills/ — 技能子系统

**模块职责：** 实现基于触发器匹配、渐进式披露的技能注入机制，支持从文件系统扫描技能包并在运行时动态激活。

#### 3.7.1 manifest.py — 技能清单

**位置：** `skills/manifest.py:1-124`

三个 frozen dataclass：

- `TriggerRule` — 触发条件，支持 4 种类型：`keyword`(子串)、`prefix`(前缀)、`regex`(正则)、`intent`(意图)
- `ReferenceEntry` — 参考文档声明，`when` 字段控制加载时机：`always` / `on_demand` / `never`
- `SkillManifest` — 完整清单，含 name, description, triggers, references, templates, scripts, tags, promoted_tools

清单目录结构约定：
```
skill-name/
├── metadata.json    → SkillManifest
├── SKILL.md         → 主要指令
├── references/      → 渐进式参考文档
├── templates/       → 输出模板
└── scripts/         → 可执行脚本
```

#### 3.7.2 loader.py — 渐进式加载器

**位置：** `skills/loader.py:1-157`

`SkillLoader` 实现三阶段渐进式披露：
1. `load()` — 加载 SKILL.md + `when="always"` 的即时参考
2. `load_reference()` — 按需加载单个参考文件（带缓存）
3. `load_template()` — 按需加载模板

`SkillContent` dataclass 持有已加载内容，`all_references` 属性合并即时和按需加载的参考。

#### 3.7.3 registry.py — 注册表与匹配

**位置：** `skills/registry.py:1-245`

`SkillRegistry` 核心职责：
- `scan(*directories)` — 扫描含 `metadata.json` 的子目录
- `match(user_input)` — 遍历所有技能触发器评分，返回降序排列的匹配列表
- `activate(match)` — 通过 SkillLoader 加载技能内容

触发器评分机制 (`registry.py:186-232`)：

| 类型 | 评分规则 |
|------|---------|
| `prefix` | 前缀匹配 → 1.0，否则 0.0 |
| `keyword` | 精确匹配 1.0；包含则 min(0.9, 覆盖率+0.3) |
| `regex` | 匹配则 min(0.95, 覆盖率+0.4) |
| `intent` | 需外部分类器，当前返回 0.0 |

#### 3.7.4 injector.py — 系统提示注入

**位置：** `skills/injector.py:1-157`

`SkillInjector` 将激活的技能内容格式化为 HTML 注释包裹的段落，追加到系统提示末尾。支持多技能同时激活（`max_active_skills` 默认 3），超限时 LRU 驱逐最旧技能。

注入格式：
```
<!-- SKILL: skill-id -->
## Skill: Name
[SKILL.md 内容]
### References
[即时参考]
### Available Templates / Scripts
<!-- /SKILL: skill-id -->
```

---

### 3.8 state/ — 状态持久化

**模块职责：** 提供带归约器的线程状态、可插拔持久化后端、功能跟踪列表和日志系统。

#### 3.8.1 thread_state.py — 线程状态

**位置：** `state/thread_state.py:1-49`

`ThreadState` 继承 `AgentState`，使用 `Annotated[..., reducer]` 模式扩展 9 个字段：

| 字段 | 类型 | 归约器 | 用途 |
|------|------|--------|------|
| `sandbox` | dict/None | `merge_sandbox` | 沙箱句柄（冲突失败关闭） |
| `artifacts` | dict | `idempotent_merge` | 执行制品 |
| `promoted_tools` | list[str] | `dedup_list_merge` | 已提升的延迟工具 |
| `todos` | list[dict] | `dedup_list_merge` | 待办事项 |
| `summary` | str | `last_wins` | 对话摘要 |
| `token_usage` | dict[str,int] | `idempotent_merge` | Token 计数 |
| `current_task` | str/None | `last_wins` | 当前 WIP 任务 ID |
| `loop_iteration` | int | `last_wins` | 迭代编号 |
| `verification_result` | dict/None | `last_wins` | 验证结果 |

#### 3.8.2 reducers.py — 归约器

**位置：** `state/reducers.py:1-73`

四个归约器函数：
- `last_wins` — 最后写入胜出
- `idempotent_merge` — 字典合并（已有键可被覆盖）
- `dedup_list_merge` — 去重列表追加
- `merge_sandbox` — 带冲突检测的沙箱合并（不同 sandbox_id 抛异常）

`_hashable()` 辅助函数处理不可哈希元素的去重（降级使用 `id()`）。

#### 3.8.3 backend.py — 持久化后端

**位置：** `state/backend.py:1-88`

- `StateBackend` ABC — 4 个抽象方法：`load`, `save`, `delete`, `list_keys`
- `LocalFileBackend` — 本地 JSON 文件实现，每个 key 对应一个 `.json` 文件

#### 3.8.4 feature_list.py — 功能跟踪

**位置：** `state/feature_list.py:1-169`

`FeatureList` 实现带状态机的任务跟踪，5 种状态：`PENDING` → `ACTIVE` → `PASSING` / `FAILED` / `BLOCKED`。

核心约束：
- WIP=1：`activate()` 时自动将其他 ACTIVE 功能降级为 PENDING
- 状态转换需要证据：`mark_passing()` 必须提供 `evidence` 参数
- BLOCKED 状态的功能不能直接激活

支持延迟加载（`_ensure_loaded`），所有状态变更自动持久化。

#### 3.8.5 progress_log.py + decision_log.py — 日志系统

- `ProgressLog` (`progress_log.py:1-84`)：仅追加的迭代日志，每条记录含 iteration, timestamp, action, result, tokens_used
- `DecisionLog` (`decision_log.py:1-80`)：决策及理由日志，记录 decision, rationale, alternatives, context

两者均基于 `StateBackend` 实现持久化，模式一致。

#### 3.8.6 wip.py — WIP 约束检查器

**位置：** `state/wip.py:1-46`

`WorkInProgressConstraint` — 纯数据层的无状态检查器，提供 `check()` 和 `can_activate()` 预检查方法。与 `runtime/wip_hook.py` 的循环钩子配合使用。

---

### 3.9 verification/ — 验证系统

**模块职责：** 实现生成器/评估器分离原则，提供多种验证策略判断目标是否完成。

#### 3.9.1 verifier.py — 协议

**位置：** `verification/verifier.py:1-44`

- `VerificationResult` — frozen dataclass，含 `passed`, `evidence`, `confidence`(0-1), `layer`, `details`
- `Verifier` — `Protocol` 类，要求实现 `async verify(goal, state) -> VerificationResult`

#### 3.9.2 四种内置验证器

| 验证器 | 文件 | 机制 |
|--------|------|------|
| `AlwaysPassVerifier` | `always_pass.py` | 空操作，始终返回通过 |
| `CommandVerifier` | `command_verifier.py` | 运行 Shell 命令，退出码 0 = 通过；支持超时（默认 120s）；输出截取最后 2000 字符 |
| `LLMVerifier` | `llm_verifier.py` | 独立 LLM 评估器，要求返回 JSON `{"passed", "confidence", "evidence"}`；confidence 低于阈值则判定不通过 |
| `CompositeVerifier` | `composite_verifier.py` | 链式多层验证（如 lint -> test -> e2e），第一个失败层快速失败 |

`LLMVerifier` 的实现细节 (`llm_verifier.py:51-104`)：
- 将对话历史精简为最近 3000 字符摘要
- 解析 LLM 响应中的 JSON（支持 Markdown 代码块格式）
- 置信度阈值机制防止低置信度误判通过

---

### 3.10 sandbox/ — 沙箱隔离

**模块职责：** 提供隔离执行环境的抽象，支持可插拔的沙箱提供者。

#### 3.10.1 base.py — 沙箱接口

**位置：** `sandbox/base.py:1-30`

`Sandbox` ABC 定义 5 个方法：`sandbox_id`(属性), `execute_command`, `read_file`, `write_file`, `close`。

#### 3.10.2 provider.py — 提供者抽象

**位置：** `sandbox/provider.py:1-60`

`SandboxProvider` ABC 定义：`acquire`(获取) / `get`(查询) / `release`(释放) / `shutdown`(关闭全部)。

`get_sandbox_provider()` — 线程安全的懒加载单例，通过配置的 `sandbox.use` 路径反射实例化提供者类。

#### 3.10.3 local_provider.py — 本地实现

**位置：** `sandbox/local/local_provider.py:1-84`

- `LocalSandbox` — 以临时目录为后端，`execute_command` 使用 `asyncio.create_subprocess_shell`
- `LocalSandboxProvider` — 在系统临时目录下管理沙箱，`acquire` 生成 UUID 式 ID

> **注意：** 这是**无隔离**的本地实现，文件头注释已明确标注。生产环境需替换为 Docker/Firecracker 等真实隔离方案。

---

### 3.11 顶层模块

**`__init__.py`** (`src/uniagent/__init__.py:1-22`)：
导出 10 个核心公共 API：`create_agent`, `create_agent_from_config`, `AgentFeatures`, `AppConfig`, `get_app_config`, `GoalLoop`, `TurnLoop`, `LoopResult`, `Verifier`, `VerificationResult`。

---

## 4. 数据流图

### 完整请求流转路径

```
用户请求 (input_messages)
│
▼
┌─────────────────────────────────────────────────────────────────┐
│ create_agent_from_config()                                       │
│  1. AppConfig.from_yaml() → 解析配置（热重载 + 环境变量替换）         │
│  2. resolve_class(model.use) → 实例化 LLM                        │
│  3. get_available_tools() → 加载工具（配置+额外+MCP，去重）          │
│  4. assemble_deferred_tools() → MCP 工具分离到延迟目录               │
│  5. AgentFeatures.resolve_middleware() → 生成有序中间件列表           │
│  6. assemble_middleware_chain() → 拓扑排序组装                      │
│  7. create_react_agent() → LangGraph 内部 Agent                   │
│  8. Budget + LoopHooks → 包装为 GoalLoop/TurnLoop                 │
└─────────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────┐
│ GoalLoop.run()                                                    │
│                                                                   │
│  for i in range(budget.max_iterations):                           │
│    ┌──────────────────────────────────────────────────────────┐   │
│    │ 1. budget.check()  →  BREAK? → 返回失败结果                │   │
│    │ 2. hooks.on_iteration_start()  →  BREAK/ROLLBACK?         │   │
│    │ 3. (首次) 注入 SystemMessage[目标]                          │   │
│    │ 4. agent.ainvoke(state)                                    │   │
│    │    ┌──────────────────────────────────────────────────┐    │   │
│    │    │ LangGraph ReAct Agent 内部执行                      │    │   │
│    │    │                                                    │    │   │
│    │    │ 中间件链（按序）：                                    │    │   │
│    │    │  SkillMiddleware.before_agent()                     │    │   │
│    │    │   → 匹配技能触发器 → 注入技能内容                      │    │   │
│    │    │  SandboxMiddleware.before_agent()                   │    │   │
│    │    │   → 延迟获取沙箱                                     │    │   │
│    │    │  DanglingToolCallMiddleware.before_agent()          │    │   │
│    │    │   → 修补孤立 tool_calls                              │    │   │
│    │    │  ToolErrorHandlingMiddleware.wrap_tool_call()       │    │   │
│    │    │   → 异常转 ToolMessage                               │    │   │
│    │    │  LoopDetectionMiddleware.before_agent()             │    │   │
│    │    │   → 检测重复模式 → 注入警告                           │    │   │
│    │    │  SummarizationMiddleware.before_agent()             │    │   │
│    │    │   → 消息过多则摘要压缩                                │    │   │
│    │    │  TokenUsageMiddleware.after_agent()                 │    │   │
│    │    │   → 统计 token 用量                                  │    │   │
│    │    │                                                    │    │   │
│    │    │ LLM 推理 → tool_calls → 工具执行 → 结果              │    │   │
│    │    └──────────────────────────────────────────────────┘    │   │
│    │ 5. budget.record_iteration()                               │   │
│    │ 6. 保存检查点（棘轮模式）                                    │   │
│    │ 7. hooks.on_iteration_end() → BREAK/RETRY?                 │   │
│    │ 8. verifier.verify(goal, state)                            │   │
│    │    → 通过? → hooks.on_goal_achieved() → 返回成功            │   │
│    │    → 失败? → 注入反馈 HumanMessage → 继续迭代               │   │
│    └──────────────────────────────────────────────────────────┘   │
│                                                                   │
│  → 超出预算 → 返回 LoopResult(success=False)                      │
└─────────────────────────────────────────────────────────────────┘
│
▼
LoopResult { success, iterations, reason, final_state, evidence }
```

### 状态流转

```
ThreadState (LangGraph AgentState 扩展)
├── messages[]         ← LangGraph 原生消息流
├── sandbox{}          ← SandboxMiddleware 写入
├── artifacts{}        ← Agent 工具执行产出
├── promoted_tools[]   ← tool_search 工具提升
├── token_usage{}      ← TokenUsageMiddleware 统计
├── current_task       ← WIP 钩子设置
├── summary            ← SummarizationMiddleware 生成
├── loop_iteration     ← 循环引擎设置
└── verification_result ← 验证器写入
```

---

## 5. 设计模式

### 5.1 工厂模式 (Factory)
- **体现位置：** `agents/factory.py`, `agents/config_factory.py`
- **说明：** 两级工厂 — SDK 工厂 (`create_agent`) 接受原始对象，配置工厂 (`create_agent_from_config`) 接受配置并通过反射实例化所有依赖

### 5.2 策略模式 (Strategy)
- **体现位置：** `verification/verifier.py` (Verifier Protocol), `sandbox/provider.py` (SandboxProvider ABC)
- **说明：** 通过协议/ABC 定义接口，具体实现（Command/LLM/Composite/AlwaysPass）可互换

### 5.3 中间件/责任链模式 (Middleware / Chain of Responsibility)
- **体现位置：** `middleware/base.py`, `middleware/chain.py`
- **说明：** 有序中间件链，每个中间件可修改状态、拦截或透传

### 5.4 观察者模式 (Observer) — 钩子系统
- **体现位置：** `runtime/hooks.py`, `runtime/loop.py`
- **说明：** 循环引擎在关键节点通知所有注册的 LoopHook，钩子可返回控制信号影响执行流

### 5.5 单例模式 (Singleton)
- **体现位置：** `config/app_config.py` (_ConfigCache), `sandbox/provider.py` (_instance), `agents/config_factory.py` (_skill_registry)
- **说明：** 线程安全的懒加载单例，配合 `threading.Lock` 实现

### 5.6 服务定位器模式 (Service Locator)
- **体现位置：** `imports/resolvers.py`, `sandbox/provider.py`
- **说明：** 通过字符串路径在运行时定位并实例化服务

### 5.7 状态机模式 (State Machine)
- **体现位置：** `state/feature_list.py` (FeatureStatus 枚举 + 转换规则)
- **说明：** Feature 具有受控的状态转换路径：PENDING → ACTIVE → PASSING/FAILED/BLOCKED

### 5.8 渐进式披露模式 (Progressive Disclosure)
- **体现位置：** `skills/loader.py`, `skills/manifest.py` (ReferenceEntry.when)
- **说明：** 技能内容分阶段加载：核心指令始终加载 → 即时参考自动加载 → 其他参考按需加载

### 5.9 归约器模式 (Reducer)
- **体现位置：** `state/reducers.py`, `state/thread_state.py`
- **说明：** 使用 `Annotated[type, reducer_fn]` 为每个状态字段定义合并策略，类似 Redux/LangGraph 的 state 管理

### 5.10 棘轮检查点模式 (Ratchet Checkpoint)
- **体现位置：** `runtime/loop.py:274-275` (GoalLoop)
- **说明：** 每次迭代后保存状态快照，ROLLBACK 信号时可回退到上一个检查点

---

## 6. 已知问题和改进点

### 6.1 功能性断层

#### P1: ProgressLogHook 未连接 ProgressLog（日志不持久化）

- **位置：** `runtime/hooks.py:69-101` vs `state/progress_log.py`
- **问题：** `ProgressLogHook` 仅使用 `logger.info()` 输出到控制台，完全没有调用 `ProgressLog` 写入 `StateBackend`。`ProgressLog` 类有完整的持久化能力但无人调用。
- **影响：** 循环迭代日志在进程结束后全部丢失，跨会话连续性功能形同虚设。

#### P2: SkillMiddleware 激活后未注入系统提示

- **位置：** `middleware/builtins/skill_middleware.py`
- **问题：** `activate()` 将技能添加到注入器内部列表，但 `before_agent` 返回 `None`。`SkillInjector.inject(base_prompt)` 从未被调用来修改系统提示。
- **影响：** 技能内容激活了但对 Agent 行为没有实际影响。

#### P3: tool_search 提升后工具实际不可用

- **位置：** `tools/deferred/tool_search.py:47-49`
- **问题：** 将工具名称写入 `state["promoted_tools"]`，但无机制将实际工具对象注入 Agent 工具集。
- **影响：** 工具仅在名义上被"提升"，Agent 无法真正使用。

### 6.2 架构性问题

#### P4: 中间件附加方式不规范

- **位置：** `agents/factory.py:95`
- **问题：** `agent._uniagent_middleware = chain` 直接设置私有属性，未通过 `create_react_agent` 的 `middleware` 参数传入。

#### P5: GoalLoop 检查点是浅拷贝

- **位置：** `runtime/loop.py:275`
- **问题：** `{**state}` 仅浅拷贝，`messages` 等列表字段共享引用，ROLLBACK 可能回退到已污染的状态。

#### P6: 循环检测硬停止逻辑永不触发

- **位置：** `loop_detection.py:65-71`
- **问题：** `before_agent` 中重置 `_repeat_count = 0`，导致 `on_iteration_end` 钩子永远看不到超限状态。

#### P7: LocalSandbox 无安全隔离

- **位置：** `sandbox/local/local_provider.py`
- **问题：** 直接在宿主系统执行命令，无进程/网络/文件系统隔离。仅适合开发环境。

### 6.3 缺失功能

- **无测试代码**：整个项目没有 tests/ 目录
- **无 README.md**：`pyproject.toml` 引用了但文件不存在
- **intent 触发器未实现**：`skills/registry.py` 中返回 0.0
- **clarification 工具为桩实现**：无真实的 human-in-the-loop 机制
- **SummarizationMiddleware 使用简单抽取式摘要**：代码注释标注"生产环境中请替换为 LLM 调用"
