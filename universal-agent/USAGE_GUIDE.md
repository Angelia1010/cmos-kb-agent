# Universal Agent 使用指南 — 核心概念详解

---

## 一、中间件（Middleware）的作用

中间件就是 **Agent 执行前后的拦截器**，可以修改输入/输出状态、包装工具调用、注入消息等。类似 Web 框架里的中间件（Express/Koa）。

### 生命周期位置

```
用户消息进入
  │
  ▼
┌─────────────────────────────────────────┐
│  before_agent()  ← 中间件按顺序执行       │
│    ↓                                     │
│  LLM 推理 ← wrap_model_call() 包装       │
│    ↓                                     │
│  工具执行 ← wrap_tool_call() 包装         │
│    ↓                                     │
│  after_agent()  ← 中间件按逆序执行        │
└─────────────────────────────────────────┘
  │
  ▼
返回结果
```

### 例1: DanglingToolCallMiddleware — 修补断裂的对话

**场景：** Agent 上一轮说"我要调用搜索工具"（AIMessage 里有 tool_calls），但因为中断/超时，工具结果（ToolMessage）丢失了。下一轮 LLM API 会报错，因为它期望每个 tool_call 都有对应的 ToolMessage。

```
消息历史（有问题）：
  [HumanMessage] "帮我查天气"
  [AIMessage]    tool_calls: [{id: "abc", name: "search"}]
  ← 缺少 ToolMessage！LLM API 会报错

中间件 before_agent 介入后：
  [HumanMessage] "帮我查天气"
  [AIMessage]    tool_calls: [{id: "abc", name: "search"}]
  [ToolMessage]  "[工具调用未执行 — 结果不可用]"   ← 自动补上
```

### 例2: ToolErrorHandlingMiddleware — 工具崩溃不影响 Agent

**场景：** Agent 调用了一个工具，工具内部抛异常了。没有中间件的话整个 Agent 直接崩溃；有了中间件，异常被捕获并转为一条错误消息返回给 LLM，让 LLM 自己决定怎么办。

```
没有中间件：
  Agent 调用 search_tool → 抛出 ConnectionError → Agent 进程崩溃 💥

有中间件：
  Agent 调用 search_tool → 抛出 ConnectionError
    → 中间件捕获 → 返回 ToolMessage("[错误] ConnectionError: ...")
    → LLM 看到错误 → "搜索失败了，让我换个方式试试" ✅
```

### 例3: LoopDetectionMiddleware — 防止 Agent 陷入死循环

**场景：** Agent 反复调用同一个工具、传同样的参数，陷入无限循环。

```
第1次: search("python教程")  → 没找到
第2次: search("python教程")  → 没找到
第3次: search("python教程")  ← 中间件检测到重复3次！

中间件注入警告：
  [HumanMessage] "[系统] 您似乎陷入了重复调用相同工具的循环中。
   请停止并尝试不同的方法。"

Agent 看到后换策略 → search("Python 入门指南") ✅
```

---

## 二、create_agent() vs create_agent_from_config() 的区别

核心区别：**一个用代码传参，一个用 YAML 配置文件驱动**。

### create_agent() — SDK 级别（程序员用代码直接控制）

```python
from langchain_openai import ChatOpenAI
from uniagent import create_agent, AgentFeatures

# 你自己实例化所有东西
model = ChatOpenAI(model="gpt-4o", temperature=0)
tools = [my_search_tool, my_calculator_tool]

# 模式1：裸 Agent（最简单）
agent = create_agent(model, tools)
result = await agent.ainvoke({"messages": [{"role": "user", "content": "你好"}]})

# 模式2：带自定义特性
agent = create_agent(
    model,
    tools,
    features=AgentFeatures(
        loop_detection=True,     # 开启循环检测
        summarization=True,      # 开启对话摘要
        sandbox=False,           # 不用沙箱
    ),
    system_prompt="你是一个Python专家。",
)

# 模式3：GoalLoop（目标驱动）
agent = create_agent(
    model,
    tools,
    goal="写一个贪吃蛇游戏",
    verifier=CommandVerifier("python -m pytest tests/"),
    budget=BudgetConfig(max_iterations=10),
)
result = await agent.run()
```

**特点：** 灵活、代码级控制，适合嵌入到其他 Python 项目中。

### create_agent_from_config() — 配置驱动（运维/非开发人员友好）

先写一个 `config.yaml`：

```yaml
models:
  - name: default
    use: "langchain_openai:ChatOpenAI"
    model: gpt-4o
    temperature: 0.0

tools:
  - name: search
    use: "mytools:search_tool"
    enabled: true
  - name: calculator
    use: "mytools:calc_tool"
    enabled: true

loop:
  max_iterations: 25
  max_tokens: 100000

verification:
  strategy: command
  command: "python -m pytest tests/"

skills:
  enabled: true
  directories: ["skills/"]
```

然后一行代码搞定：

```python
from uniagent import create_agent_from_config

# 自动读取 config.yaml，实例化模型、加载工具、组装中间件
agent = create_agent_from_config(goal="写一个贪吃蛇游戏")
result = await agent.run()
```

**特点：** 修改配置不需要改代码，支持热重载（改了 yaml 自动生效），适合生产部署。

### 对比总结

| | `create_agent()` | `create_agent_from_config()` |
|---|---|---|
| **输入** | Python 对象（model, tools...） | YAML 配置文件 |
| **模型实例化** | 你自己 `ChatOpenAI(...)` | 框架通过反射自动创建 |
| **工具加载** | 你传入工具列表 | 框架从配置读取点分路径自动导入 |
| **热重载** | 不支持 | 支持（改 yaml 自动生效） |
| **适用场景** | 开发/嵌入其他项目 | 生产部署/运维管理 |
| **调用关系** | 底层 | 内部最终调用 `create_agent()` |

---

## 三、Loop 的三种模式

### 模式1：裸 Agent（无循环）

```python
agent = create_agent(model, tools)
result = await agent.ainvoke({
    "messages": [{"role": "user", "content": "1+1等于几"}]
})
```

```
用户 → Agent 推理一次 → 返回结果（结束）
```

**适用于：** 简单问答、单轮对话。

### 模式2：TurnLoop（限次循环）

```python
agent = create_agent(
    model, tools,
    features=AgentFeatures(goal_loop=True),
    budget=BudgetConfig(max_iterations=5),
)
result = await agent.run(
    input_messages=[{"role": "user", "content": "帮我整理这些文件"}]
)
```

```
用户 → Agent 推理 → 工具调用 → Agent 推理 → 工具调用 → ... → 最多5轮后停止
         ↑                                                      │
         └──────────── 每轮检查预算（次数/token/时间）───────────────┘
```

**适用于：** 多步骤任务，但不需要验证目标是否完成。比如"帮我把这个目录下的文件改名"，Agent 可能需要多轮工具调用，但你信任它自己能判断什么时候做完。

### 模式3：GoalLoop（目标驱动 + 验证）

```python
from uniagent.verification.builtins import CommandVerifier, LLMVerifier

# 例子A：用命令验证（测试必须通过）
agent = create_agent(
    model, tools,
    goal="用 Python 实现一个 TODO 应用，包含增删改查功能",
    verifier=CommandVerifier("python -m pytest tests/ -v"),
    budget=BudgetConfig(max_iterations=15, max_time_seconds=300),
)
result = await agent.run()
# result.success  → True/False
# result.evidence → 测试输出

# 例子B：用 LLM 验证（独立评估器判断）
agent = create_agent(
    model, tools,
    goal="写一篇关于量子计算的科普文章，至少1000字",
    verifier=LLMVerifier(
        model=ChatOpenAI(model="gpt-4o-mini"),  # 用不同的模型做评估
        confidence_threshold=0.8,
    ),
    budget=BudgetConfig(max_iterations=10),
)
result = await agent.run()
```

GoalLoop 执行流程：

```
用户设定目标 → 注入 SystemMessage[目标]
  │
  ▼
┌──────────────────────────────────────────────────┐
│ 迭代1: Agent 写代码 → 保存检查点                      │
│   → 验证器运行 pytest → 失败（3个测试没过）              │
│   → 注入反馈: "[验证失败] 3个测试未通过，请修复"          │
│                                                    │
│ 迭代2: Agent 修 bug → 保存检查点                      │
│   → 验证器运行 pytest → 失败（还有1个）                  │
│   → 注入反馈: "[验证失败] 1个测试未通过"                 │
│                                                    │
│ 迭代3: Agent 再修 → 保存检查点                         │
│   → 验证器运行 pytest → 全部通过 ✅                      │
│   → on_goal_achieved() → 返回成功                     │
└──────────────────────────────────────────────────┘
  │
  ▼
LoopResult(success=True, iterations=3, evidence="6 passed")
```

**适用于：** 自主编码、复杂任务。Agent 持续工作直到验证器确认目标完成，或预算耗尽。这是框架最核心的能力——"写代码 → 跑测试 → 改 bug → 再跑测试"的自动闭环。

### 三种模式对比

| | 裸 Agent | TurnLoop | GoalLoop |
|---|---|---|---|
| **迭代次数** | 1次 | 最多N次 | 最多N次 |
| **验证** | 无 | 无 | 每轮验证 |
| **预算控制** | 无 | 次数/token/时间 | 次数/token/时间 |
| **自动反馈** | 无 | 无 | 验证失败自动注入反馈 |
| **检查点** | 无 | 无 | 每轮保存，支持回退 |
| **典型场景** | 问答 | 多步任务 | 自主编程 |

---

## 四、中间件（Middleware）和钩子（Hook）的区别

中间件和钩子都能在执行过程中"插一脚"，但它们运行在**不同的层级**，能力完全不同。

### 运行层级对比

```
┌─── GoalLoop / TurnLoop ──────────────────────────────────────────┐
│                                                                   │
│  Hook: on_iteration_start()          ← 钩子在这层（迭代级别）      │
│    │                                                              │
│    ▼                                                              │
│  ┌─── Agent 节点 ─────────────────────────────────────────────┐   │
│  │                                                             │   │
│  │  Middleware: before_agent()       ← 中间件在这层（节点内部）  │   │
│  │    │                                                        │   │
│  │    ▼                                                        │   │
│  │  LLM 推理 ← Middleware: wrap_model_call()                   │   │
│  │    │                                                        │   │
│  │    ▼                                                        │   │
│  │  工具执行 ← Middleware: wrap_tool_call()                     │   │
│  │    │                                                        │   │
│  │    ▼                                                        │   │
│  │  Middleware: after_agent()                                   │   │
│  │                                                             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│    │                                                              │
│    ▼                                                              │
│  Hook: on_iteration_end()                                         │
│    │                                                              │
│    ▼                                                              │
│  验证器 → Hook: on_goal_achieved() / on_budget_exhausted()        │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

**简单理解：**
- **中间件**运行在 Agent 节点**内部**，管的是"LLM 怎么推理、工具怎么调用、消息怎么处理"
- **钩子**运行在循环引擎**外部**，管的是"这轮迭代要不要执行、执行完了要不要继续、要不要回退"

### 能力对比

| | 中间件 (Middleware) | 钩子 (LoopHook) |
|---|---|---|
| **运行层级** | Agent 节点内部（单次推理） | 循环引擎迭代级别 |
| **触发时机** | 每次 LLM 调用前后 | 每轮迭代开始/结束时 |
| **能做什么** | 修改消息、包装工具调用、注入提示、统计 token | 发出控制信号（停止/重试/回退） |
| **不能做什么** | 不能终止整个循环 | 不能修改消息内容或包装工具 |
| **控制粒度** | 细粒度（单条消息、单次工具调用） | 粗粒度（整轮迭代） |
| **返回值** | 修改后的 state 或 None | `HookResponse`（含控制信号） |
| **有无循环都能用** | 裸 Agent 也能用 | 必须在 Loop 中才有意义 |

### 控制信号——钩子独有的能力

钩子能返回 4 种控制信号，中间件做不到：

```python
class LoopSignal(Enum):
    CONTINUE  = "continue"   # 继续下一轮（默认）
    BREAK     = "break"      # 立即终止循环
    RETRY     = "retry"      # 重试当前这轮
    ROLLBACK  = "rollback"   # 回退到上一个检查点
```

### 用具体例子说明区别

#### 场景：Agent 消耗了太多 token

**中间件做的事（TokenUsageMiddleware）：**

```
Agent 推理完毕
  → after_agent() 执行
  → 从 AIMessage.usage_metadata 里提取 token 数
  → 写入 state["token_usage"] = {total_tokens: 15000}
  → 结束，把数据传出去，自己不做任何决策
```

中间件只负责**收集数据**，不决定要不要停。

**钩子做的事（TokenBudgetHook）：**

```
一轮迭代结束
  → on_iteration_end() 执行
  → 读取 state["token_usage"]["total_tokens"] = 15000
  → 同步到 Budget: budget.tokens_used = 15000
  → 下一轮开始前，Budget.check() 发现超过 max_tokens
  → 返回 HookResponse(signal=BREAK, message="Token 预算耗尽")
  → 整个循环终止
```

钩子负责**做决策**——够了就停。

#### 场景：循环检测（LoopDetectionMiddleware 跨层工作）

这个中间件比较特殊，**同时在两个层级工作**：

```python
class LoopDetectionMiddleware(Middleware):
    # 中间件层：软控制 —— 注入警告消息
    async def before_agent(self, state):
        if self._repeat_count >= self._hard_limit:
            warning = HumanMessage(content="[系统] 您陷入了循环...")
            return {"messages": messages + [warning]}

    # 循环层：硬控制 —— 通过 loop_hooks() 暴露一个钩子
    def loop_hooks(self):
        class _HardStop(LoopHook):
            async def on_iteration_end(self_hook, iteration, state, output):
                if self._repeat_count >= self._hard_limit:
                    return HookResponse(signal=LoopSignal.BREAK,
                                        message="强制停止循环")
                return HookResponse()
        return [_HardStop()]
```

| 层级 | 做了什么 | 效果 |
|------|---------|------|
| 中间件层 | 注入警告消息 | 软控制：LLM **可能**会改变行为（也可能忽略） |
| 钩子层 | 发出 BREAK 信号 | 硬控制：循环**一定**会终止 |

### 一句话总结

> **中间件是 Agent 的"秘书"** — 帮忙整理输入、修补问题、收集数据，但不做决策。
>
> **钩子是循环的"裁判"** — 不碰具体执行细节，但有权吹哨叫停、要求重来。

### 什么时候该用中间件，什么时候该用钩子？

| 需求 | 用中间件 | 用钩子 |
|------|---------|-------|
| 修改/注入消息 | ✅ | ❌ |
| 包装工具调用（加 try/catch） | ✅ | ❌ |
| 统计 token 用量 | ✅ | ❌ |
| 压缩对话历史 | ✅ | ❌ |
| 根据预算决定是否停止 | ❌ | ✅ |
| 目标达成后触发通知 | ❌ | ✅ |
| 出错后决定重试还是终止 | ❌ | ✅ |
| 回退到之前的检查点 | ❌ | ✅ |
| 强制执行 WIP=1 约束 | ❌ | ✅ |
| 记录迭代日志 | ❌ | ✅ |
| 需要同时做两件事 | ✅ 通过 `loop_hooks()` 桥接两层 | — |
