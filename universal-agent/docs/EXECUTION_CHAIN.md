# KB-Agent 执行链路与调用关系详解

> 适用版本：KB-Agent v2.0.2（裁剪版 uniagent 框架）
> 本文以“一次用户提问从进入到返回”为主线，逐层拆解每个组件的职责、函数级调用关系、
> 数据在各阶段的流转形态，以及降级/重试等异常路径。所有路径均给出源码位置。

---

## 目录

1. [全景图](#1-全景图)
2. [入口：MainAgent（固定编排层）](#2-入口mainagent固定编排层)
3. [共享基础设施](#3-共享基础设施)
4. [阶段①：检索子智能体（GoalLoop 自主循环）](#4-阶段检索子智能体goalloop-自主循环)
5. [阶段②：处理子智能体（裸 ReAct + 技能注入）](#5-阶段处理子智能体裸-react--技能注入)
6. [阶段③：答案子智能体（直调 LLM + 确定性锚定）](#6-阶段答案子智能体直调-llm--确定性锚定)
7. [降级兜底路径](#7-降级兜底路径)
8. [uniagent 框架层深挖](#8-uniagent-框架层深挖)
9. [离线模式：ScriptedChatModel 的行为脚本](#9-离线模式scriptedchatmodel-的行为脚本)
10. [端到端时序图](#10-端到端时序图)
11. [数据结构全程流转](#11-数据结构全程流转)
12. [Trace 事件清单](#12-trace-事件清单)
13. [注意事项与已知差异](#13-注意事项与已知差异)

---

## 1. 全景图

### 1.1 分层结构

```
┌─────────────────────────────────────────────────────────────────┐
│ 演示/测试入口                                                     │
│   main.py（4场景）  test_kbagent_e2e.py（TC01~TC07）             │
├─────────────────────────────────────────────────────────────────┤
│ kbagent 业务层（src/kbagent/）                                    │
│   MainAgent ── 固定三阶段编排 + 降级兜底 + 全链路 trace            │
│     ├─ RetrievalSubAgent   检索（GoalLoop 自主循环）              │
│     ├─ ProcessingSubAgent  处理（裸 ReAct + SkillMiddleware）    │
│     └─ AnswerSubAgent      答案（直调 LLM + 逐句锚定校验）        │
│   shared/：Workspace(请求隔离) / Tracer / Config / search / lexicon│
│   scripted_model.py：离线 Mock LLM（规则模拟工具决策）             │
├─────────────────────────────────────────────────────────────────┤
│ uniagent 框架层（src/uniagent/）                                  │
│   agents/     create_agent 工厂（三种组装模式）                    │
│   runtime/    GoalLoop / TurnLoop / Budget / LoopHook / signals  │
│   middleware/ 洋葱模型中间件链（5 个内置）                         │
│   skills/     技能注册表 / 渐进式披露 / 脚本工具加载               │
│   verification/ Verifier 协议 + 内置验证器                        │
│   state/      ThreadState（扩展 AgentState + reducers）          │
│   models/     ModelFactory（配置驱动建模，本演示未走此路）         │
├─────────────────────────────────────────────────────────────────┤
│ 外部依赖：LangGraph create_react_agent / LangChain-Core          │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 一次请求的主链路

```
用户 Query
   │
   ▼
MainAgent.run(query) ── asyncio.run ──▶ MainAgent.arun(query)
   │  新建 Tracer（trace_id 贯穿）；创建 RunWorkspace 并写入 ContextVar
   │
   ├─① await RetrievalSubAgent(model, cfg, tracer).run(query)   每次请求新建
   │      └─ create_agent(..., goal, verifier, budget) → GoalLoop
   │            └─ 每轮: Budget检查 → 中间件before → ReAct(模型↔工具)
   │                      → 中间件after → SufficiencyVerifier 验证
   │      ◀── List[Chunk]（写入 ws.data["chunks"]，失败时携最优退出）
   │
   ├─② await self._processing.run(query, chunks)                实例复用
   │      └─ 手工执行中间件 before_agent（技能在此注入）
   │         → agent.ainvoke（7个处理工具，LLM 自主取舍）
   │         → 产出为空时走确定性保底流水线
   │      ◀── List[Chunk]（清洗/去噪/去重/结构化/排序后）
   │
   ├─③ AnswerSubAgent(model, cfg, tracer).run(query, processed, trace_id)  同步
   │      └─ select_fragments（top4，同文档≤2）
   │         → generate：model.invoke([TASK:answer]) 组织答案
   │         → 逐句 model.invoke([TASK:anchor_check]) 锚定校验
   │         → 硬事实锚定失败直接删句；软表述标注"建议核实"
   │      ◀── FinalAnswer（业务说明/办理建议/句子/知识来源）
   │
   └─ 任一环节抛异常 ──▶ _degrade(query)：原始query关键词单轮检索，
                          返回原始片段（degraded=True），坐席永远有东西可看
```

### 1.3 组件职责速查表

| 组件 | 源码位置 | 自主性 | 护栏（确定性代码） |
|---|---|---|---|
| MainAgent | `kbagent/main_agent.py:30` | 无（固定编排） | 异常捕获 + 降级兜底 |
| RetrievalSubAgent | `kbagent/retrieval/agent.py:28` | 工具顺序/参数由 LLM 定 | GoalLoop 轮次时间预算 + 充分性规则 + DSL 白名单 |
| ProcessingSubAgent | `kbagent/processing/agent.py:28` | 清洗工具取舍由 LLM 定 | 空产出保底流水线 + “不裁剪片段”写入提示词 |
| AnswerSubAgent | `kbagent/answer/agent.py:17` | 素材取舍/答案组织由 LLM 定 | 逐句锚定校验为纯代码，硬事实零容忍 |
| ScriptedChatModel | `kbagent/scripted_model.py:45` | —（离线替身） | 让 ReAct/反馈注入机制真实跑通 |

---

## 2. 入口：MainAgent（固定编排层）

### 2.1 构造 `MainAgent.__init__`（`main_agent.py:31`）

```
MainAgent(model, es, cfg=DEFAULT_CONFIG, enable_skills=True, skill_dirs=None)
  ├─ self.tracer = Tracer()                      # 每次 arun 会重建
  ├─ enable_skills=True 时:
  │    └─ register_skill_directory(d)            # uniagent/agents/config_factory.py:225
  │         └─ SkillRegistry 单例（线程安全延迟初始化）→ scan("skills")
  │              └─ 每个含 metadata.json 的子目录 → SkillManifest.from_json → 注册
  │                 （taocan-skill 在此被注册，正则触发器预编译）
  └─ self._processing = ProcessingSubAgent(model, tracer, enable_skills)
       # 处理子智能体创建成本高（create_agent 组装链），全实例生命周期复用；
       # 检索/答案子智能体则每次请求新建
```

### 2.2 同步/异步入口

```
MainAgent.run(query)                      main_agent.py:47
  └─ asyncio.run(self.arun(query))        # 已在事件循环中时须直接调 arun()

MainAgent.arun(query)                     main_agent.py:50
  ├─ self.tracer = Tracer()               # 每次请求新 trace（新 trace_id）
  ├─ self._processing.tracer = self.tracer  # 复用实例须同步 tracer
  ├─ tracer.log("run", "start", query=query)
  ├─ ws = RunWorkspace(query, cfg, es, tracer)
  ├─ set_workspace(ws)                    # ContextVar，请求级隔离的关键
  ├─ try: ① → ② → ③（见全景图）
  └─ except Exception as exc:
       tracer.log("degrade", "triggered", error=repr(exc))
       return self._degrade(query, repr(exc))
```

**要点**

- 阶段间数据**不走返回值传递上下文**：三阶段共享同一个 `RunWorkspace`（ContextVar），
  工具内部通过 `get_workspace()` 读写 `ws.data`；阶段间只有 `chunks` 列表经参数显式传递。
- 捕获范围是 `Exception` 全集——任何子智能体/工具/模型异常都转降级，绝不向坐席抛错。
- **并发约束**：同一实例不支持并发 `run`（tracer/处理子智能体均为实例级状态）；
  并发服务需为每请求创建新 MainAgent。

---

## 3. 共享基础设施

### 3.1 RunWorkspace（请求级工作区，`shared/workspace.py`）

```
RunWorkspace { query, cfg, es, tracer, stage, data: Dict[str, Any] }
_workspace: ContextVar[RunWorkspace]      # set_workspace() / get_workspace()
```

- `get_workspace()` 未初始化时抛 `RuntimeError`（工具在请求外被误调用的防线）。
- `stage` 字段由每个子智能体进入时设置（`"retrieval"` / `"processing"`），
  供 trace 事件的阶段前缀使用。
- `data` 常用键（全链路共享的“黑板”）：

| 键 | 写入者 | 读取者 |
|---|---|---|
| `intents` | query_understanding | question_rewrite / coarse_recall |
| `rewritten_query`, `is_retry` | question_rewrite | keyword_extraction / coarse_recall |
| `keywords`, `expanded_terms` | keyword_extraction | coarse_recall |
| `chunks` | coarse_recall / ProcessingSubAgent.run / 各处理工具 | 验证器、下游阶段 |
| `last_params`, `last_dsl`, `recall_round` | coarse_recall | trace/调试 |
| `analysis` | analyze_data | LLM 决策参考 |

### 3.2 Tracer（全链路追踪，`shared/tracing.py`）

```
Tracer.trace_id = new_id("trace")         # 形如 trace_6b791f84df54
Tracer.log(stage, event, **payload) → events.append(TraceEvent(ts_ms, stage, event, payload))
Tracer.elapsed_ms() / Tracer.export()     # export() 输出完整 JSON，badcase 可回放
```

trace_id 同时用作 GoalLoop 的 `thread_id` 和 `FinalAnswer.trace_id`——三处同源。

### 3.3 Config（领域配置，`shared/config.py`）

| 配置 | 默认值 | 作用点 |
|---|---|---|
| `max_retrieval_rounds` | 2 | GoalLoop 的 `max_iterations` |
| `recall_size` | 10 | build_dsl 的 size |
| `fuse_top_n` | 8 | RRF 融合保留条数 |
| `top3_score_threshold` | 0.4 | SufficiencyVerifier 得分判据 |
| `min_chunk_count` | 3 | SufficiencyVerifier 数量判据 |
| `budget["retrieval_total"]` | 2000ms | GoalLoop 的 `max_time_seconds`（=2.0s） |
| `stale_days` | 365 | 答案阶段知识过旧标注 |

### 3.4 检索层（`shared/search.py`）

```
build_dsl(params, size)            search.py:18
  ├─ boost 字段白名单过滤：ALLOWED_BOOST_FIELDS = {title, content, keywords}
  ├─ filter 字段白名单过滤：ALLOWED_FILTER_FIELDS = {category, status, region}   ← 安全边界
  └─ 输出固定形态 {"size", "query":{"bool":{"must":[multi_match],"filter":[term...]}}}

rrf_fuse(keyword_hits, vector_hits, k=60, top_n)   search.py:40
  └─ 倒数排名融合：score = Σ 1/(k+rank+1)，最终 chunk.score = round(rrf*30, 4)
     （这就是为什么充分性阈值 0.4 对“双路都命中前列”的片段轻松达标：1/61*2*30≈0.98）

ESClient（抽象）→ MockESClient（离线）
  ├─ keyword_search(dsl)：按 multi_match 词频打分（标题权重×2），应用 term 过滤
  ├─ vector_search(query_text, filters, size)：字符集重合度模拟向量相似度（阈值 0.15）
  └─ 语料 _KB：7 条知识（套餐×3 / 宽带 / 账单 / 投诉 / 下架套餐×1）
```

### 3.5 词表（`shared/lexicon.py`）

`detect_categories`（类目识别）/ `extract_keywords`（无命中时退化为前 8 字符）/
`expand_terms`（同义扩展）。同时被检索工具与降级路径使用。

---

## 4. 阶段①：检索子智能体（GoalLoop 自主循环）

### 4.1 组装调用链

```
RetrievalSubAgent.run(query)                            retrieval/agent.py:37
  ├─ ws = get_workspace(); ws.stage = "retrieval"
  ├─ loop = create_agent(                               uniagent/agents/factory.py:23
  │      model, tools=RETRIEVAL_TOOLS,
  │      features=AgentFeatures(skill=False),           # 检索不注入技能
  │      system_prompt="你是候选知识检索子智能体...",
  │      goal=_RETRIEVAL_GOAL,                          # 目标文本（含重试指引）
  │      verifier=SufficiencyVerifier(),
  │      budget=Budget(BudgetConfig(max_iterations=2, max_time_seconds=2.0)),
  │      name="retrieval_subagent")
  │   │
  │   ├─ feat.resolve_middleware() → 链 = [DanglingToolCall, ToolErrorHandling,
  │   │                                     LoopDetection, TokenUsage]   （无 Skill）
  │   ├─ create_react_agent(model, tools, state_schema=ThreadState, prompt, name)
  │   │      → LangGraph CompiledGraph（内部即"模型节点 ↔ 工具节点"的 ReAct 图）
  │   │      ⚠ factory.py:113 触发 LangGraphDeprecatedSinceV10 弃用警告
  │   ├─ agent._uniagent_middleware = chain             # 猴子补丁挂载（factory.py:124）
  │   └─ goal 非空 → 返回 GoalLoop(agent, goal, verifier, hooks=[ProgressLogHook,
  │                                                     TokenBudgetHook], budget)
  │
  ├─ result = await loop.run(
  │      input_messages=[{"role":"user","content":f"用户问题:{query}"}],
  │      thread_id=tracer.trace_id)
  │
  └─ 结果解释（retrieval/agent.py:57-66）：
       tracer.log("retrieval","loop_result", success, iterations, reason)
       chunks = ws.data.get("chunks", [])
       if not result.success:
         ├─ chunks 为空 且 reason 以 "错误" 开头 → raise RuntimeError → 主智能体降级
         └─ 否则 → tracer.log("retrieval","exit_with_best") → 携最优结果继续
       return chunks
```

> **关键语义**：轮次耗尽（`reason="已达最大迭代次数，目标未完成"`）不算致命错误——
> 只要工作区里有 chunks 就携最优退出；只有“出错且颗粒无收”才上抛触发降级。

### 4.2 工具集（`retrieval/tools.py`）

LLM 只传结构化参数，**永远接触不到 ES DSL**：

| 工具 | 职责 | 工作区写入 |
|---|---|---|
| `query_understanding` | 类目/子意图识别（lexicon.detect_categories） | `intents` |
| `question_rewrite` | 重试轮改写（怎么→如何、补意图词） | `rewritten_query`, `is_retry=True` |
| `keyword_extraction` | 关键词提取 + 同义扩展 | `keywords`, `expanded_terms` |
| `coarse_recall` | 混合召回（见下） | `chunks`, `last_params`, `last_dsl`, `recall_round` |

`coarse_recall` 内部调用链（安全边界所在）：

```
coarse_recall(relax_filters, retrieval_mode)
  ├─ 无 keywords → 直接返回错误观察值（提示先调 keyword_extraction）
  ├─ 过滤条件：仅当 非relax 且 非重试轮 且 intents 中恰有 1 个已知类目时，
  │            加 {category: 类目, status: "在售"}
  ├─ RetrievalParams.from_llm_output({...})    # models.py:46 清洗：
  │     · 列表 ≤10 项、字符串 ≤64 字符
  │     · boost 值 clamp 到 [0.1, 10.0]
  │     · retrieval_mode 限定 keyword|vector|hybrid
  ├─ dsl = build_dsl(params, size=cfg.recall_size)     # 白名单二次过滤
  ├─ khits = es.keyword_search(dsl)                    （mode ∈ keyword/hybrid）
  ├─ vhits = es.vector_search(qtext, filters, size)    （mode ∈ vector/hybrid）
  ├─ fused = rrf_fuse(khits, vhits, top_n=cfg.fuse_top_n)
  ├─ ws.data["chunks"] = fused
  └─ tracer.log(f"retrieval.round{rnd}", "recall", dsl, titles, scores)
```

### 4.3 GoalLoop 每轮迭代算法（`runtime/loop.py:289`）

```
run():
  注入目标：SystemMessage("[目标] 您的任务目标：{goal}...") 置于 messages 首位

  for i in range(budget.max_iterations):          # 本场景=2
    ① Budget.check()：迭代/时间超限 → on_budget_exhausted 通知
                     → LoopResult(success=False, reason=原因) 退出
    ② hooks.on_iteration_start(i, state)
         BREAK → 失败退出；ROLLBACK → 恢复检查点快照继续
    ③ _invoke_agent(state, thread_id)             # loop.py:64，见 4.4
         异常 → hooks.on_error：BREAK→失败退出(reason="错误：{exc}")；
                             RETRY→记录迭代后重来；默认 LoopHook.on_error=BREAK
    ④ budget.record_iteration()
    ⑤ last_checkpoint_state = deepcopy(state)     # 棘轮：进度不丢
    ⑥ hooks.on_iteration_end(i, state, result)
         BREAK → 失败退出（LoopDetection 硬停在此生效）；RETRY → 跳过验证重来
    ⑦ verifier.verify(goal, state)                # 每轮验证（verify_every=1）
         通过 → on_goal_achieved 通知 → LoopResult(success=True, evidence) 退出
         失败 → 注入反馈（见下），进入下一轮

  循环耗尽 → LoopResult(success=False, reason="已达最大迭代次数，目标未完成")
```

**验证失败反馈注入**（这是"自主改写重召"的驱动源）：

```
feedback = HumanMessage("[验证失败] 目标尚未达成。\n依据：{evidence}\n请继续朝目标推进：{goal}")
先删除历史中所有 "[验证失败]" 前缀消息（防膨胀），只追加最新一条
```

### 4.4 `_invoke_agent`：中间件 + ReAct 的执行点（`loop.py:64`）

```
_invoke_agent(state, thread_id)
  ├─ chain = agent._uniagent_middleware
  ├─ state_snapshot = {**state}                   # H1：失败可回滚
  ├─ for mw in chain:        patch = await mw.before_agent(state)   # 正序
  │      异常 → 恢复 snapshot 后上抛
  ├─ result = await agent.ainvoke(state, config={"configurable":{"thread_id"}})
  │      │   ← LangGraph ReAct 图内部：模型节点 →（有tool_calls?）→ 工具节点
  │      │      → 回灌 ToolMessage → 模型节点 → … 直到无 tool_calls 的 AIMessage
  │      └─ 异常 → 依次询问中间件 handle_invoke_error；
  │                ToolErrorHandlingMiddleware 接管（致命异常除外，见 §7）；
  │                无人接管则上抛
  └─ for mw in reversed(chain): patch = await mw.after_agent(result)  # 逆序
```

### 4.5 SufficiencyVerifier（纯规则，`retrieval/sufficiency.py:16`）

```
async verify(goal, state) -> VerificationResult
  ├─ chunks = ws.data["chunks"]
  ├─ rule_top3  = len≥3 且 top3 分数全部 ≥ 0.4
  ├─ rule_count = len ≥ 3
  ├─ 失败 → evidence = "候选数 X < 3; top3 得分未达阈值 0.4: [...]。
  │          请换策略:改写问题/扩展同义词/放宽过滤(relax_filters=true)。"
  │          trace: sufficiency.rule_fail
  │          → VerificationResult(passed=False, layer="rules", confidence=1.0)
  └─ 通过 → trace: sufficiency.passed → VerificationResult(passed=True)
```

验证器实现的是 `uniagent.verification.verifier.Verifier` 协议（结构化鸭子类型，
`async verify(goal, state) -> VerificationResult`），与框架完全解耦。

### 4.6 检索阶段典型消息流（正常链路，1 轮通过）

```
[SystemMessage] [目标] 您的任务目标：为用户问题召回足量...
[HumanMessage]  用户问题:用户想办理流量套餐,如何推荐?
[AIMessage]     先理解问题意图。            tool_calls=[query_understanding]
[ToolMessage]   {"intents": ["套餐"]}
[AIMessage]     提取关键词并做同义扩展。     tool_calls=[keyword_extraction]
[ToolMessage]   {"keywords": ["流量","套餐"], "expanded_terms": [...]}
[AIMessage]     执行混合召回。              tool_calls=[coarse_recall]
[ToolMessage]   {"recalled": 3, "titles": [...], "scores": [0.98,0.96,0.96]}
[AIMessage]     已完成当前阶段任务,结果写入工作区。   ← 验证随后通过，循环结束
```

---

## 5. 阶段②：处理子智能体（裸 ReAct + 技能注入）

### 5.1 组装（构造期，`processing/agent.py:31`）

```
ProcessingSubAgent.__init__(model, tracer, enable_skills)
  └─ self._agent = create_agent(
        model, tools=PROCESSING_TOOLS,
        features=AgentFeatures(skill=enable_skills, goal_loop=False),
        system_prompt=_PROCESSING_PROMPT,          # 含"不要裁剪片段"约束
        name="processing_subagent")
     │
     ├─ enable_skills=True 时，create_agent 还会预加载技能工具（factory.py:86-103）：
     │     final_tools = 7 个处理工具
     │                 + load_skill_reference        # 按需加载技能参考文档
     │                 + validate_taocan_price        # taocan-skill scripts/ 中 @tool 脚本
     ├─ 无 goal、goal_loop=False → 返回裸 CompiledGraph（create_agent 模式1）
     └─ 中间件链 = [Skill, Dangling, ToolError, LoopDetection, TokenUsage]
```

### 5.2 运行（请求期，`processing/agent.py:41`）

```
ProcessingSubAgent.run(query, chunks)
  ├─ ws.stage = "processing"; ws.data["chunks"] = list(chunks)
  ├─ before = [chunk_id...]; cat_hint = "业务类目:套餐"（类目唯一时）
  ├─ state = {"messages": [HumanMessage(
  │       "清洗候选知识,共 3 条。业务类目:套餐。用户问题:{query}")]}
  │
  ├─ for mw in self._agent._uniagent_middleware:      ← 裸 agent 无 GoalLoop，
  │      patch = await mw.before_agent(state)            中间件由这里手工驱动
  │      ★ SkillMiddleware.before_agent 在此执行（见 5.3）
  │
  ├─ try: await self._agent.ainvoke(state)
  │     （ReAct 内部：LLM 依次调用处理工具，工具直接改写 ws.data["chunks"]）
  │   except: tracer.log("processing","agent_error")   ← 吞掉异常，靠保底兜住
  │
  ├─ out = ws.data["chunks"]
  ├─ if not out:                                        # 保底流水线
  │     tracer.log("processing","fallback_pipeline")
  │     ws.data["chunks"] = 原始 chunks
  │     run_fallback_pipeline()    # tools.py:106
  │       = clean → denoise → dedupe → structure → sort（直接调 .func()，不经 LLM）
  └─ tracer.log("processing","snapshot", before, after); return out
```

> 注意与检索阶段的差异：处理阶段**没有 GoalLoop / 验证器**，是"一轮到底"的裸 ReAct；
> 也没有执行 `after_agent` 中间件与 `handle_invoke_error`（异常由自己的 try/except 兜）。

### 5.3 技能注入的完整调用链（SkillMiddleware）

```
SkillMiddleware.before_agent(state)         middleware/builtins/skill_middleware.py:59
  ├─ _ensure_initialized()：get_skill_registry() 为空则跳过（幂等延迟初始化）
  ├─ 幂等检查：messages 中已有 "<!-- SKILL:" 标记 → 跳过（防 GoalLoop 重复注入）
  ├─ 从后往前找最后一条 HumanMessage（跳过 "[验证失败]" 反馈消息）
  ├─ matches = registry.match(content, max_results=1)     skills/registry.py:201
  │     逐技能取触发器最高分：
  │       keyword "套餐" 子串命中 → score = min(0.9, 2/len+0.3) ≈ 0.35 ≥ 0.3 阈值
  │       （正则触发器带 ReDoS 超时保护；intent 类型未实现恒 0）
  ├─ content = registry.activate(match)                   registry.py:259
  │     ├─ SkillLoader.load：SKILL.md 全文 → instruction；when="always" 的参考即时加载
  │     └─ load_skill_scripts：scripts/validate_taocan.py 中的 @tool → script_tools
  ├─ self._injector.activate(content)      # 最多 3 个激活技能，去重、最旧驱逐
  └─ 返回 patch：
       messages += [SystemMessage("<!-- SKILL: taocan-skill -->\n{SKILL.md}\n
                    可按需加载的参考文档（调用 load_skill_reference 获取）：
                    - field_rules.md / examples.md ...")]
       promoted_tools = ["calculator"]（manifest 声明，写入 state）
```

LLM（含离线脚本模型）看到 `SKILL:` 标记与 SKILL.md 第 6 条规则后，会在通用清洗
结束时调用 `apply_business_skill(category="套餐")` 完成字段归一（"月费"→"月费(每月)"）。

### 5.4 处理工具集（`processing/tools.py`）

| 工具 | 行为 | 对 chunks 的影响 |
|---|---|---|
| `analyze_data` | 数量/类目/状态分布报告 | 只读（写 `ws.data["analysis"]`） |
| `clean_data` | 压缩空白字符 | 原地改 content |
| `denoise_data` | 剔除 `status=="下架"` | 过滤 |
| `dedupe_data` | 同 chunk_id 保留最高分 | 过滤 |
| `structure_data` | 正则抽取 `fees_yuan` / `deadlines_hours` | 只增 extra |
| `sort_data` | 按 (score, updated_at) 降序 | 排序 |
| `apply_business_skill` | 类目归一（限定 套餐/宽带/账单/投诉） | 改 content + extra.skill_applied |

---

## 6. 阶段③：答案子智能体（直调 LLM + 确定性锚定）

本阶段**不走 ReAct**：`AnswerSubAgent.run` 内是两次（类）同步 `model.invoke` 直调。

```
AnswerSubAgent.run(query, chunks, trace_id)        answer/agent.py:25
  ├─ materials = select_fragments(query, chunks)   generate.py:49
  │     取前 4 条，同一 doc_id 最多 2 条（依赖上游已排序）
  └─ generate(model, query, materials, cfg, tracer, trace_id)   generate.py:61
```

### 6.1 generate 内部流程

```
① tracer.log("answer","materials", chunk_ids)
② material_text = 每片段一行 <chunk id="kb_0001#p1">内容</chunk>
③ 组织答案：_invoke_json(model, _ANSWER_SYSTEM, "用户问题:...\n知识片段:\n...")
     _ANSWER_SYSTEM 以 "[TASK:answer]" 开头，要求输出严格 JSON：
       {business_explanation, handling_suggestion,
        sentences:[{text, citations:[chunk_id], hard_fact}]}
     _invoke_json = model.invoke([SystemMessage, HumanMessage]) + JSON 容错解析
④ 逐句锚定校验（确定性代码，不交给 LLM 裁量）：
     for sent in sentences:
       real_cites = sent.citations ∩ materials 的 chunk_id 集合
       ├─ 无有效引用 → anchored=False
       └─ 有 → 拼接被引片段原文，再次直调：
              _invoke_json(model, _ANCHOR_SYSTEM "[TASK:anchor_check]",
                           "句子:{text}\n片段:{chunk_text}")
              → anchored = consistent
       if not anchored:
           hard_fact=True  → sent.dropped = True        ← 硬事实零容忍，直接删句
           hard_fact=False → sent.note = "建议核实"
       tracer.log("answer","anchor_check", ...)
⑤ 善后：
     kept = 未删句子；被删句子的文本从 business_explanation/handling_suggestion 中剔除
     cited_ids = kept 句子引用按出现顺序去重
     sources = 对每个被引 chunk 生成 SourceRef，
               updated_at 早于 now-365天 → stale=True（展示时提示"知识可能过旧"）
⑥ 返回 FinalAnswer(trace_id, query, business_explanation, handling_suggestion,
                   sentences=kept, sources)
```

> **硬事实判定**由 LLM 在 [TASK:answer] 输出中声明（`hard_fact` 字段），但**删除动作**
> 是确定性代码执行的——生成与校验严格分离（生成器/评估器分离原则）。

---

## 7. 降级兜底路径

```
任意异常（模型超时 / 工具崩溃 / 检索颗粒无收且出错 / JSON 全坏…）
   │
   ▼
MainAgent.arun 的 except Exception
   ├─ tracer.log("degrade","triggered", error)
   └─ _degrade(query, reason)                         main_agent.py:80
        ├─ params = RetrievalParams(keywords=lexicon.extract_keywords(query),
        │                            retrieval_mode="keyword")
        ├─ hits = es.keyword_search(build_dsl(params, size=5))
        │     （ES 也挂 → hits=[]，仍然返回答案壳）
        ├─ FinalAnswer(business_explanation="(系统降级,以下为原始知识片段,请人工核实)",
        │               handling_suggestion="",
        │               sources=[SourceRef(原始片段)...],
        │               degraded=True)
        └─ tracer.log("degrade","done", reason, hit_count)
```

**异常如何穿透到主智能体**（以 LLM 超时为例，即 main.py 场景4）：

```
ScriptedChatModel._generate 抛 TimeoutError
  → LangGraph ainvoke 上抛
  → _invoke_agent 询问中间件：ToolErrorHandlingMiddleware.handle_invoke_error
       TimeoutError ∈ _FATAL_EXCEPTIONS（超时/连接/系统级）→ 原样重抛   ← 故意不吞
  → GoalLoop 捕获 → hooks.on_error（默认 LoopHook 返回 BREAK）
  → LoopResult(success=False, reason="错误：LLM gateway timeout")
  → RetrievalSubAgent：chunks 为空且 reason 以"错误"开头 → raise RuntimeError
  → MainAgent.arun except → _degrade
```

非致命异常（普通工具错误）则被 ToolErrorHandlingMiddleware 转成
`AIMessage("[错误] 工具执行失败：...")` 回灌对话，让 LLM 下一轮自行决定补救——
不上抛、不降级。

---

## 8. uniagent 框架层深挖

### 8.1 create_agent 的三种组装模式（`agents/factory.py:23`）

| 模式 | 条件 | 返回 | kbagent 使用者 |
|---|---|---|---|
| 1 裸 Agent | 无 goal、无 goal_loop | CompiledGraph（`.ainvoke`） | ProcessingSubAgent |
| 2 TurnLoop | 无 goal 但 `features.goal_loop=True` | TurnLoop（`.run(goal=...)` 可动态升级） | —（未使用） |
| 3 GoalLoop | `goal` 非空（+verifier，缺省退化为 AlwaysPassVerifier 并告警） | GoalLoop（`.run()`） | RetrievalSubAgent |

中间件组装与模式正交：`AgentFeatures.resolve_middleware()` 按固定顺序产出
`[Skill?, Dangling, ToolError, LoopDetection, TokenUsage]`，
`assemble_middleware_chain()` 再按 `@after/@before` 锚点约束插入自定义中间件并做
冲突/环检测（`middleware/chain.py`）。链通过猴子补丁挂在 `agent._uniagent_middleware`
——序列化/deepcopy 会丢失此属性（源码注释已声明的已知限制）。

### 8.2 中间件洋葱模型执行时序

```
请求方向 ──────────────────────────────────────────────────────────▶
before_agent（正序）：
   Skill → Dangling → ToolError → LoopDetection → TokenUsage
                                                ┌─ agent.ainvoke（LangGraph ReAct）
after_agent（逆序）：                            │
   TokenUsage → LoopDetection → ToolError → Dangling → Skill
异常路径：ainvoke 抛错 → 顺序询问 handle_invoke_error，第一个接管者生效
```

各内置中间件职责：

| 中间件 | 层 | 行为 |
|---|---|---|
| SkillMiddleware | before | 触发器匹配→注入 `<!-- SKILL: -->` SystemMessage（幂等） |
| DanglingToolCallMiddleware | before | 为无 ToolMessage 对应的 tool_calls 补合成占位，防 LLM API 报错 |
| ToolErrorHandlingMiddleware | 异常 | 非致命错误转消息回灌；致命错误重抛触发上层降级 |
| LoopDetectionMiddleware | before + 循环钩子 | 相同 tool_calls 签名重复 ≥3 次：先注入警告（软），再经 `_LoopDetectionHook.on_iteration_end` 发 BREAK（硬） |
| TokenUsageMiddleware | after | 增量统计 AIMessage 用量 → `state["token_usage"]`，可同步到 Budget |

### 8.3 Budget / Hook / Signal

```
Budget(BudgetConfig(max_iterations, max_tokens, max_time_seconds))   runtime/budget.py
  check() 在每轮迭代开始时调用（锁内读计数，无 TOCTOU），任一超限 → (BREAK, 原因)
  record_iteration()/record_tokens() 线程安全

LoopHook（迭代级生命周期，区别于节点级中间件）        runtime/hooks.py
  on_iteration_start / on_iteration_end / on_goal_achieved /
  on_budget_exhausted / on_error（默认 BREAK）
  内置：ProgressLogHook（进度日志）、TokenBudgetHook（把累计 token 绝对值写回 Budget）

LoopSignal：CONTINUE / BREAK / RETRY / ROLLBACK
HookResponse：signal + message + state_patch（H2：patch 会被循环引擎合并进 state）
LoopResult：success / iterations / reason / final_state / evidence
```

### 8.4 技能子系统全链路

```
注册期（进程启动）:
  register_skill_directory("skills")
    → SkillRegistry.scan：子目录 metadata.json → SkillManifest（触发器/参考/脚本/
      promoted_tools），正则触发器预编译；目录级幂等（渐进式加载）

激活期（每次 before_agent）:
  SkillMiddleware → registry.match（触发器打分，取最高）→ registry.activate
    → SkillLoader.load（SKILL.md + when="always" 参考；路径穿越防护）
    → load_skill_scripts（importlib 动态加载 @tool 脚本工具）
    → SkillInjector.activate（≤3 个激活技能）
    → SystemMessage 注入 messages（含 on_demand 参考清单）

按需披露期（LLM 主动）:
  load_skill_reference(skill_name, filename) 工具
    → registry.match_by_name → SkillLoader.load_reference（带缓存）
```

技能目录约定：`metadata.json` / `SKILL.md` / `references/` / `templates/` / `scripts/`。
本仓库的 `skills/taocan-skill` 是唯一实例：4 条触发器（套餐/流量/资费关键词 + 资费正则）、
2 份 on_demand 参考、1 个脚本工具 `validate_taocan_price`。

### 8.5 其余框架组件（本链路未直接走，但属框架能力）

- `state/thread_state.py`：ThreadState 扩展 LangGraph AgentState，增加
  `artifacts / promoted_tools / todos / summary / token_usage / current_task /
  loop_iteration / verification_result` 字段（均带合并 reducer）。
- `agents/config_factory.py`：`create_agent_from_config`（AppConfig YAML 驱动：
  模型解析 → 工具加载 → 中间件解析 → 预算 → 技能初始化 → create_agent）。
  kbagent 只用了其中的 `register_skill_directory` / `get_skill_registry`。
- `models/factory.py`：ModelFactory 从 ModelConfig 构建/缓存模型
  （api_key/base_url/timeout/max_retries/extra_headers 注入，kwargs 最高优先级）。
- `verification/builtins/`：AlwaysPassVerifier / LLMVerifier / CompositeVerifier。
- `runtime/`：checkpointer、context、protocols 等支撑模块。

---

## 9. 离线模式：ScriptedChatModel 的行为脚本

`kbagent/scripted_model.py` 是 `BaseChatModel` 的离线实现，让整套
ReAct/反馈注入/技能机制**不接真实 LLM 也能真实跑通**。它根据消息历史做确定性决策：

```
_generate(messages)
  ├─ 含 "[TASK:answer]"      → _scripted_answer：
  │      取前 3 个 <chunk>，各取第一句（≤60字）作事实句并引用对应 chunk；
  │      含 元/资费/条件/生效 → hard_fact=True；另附一句软性办理建议
  ├─ 含 "[TASK:anchor_check]" → _scripted_anchor：
  │      句子与片段字符重合 ≥ max(3, 30%) → consistent=true
  └─ 否则（ReAct 模式，按绑定工具名+文本区分阶段）：
       检索阶段（绑定含 coarse_recall）:
         非重试轮: query_understanding → keyword_extraction → coarse_recall
         重试轮（存在"[验证失败]"消息，且只统计其后的调用）:
                   question_rewrite → keyword_extraction → coarse_recall(relax_filters=True)
       处理阶段（文本含"清洗候选知识"）:
         analyze → clean → denoise → dedupe → structure → sort
         若消息流含 "SKILL:" → 追加 apply_business_skill(category=从"业务类目:"解析)
       工具全部完成 → AIMessage("已完成当前阶段任务,结果写入工作区。")（无 tool_calls，
       LangGraph ReAct 图据此收尾）
```

`bind_tools` 仅记录工具名（`bound_tool_names`），这也是它区分检索/处理阶段的手段之一。
生产接入时把 `model=` 换成 `ChatOpenAI` 等真实模型即可，链路不变。

---

## 10. 端到端时序图

### 10.1 正常链路（1 轮验证通过 + 技能触发）

```
User    MainAgent      Retrieval/GlLp   Scripted   检索Tools   Verifier   Processing   Answer
 │  run(q) │                │              │           │          │           │          │
 │────────▶│ arun: Tracer/Workspace(ContextVar)        │          │           │          │
 │         │──① run(q)─────▶│ 注入[目标]    │           │          │           │          │
 │         │                │──invoke──────▶│ query_understanding▶│          │           │
 │         │                │               │ keyword_extraction ▶│          │           │
 │         │                │               │ coarse_recall──────▶│(build_dsl/ES/RRF)     │
 │         │                │               │◀─chunks写入ws.data──│          │           │
 │         │                │──verify──────────────────────────────▶ top3/count 通过      │
 │         │                │◀─LoopResult(success=True, 1轮)──────│          │           │
 │         │◀─chunks────────│               │           │          │           │          │
 │         │──② run(q,chunks)──────────────────────────────────────▶ Skill注入 │          │
 │         │                │               │  analyze→…→sort → apply_business_skill      │
 │         │◀─processed（空则保底流水线）────────────────────────────│          │          │
 │         │──③ run(q,proc,trace_id)──────────────────────────────────────────▶ select(4) │
 │         │                │               │◀─[TASK:answer] 直调──│          │ 组答案    │
 │         │                │               │◀─[TASK:anchor_check]×N 逐句锚定 │          │
 │         │◀─FinalAnswer（degraded=False, elapsed_ms）────────────│          │          │
 │◀─render()│               │               │           │          │           │          │
```

### 10.2 重试链路（冷门问题，2 轮耗尽携最优退出）

```
轮1: query_understanding → keyword_extraction → coarse_recall
     → verify 失败（候选不足/分数低）
     → GoalLoop 注入 HumanMessage "[验证失败] …请换策略…"
轮2: Scripted 检测到 "[验证失败]" → question_rewrite → keyword_extraction
     → coarse_recall(relax_filters=True)
     → verify 仍失败 → 但 max_iterations=2 耗尽
     → LoopResult(success=False, "已达最大迭代次数，目标未完成")
     → chunks 非空 → "exit_with_best" 携最优继续（不上抛）
     → 后续阶段照常执行
```

---

## 11. 数据结构全程流转

以 `chunk_id` 为主线的溯源链（关键设计约束：全链路透传）：

```
MockES._KB 原始行
   │ _to_chunk()
   ▼
Chunk{chunk_id="kb_0001#p1", score=ES打分}
   │ keyword_search / vector_search
   ▼
Chunk{score := RRF 融合分}                      ← coarse_recall 写入 ws.data["chunks"]
   │ RetrievalSubAgent 返回（显式参数传递）
   ▼
Chunk{content 清洗, extra + fees_yuan/status 过滤}   ← 处理阶段原地改写
   │ ProcessingSubAgent 返回
   ▼
select_fragments → materials（top4，同文档≤2）
   │ [TASK:answer] 中作为 <chunk id="..."> 上下文
   ▼
AnswerSentence{citations=[chunk_id]}            ← LLM 生成引用
   │ 锚定校验过滤
   ▼
SourceRef{chunk_id, doc_title, snippet, updated_at, stale}  ← FinalAnswer.sources
   │ render()
   ▼
坐席可见文本："1. 5G畅享套餐资费说明 [kb_0001#p1] 更新于 2026-06-10 ..."
```

核心数据结构一览（`shared/models.py`）：

| 结构 | 用途 |
|---|---|
| `Chunk` | 知识片段（内容+元数据+分数+溯源） |
| `RetrievalParams` | LLM 与 DSL 之间的结构化契约（`from_llm_output` 清洗） |
| `SufficiencyResult` / `RetrievalRound` | 验证判据与轮次记录 |
| `AnswerSentence` | 答案句（引用/硬事实/锚定/删除标记） |
| `SourceRef` | 最终展示的知识来源（含过旧标记） |
| `FinalAnswer` | 顶层返回（含 `render()` 坐席视图与 `degraded` 标志） |

---

## 12. Trace 事件清单

| stage | event | 触发点 |
|---|---|---|
| run | start | MainAgent.arun 开始 |
| retrieval.round{N} | recall | coarse_recall 每次召回（含 dsl/titles/scores） |
| retrieval | sufficiency.rule_fail / sufficiency.passed | 验证器判定 |
| retrieval | loop_result | GoalLoop 结束（success/iterations/reason） |
| retrieval | exit_with_best | 轮次耗尽但携结果退出 |
| processing | agent_error | ReAct 调用异常（被吞） |
| processing | fallback_pipeline | 产出为空触发保底 |
| processing | snapshot | 处理前后 chunk_id 对比 |
| answer | materials | 进入答案阶段的片段清单 |
| answer | anchor_check | 每句锚定结果（含 dropped） |
| finalize | done | 正常结束（含总耗时） |
| degrade | triggered / done | 降级进入/完成 |

`tracer.export()` 可导出完整 JSON 用于 badcase 回放。

---

## 13. 注意事项与已知差异

1. **文档与代码差异——缓存**：`main.py` 场景3 注释与 CLAUDE.md 流程图提到
   "缓存快速通道"，但当前 `main_agent.py` **没有实现缓存**；场景3 实际走完整链路。
   若需要缓存，应在 `MainAgent.arun` 入口（Workspace 创建前）加 query 归一化缓存层。
2. **LangGraph 弃用警告**：`factory.py:113` 的 `create_react_agent` 在 LangGraph V1.0
   起弃用（V2.0 移除），需迁移为 `from langchain.agents import create_agent`。
3. **处理子智能体的中间件只执行了 before**：裸 Agent 模式下 `after_agent` 与
   `handle_invoke_error` 不会被触发（无循环引擎驱动），异常由
   `ProcessingSubAgent.run` 自己的 try/except 兜底。
4. **复用的处理子智能体持有中间件实例状态**：`TokenUsageMiddleware._last_msg_count`、
   `LoopDetectionMiddleware` 计数器等跨请求留存。当前离线模型无 token 用量、且每请求
   消息列表重建，影响有限；但并发/复用场景下应注意（这也是"每请求新建 MainAgent"
   约束的原因之一）。
5. **中间件链挂载方式是猴子补丁**（`agent._uniagent_middleware`）：对 agent 做
   序列化或 deepcopy 会丢链。
6. **验证器异常的语义**：`verifier.verify` 抛异常时 GoalLoop 仅告警并消耗一个迭代
   （`continue`），不中断循环。
7. **并发**：同一 `MainAgent` 实例禁止并发 `run`；已在事件循环中请用 `arun()`
   （`run()` 内部是 `asyncio.run`，嵌套事件循环会冲突）。
8. **ContextVar 依赖同一执行上下文**：所有工具通过 `get_workspace()` 访问请求上下文，
   跨线程池投递任务时需手动拷贝 contextvars。

---

## 附：函数级调用图（精简）

```
MainAgent.run                                            kbagent/main_agent.py:47
└─ asyncio.run(arun)                                     kbagent/main_agent.py:50
   ├─ Tracer / RunWorkspace / set_workspace              shared/tracing,workspace
   ├─ ① RetrievalSubAgent.run                            kbagent/retrieval/agent.py:37
   │  └─ uniagent.create_agent → GoalLoop                uniagent/agents/factory.py:23
   │     └─ GoalLoop.run                                  uniagent/runtime/loop.py:289
   │        └─ 每轮:
   │           ├─ Budget.check                            runtime/budget.py:54
   │           ├─ _run_hooks(on_iteration_start/end)      runtime/loop.py:112
   │           ├─ _invoke_agent                           runtime/loop.py:64
   │           │  ├─ mw.before_agent × N（正序）
   │           │  ├─ CompiledGraph.ainvoke（LangGraph ReAct）
   │           │  │  └─ 工具: query_understanding / question_rewrite /
   │           │  │          keyword_extraction / coarse_recall
   │           │  │     └─ RetrievalParams.from_llm_output → build_dsl
   │           │  │        → ESClient.keyword_search + vector_search → rrf_fuse
   │           │  └─ mw.after_agent × N（逆序）
   │           └─ SufficiencyVerifier.verify              kbagent/retrieval/sufficiency.py:19
   ├─ ② ProcessingSubAgent.run                           kbagent/processing/agent.py:41
   │  ├─ mw.before_agent × N（手工，SkillMiddleware 注入技能）
   │  ├─ CompiledGraph.ainvoke（7+2 个工具，LLM 自主编排）
   │  └─ 空产出 → run_fallback_pipeline                   kbagent/processing/tools.py:106
   ├─ ③ AnswerSubAgent.run                               kbagent/answer/agent.py:25
   │  ├─ select_fragments                                 kbagent/answer/generate.py:49
   │  └─ generate                                         kbagent/answer/generate.py:61
   │     ├─ model.invoke([TASK:answer])
   │     ├─ model.invoke([TASK:anchor_check]) × 句数
   │     └─ FinalAnswer（含 SourceRef 过旧判定）
   └─ except → _degrade                                  kbagent/main_agent.py:80
      └─ lexicon.extract_keywords → build_dsl → keyword_search → FinalAnswer(degraded)
```
