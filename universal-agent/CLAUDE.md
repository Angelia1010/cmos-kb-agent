# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KB-Agent v2.0.2 — 10086坐席知识库智能体。基于裁剪版uniagent框架，采用"编排好的主智能体 + 三个自主规划的子智能体"架构。Python >= 3.12，核心依赖：LangGraph、LangChain-Core、Pydantic v2。

## Commands

```bash
# 安装依赖
pip install langgraph langchain-core pydantic pyyaml

# 运行演示（kbagent 业务层，离线 ScriptedChatModel + MockES）
PYTHONPATH=src python main.py

# uniagent 框架端到端演示/测试（7个场景，不依赖 kbagent）
PYTHONPATH=src python -m unittest test_uniagent_e2e -v

# kbagent 整体集成测试（7个测试类，含正常/重试/降级/透传/隔离）
PYTHONPATH=src python -m unittest test_kbagent_e2e -v

# kbagent 子智能体单元测试（检索/处理/答案）
PYTHONPATH=src python -m unittest discover tests -v

# 一次运行所有 kbagent 测试
PYTHONPATH=src python -m unittest discover -s . -p "test_kbagent*.py" -v
```

无lint/格式化工具配置。测试使用标准库unittest，无pytest。

## Architecture

### 执行流程（主智能体固定编排，零LLM决策）

```
用户Query → 缓存快速通道(命中直接返回)
  → ① RetrievalSubAgent (自主规划, GoalLoop护栏, max 2轮)
  → ② ProcessingSubAgent (自主规划, SkillMiddleware注入业务技能包)
  → ③ AnswerSubAgent (LLM组织答案, 确定性逐句锚定校验)
  → 任一异常/超时 → 降级：原始query单轮检索返回原文
```

### 代码布局

- `src/uniagent/` — 裁剪版通用框架
  - `agents/` — ReAct Agent 工厂（create_agent / create_agent_from_config）
  - `models/` — 模型工厂子包（ModelFactory / build_model / get_model）；ModelConfig 支持 api_key / base_url / timeout / max_retries / extra_headers
  - `middleware/` — 洋葱模型中间件（SkillMiddleware / ToolErrorHandling / LoopDetection / DanglingToolCall / TokenUsage）
  - `runtime/` — GoalLoop / TurnLoop 循环引擎、Budget、LoopHook
  - `skills/` — 技能子系统（SkillRegistry / SkillManifest / load_skill_scripts / load_skill_reference）
  - `verification/` — Verifier 协议 + LLMVerifier / AlwaysPassVerifier
  - `state/` — ThreadState / reducers / backend
  - `config/` — AppConfig YAML 热重载、ModelConfig、SkillConfig 等子配置
- `src/kbagent/` — 业务层实现（每个子智能体独立目录）
  - `main_agent.py` — 主智能体，三阶段固定编排入口（缓存快速通道 / 降级兜底 / 全链路trace / judge接口桥接）
  - `scripted_model.py` — 离线Mock模型，处理 ReAct / [TASK:answer] / [TASK:anchor_check] / [TASK:rerank_*]
  - `shared/` — 跨子智能体共享的基础设施（不放子智能体实现）
    - `models.py` — 核心数据结构（Chunk、RetrievalParams、FinalAnswer）
    - `search.py` — ES检索层抽象（build_dsl / rrf_fuse / MockESClient / ProduceESClient / merged_to_chunks）
    - `config.py` — 领域配置（阈值/预算/溯源天数/缓存阈值）
    - `cache.py` — 高频问题缓存（快速通道）
    - `workspace.py` — ContextVar 请求级工作区
    - `tracing.py` — 全链路 trace（TraceEvent / Tracer）
    - `lexicon.py` — 关键词提取 / 意图识别 / 同义扩展
    - `lingxi_provider.py` — 灵犀网关 SSL 策略模型类（生产大模型统一入口）
    - `llm_bridge.py` — 把普通 BaseChatModel 适配出 judge 接口（检索充分性判据用）
    - `tools.py` — 无 Workspace 依赖的共享知识工具（SHARED_KNOWLEDGE_TOOLS）
    - `knowledge_processing/` — 知识级处理库（规范化/分析/适用性过滤/Markdown/重排支撑）
  - `retrieval/` — 检索子智能体
    - `agent.py` — RetrievalSubAgent（GoalLoop 护栏）
    - `sufficiency.py` — SufficiencyVerifier（规则先行 + LLM意图覆盖）
    - `tools.py` — 5个检索工具（query_understanding / question_rewrite / keyword_extraction / coarse_recall / intergrate_all）
  - `processing/` — 处理子智能体
    - `agent.py` — ProcessingSubAgent（SkillMiddleware + 保底流水线）+ KnowledgeProcessingOrchestrator（知识级固定流水线，供 processing_service）
    - `tools.py` — 7个Chunk处理工具（analyze/clean/denoise/dedupe/structure/sort/apply_business_skill）+ 知识级工具工厂
    - `rerank.py` / `output.py` / `prompts.py` — 两阶段重排、Top3→Chunk适配、重排Prompt
  - `answer/` — 答案子智能体
    - `agent.py` — AnswerSubAgent（select_fragments + generate）
    - `generate.py` — 答案生成 + 逐句锚定校验（model.invoke 标准调用，任意BaseChatModel可直连）
- `skills/` — 业务技能包（当前：taocan-skill），零代码可扩展
- `tests/` — kbagent 子智能体/服务单元测试（共202项）
  - `test_retrieval.py` — DSL白名单 / 检索工具 / 充分性验证 / RetrievalSubAgent
  - `test_processing.py` — 7个处理工具 / 保底流水线 / ProcessingSubAgent
  - `test_answer.py` — 片段精选 / JSON解析 / 锚定校验 / AnswerSubAgent / 渲染
  - `test_knowledge_processing.py` / `test_processing_chunk_output.py` / `test_shared_knowledge_tools.py` — 知识级处理链路
  - `test_processing_service.py` / `test_processing_demo.py` — processing 服务与演示
  - `test_state_redis.py` — Redis 状态后端（需安装 redis 包）
- `test_kbagent_e2e.py` — kbagent 整体集成测试（40项）
- `test_uniagent_e2e.py` — uniagent 框架端到端演示（7个场景，含 Skill 系统全链路）
- `main.py` — kbagent 离线演示入口（4个场景）
- `scripts/run_e2e_real_model.py` — 真实大模型端到端验证（生产网络内执行）

### 关键设计约束

1. **DSL字段白名单**：LLM永不接触原始ES DSL，只输出结构化RetrievalParams，code侧在build_dsl()中强制ALLOWED_FILTER_FIELDS白名单
2. **GoalLoop护栏**：Budget(max_iterations=2, max_time_seconds=2.0)，验证失败→注入负例反馈→自主改写重召→轮次耗尽携最优退出
3. **硬事实零容忍**：资费/办理条件等hard_fact=True的句子若锚定失败直接删除，不让不准确信息流出
4. **降级兜底**：任一环节异常/超时→坐席永远有东西可看
5. **chunk_id全链路透传**：从召回到最终答案的知识溯源

### 中间件（洋葱模型）

SkillMiddleware → DanglingToolCallMiddleware → ToolErrorHandlingMiddleware → LoopDetectionMiddleware → TokenUsageMiddleware

### 并发注意

MainAgent实例持有tracer，不支持同一实例并发run。并发服务需为每个请求创建新MainAgent实例。已在事件循环中时用`arun()`而非`run()`。
