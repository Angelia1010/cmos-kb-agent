# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KB-Agent v2.0.2 — 10086坐席知识库智能体。基于裁剪版uniagent框架，采用"编排好的主智能体 + 三个自主规划的子智能体"架构。Python >= 3.12，核心依赖：LangGraph、LangChain-Core、Pydantic v2。

## Commands

```bash
# 安装依赖
pip install langgraph langchain-core pydantic pyyaml

# 运行演示（离线，ScriptedChatModel + MockES）
PYTHONPATH=src python main.py

# 测试
python tests.py              # 8项关键约束测试
python tests_subagents.py    # 4项子智能体测试
python tests_bugfix.py       # 8项修复回归测试
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

- `src/uniagent/` — 裁剪版通用框架（agents工厂、GoalLoop/TurnLoop循环引擎、中间件系统、技能子系统、工具注册表、充分性验证协议、ThreadState、配置热重载）
- `src/kbagent/` — 业务层实现
  - `main_agent.py` — 主智能体，三阶段固定编排入口
  - `subagents.py` — 三个子智能体定义（检索/处理/答案）
  - `tools.py` — 11个工具（4检索 + 7处理）
  - `answer.py` — 答案生成 + 逐句锚定校验（硬事实删句，软性表述标注）
  - `sufficiency.py` — 充分性验证器（规则先行 + LLM补充）
  - `search.py` — ES检索层抽象（BM25 + 向量 + RRF融合）
  - `llm_bridge.py` — LLM接口适配（自动包装BaseChatModel为judge/small_json/large_json）
  - `models.py` — 核心数据结构（Chunk、RetrievalParams、FinalAnswer）
  - `scripted_model.py` — 离线Mock模型，用于测试和演示
  - `cache.py` / `workspace.py` / `tracing.py` / `config.py`
- `skills/` — 业务技能包（taocan/kuandai/zhangdan/tousu），零代码可扩展
- `main.py` — 演示入口（4个场景）

### 关键设计约束

1. **DSL字段白名单**：LLM永不接触原始ES DSL，只输出结构化RetrievalParams，code侧在build_dsl()中强制ALLOWED_FILTER_FIELDS白名单
2. **GoalLoop护栏**：Budget(max_iterations=2, max_time_seconds=2.0)，验证失败→注入负例反馈→自主改写重召→轮次耗尽携最优退出
3. **硬事实零容忍**：资费/办理条件等hard_fact=True的句子若锚定失败直接删除，不让不准确信息流出
4. **降级兜底**：任一环节异常/超时→坐席永远有东西可看
5. **chunk_id全链路透传**：从召回到最终答案的知识溯源

### 中间件（洋葱模型）

DanglingToolCallMiddleware → ToolErrorHandlingMiddleware → LoopDetectionMiddleware → SkillMiddleware → TokenUsageMiddleware

### 并发注意

MainAgent实例持有tracer，不支持同一实例并发run。并发服务需为每个请求创建新MainAgent实例。已在事件循环中时用`arun()`而非`run()`。
