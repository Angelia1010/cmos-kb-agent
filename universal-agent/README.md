# 10086 坐席知识库智能体 —— 编排好的主智能体 + 三个自主规划的子智能体

基于裁剪版 uniagent v2.0.2。核心定位:**主智能体编排固定,子智能体自主规划,护栏确定性兜底。**

## 快速开始

```bash
pip install langgraph langchain-core pydantic pyyaml

# kbagent 业务层演示（4个场景，ScriptedChatModel + MockES，完全离线）
PYTHONPATH=src python main.py

# uniagent 框架端到端演示（7个场景，不依赖 kbagent，可单独 debug）
PYTHONPATH=src python -m unittest test_uniagent_e2e -v

# kbagent 整体集成测试（7个测试类，108项）
PYTHONPATH=src python -m unittest test_kbagent_e2e -v

# kbagent 子智能体单元测试（检索34项/处理33项/答案41项）
PYTHONPATH=src python -m unittest discover tests -v
```

## 架构

```
用户Query / 坐席问题
   │
   ▼
MainAgent 主智能体(编排好的,零 LLM 决策)          kbagent/main_agent.py
   │
   ├─ ① RetrievalSubAgent 检索候选知识子智能体(自主规划)
   │     ReAct: LLM 面对 Tool List 自主编排
   │       query_understanding / question_rewrite / keyword_extraction / coarse_recall
   │     护栏(不交给 LLM):
   │       GoalLoop + Budget(max_iterations=2, max_time_seconds) ← 轮次/延迟硬上限
   │       SufficiencyVerifier(纯规则:top3得分+数量下限)         ← 充分性检验
   │       验证失败 → GoalLoop 注入负例反馈 → 子智能体重新规划(改写/放宽)
   │       coarse_recall 内部: 结构化参数清洗 + DSL 字段白名单     ← LLM 永不接触 DSL
   │
   ├─ ② ProcessingSubAgent 数据处理子智能体(自主规划)
   │     ReAct: 自主决定清洗工具取舍与顺序
   │       analyze/clean/denoise/dedupe/structure/sort/apply_business_skill
   │     业务个性化: skills/ 技能包关键词触发,
   │       SkillMiddleware 把归一规则注入提示 → 子智能体套用对应 skill
   │     护栏: 无产出时确定性保底流水线;"不裁剪片段"写入工具约束
   │
   ├─ ③ AnswerSubAgent 答案生成子智能体(自主组织)
   │     LLM 自主取舍素材、组织"业务说明+办理建议"、生成内联引用
   │     护栏: 逐句锚定校验为确定性代码,硬事实锚定失败直接删句
   │
   └─ 任一环节异常/超时 → 降级: 原始query单轮检索返回原文,坐席永远有东西可看
   ▼
最终答案: 业务说明 + 办理建议 + 知识来源(chunk_id 全链路透传,可点开核实)
```

## 自主性与护栏的边界(设计要点)

子智能体的"自主规划"是有边界的自主:**流程内自主,边界处确定**。
LLM 决定"怎么做"(工具顺序、参数、素材取舍);代码决定"何时停、什么算够、
什么不许出"(轮次上限、充分性判据、DSL 白名单、锚定删句、保底流水线)。
这条边界是坐席场景能上生产的前提,任何放宽都应先过评测集。

## uniagent 保留/删除

保留:agents(ReAct 工厂)、**models(ModelFactory + 增强 ModelConfig)**、
**skills 技能子系统**、middleware(技能注入/工具异常/死循环检测/孤立调用修补/用量统计)、
runtime(GoalLoop/Budget)、verification、tools/registry、state、config ——
自主规划子智能体的完整底座。

删除:sandbox、MCP 延迟工具发现、澄清工具、WIP/FeatureList/日志状态机、
命令验证器、摘要中间件。详细理由与框架兼容性修复见 ADAPTATION.md。

## 代码地图

```
src/uniagent/            裁剪版通用框架(无业务概念)
├── agents/              ReAct Agent 工厂(create_agent / create_agent_from_config)
├── models/              模型工厂子包
│   └── factory.py       ModelFactory(build/get/invalidate) + build_model/get_model 便捷函数
├── middleware/          洋葱模型中间件系统
├── runtime/             GoalLoop / TurnLoop / Budget / LoopHook
├── skills/              技能子系统(SkillRegistry/Manifest/Loader/ScriptLoader/Tools)
├── verification/        Verifier 协议 + LLMVerifier / AlwaysPassVerifier
├── state/               ThreadState / reducers / backend
└── config/              AppConfig YAML 热重载；ModelConfig(含 api_key/base_url/timeout/…)

src/kbagent/
├── main_agent.py        主智能体(编排好的): 三阶段固定编排 + 降级/trace
├── scripted_model.py    离线 BaseChatModel(生产换 ChatOpenAI;处理 ReAct/answer/anchor)
├── shared/              跨子智能体共享模块
│   ├── models.py        Chunk / RetrievalParams / FinalAnswer 等核心数据结构
│   ├── search.py        build_dsl / rrf_fuse / ESClient / MockESClient
│   ├── config.py        Config(阈值/预算/溯源天数)
│   ├── workspace.py     ContextVar 请求级工作区(Chunk 实体不走 prompt)
│   ├── tracing.py       TraceEvent / Tracer(全链路 trace)
│   └── lexicon.py       关键词提取 / 意图识别 / 同义扩展
├── retrieval/           检索子智能体
│   ├── agent.py         RetrievalSubAgent(GoalLoop 护栏)
│   ├── sufficiency.py   SufficiencyVerifier(纯规则:top3得分+数量下限)
│   └── tools.py         4个工具:query_understanding/question_rewrite/keyword_extraction/coarse_recall
├── processing/          处理子智能体
│   ├── agent.py         ProcessingSubAgent(SkillMiddleware+保底流水线)
│   └── tools.py         7个工具:analyze/clean/denoise/dedupe/structure/sort/apply_business_skill
└── answer/              答案子智能体
    ├── agent.py         AnswerSubAgent(select_fragments + generate)
    └── generate.py      答案生成/逐句锚定校验/stale检测(model.invoke 标准调用)

skills/                  业务技能包(当前: taocan-skill)
                         新增业务类目 = 加一个目录(metadata.json 触发词 + SKILL.md 归一规则)
                         技能包结构: metadata.json / SKILL.md / references/ / scripts/

tests/                   kbagent 子智能体单元测试
├── test_retrieval.py    34项: DSL白名单/检索工具/充分性验证/RetrievalSubAgent
├── test_processing.py   33项: 7个处理工具/保底流水线/ProcessingSubAgent
└── test_answer.py       41项: 片段精选/JSON解析/锚定校验/AnswerSubAgent/渲染

test_kbagent_e2e.py      kbagent 整体集成测试(7个测试类,108项)
test_uniagent_e2e.py     uniagent 框架端到端演示(7个场景,含 Skill 系统全链路)
main.py                  kbagent 演示入口(4个场景)
config.example.yaml      配置模板(含新增 api_key/base_url/timeout/max_retries 等字段示例)
```

## 变更记录

### kbagent 重构(最新)

- **删除缓存层**：移除 `cache.py` / `AnswerCache`，链路更透明，无请求间状态干扰
- **删除多模型协同**：移除 `llm_bridge.py` / LLMBridge / judge/small_json/large_json；
  答案生成改用标准 `model.invoke([SystemMessage, HumanMessage])`
- **目录重构**：每个子智能体独立目录(retrieval/ / processing/ / answer/ / shared/)
- **SufficiencyVerifier 纯规则化**：移除 LLM intent coverage 判断，只保留 top3 得分 + 数量规则
- **测试体系**：新增 `tests/` 子智能体单元测试(108 项)+ `test_kbagent_e2e.py` 整体集成

### v2.0.2 (uniagent 框架增强)

- **models/ 子包**：ModelFactory(带缓存, 线程安全) + build_model/get_model 便捷函数
- **ModelConfig 增强**：api_key / base_url / timeout / max_retries / extra_headers 五个新字段，向后兼容
- **Skill 脚本工具**：skills/scripts/ 目录下 @tool 函数自动发现并注入 Agent 工具集
- **兼容性修复**：loop.py logger 恢复、middleware 执行链修复、循环依赖延迟导入

已知限制(设计取舍,文档化而非修复):
- 同一 MainAgent 实例不支持并发 run(tracer/工作区按请求隔离依赖"每请求一实例"或外层锁);
- 已在事件循环中时用 arun(),run() 的 asyncio.run 会与现有循环冲突;
- 处理子智能体的中间件只在外层调用时执行一次,ReAct 内部步骤依赖 langgraph
  自身的 recursion_limit 防死循环。

## 生产接入

1. **LLM**:在 `config.yaml` 的 `models` 段填写 `api_key` / `base_url` / `timeout`
   等字段，或传入 `ModelFactory.build(config)` 手动实例化；
2. **ES**:实现 `kbagent.search.ESClient`(keyword_search 执行白名单 DSL,
   vector_search 走 ES 8.x kNN);
3. **技能包**:向 `skills/` 加目录即接入新业务类目,零代码;
4. **缓存**:换 Redis + embedding 相似度;`Tracer.export()` 落日志管道;
5. **上线前**:用评测集(300~500 条真实坐席 query)校准 `config.py` 判据阈值,
   并实测子智能体自主规划带来的额外 LLM 调用对延迟预算的占用。
