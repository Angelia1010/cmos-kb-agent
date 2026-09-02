  ## 实现文件

  - 新增 universal-agent/scripts/run_processing_demo.py:516
  - 新增 universal-agent/tests/test_processing_demo.py:17
  - 扩充现有确定性 Mock universal-agent/tests/processing_mock_data.py:5

  Mock 函数现在是：

  make_top100_candidates(count: int = 100)

  同样参数始终生成相同数据，无随机数，并保留原有 42、73、99 三条高相关候选。新增固定索引代表样例包括地区 except、HTML 表格、结构化列表、
  受众注解、单位、下架、非法时间、过期、地区/渠道不匹配和空内容。

  ## 真实入口与模型

  Demo 只调用一次真实入口：

  universal-agent/src/kbagent/processing/orchestrator.py:11

  真实执行顺序仍是：

  analyze → filter → build_markdown → rerank

  模型使用现有：

  universal-agent/src/kbagent/scripted_model.py:45

  Demo 通过 universal-agent/scripts/run_processing_demo.py:162 包装捕获实际 Prompt。normal 模式直接委托给真实 ScriptedChatModel；
  timeout、invalid_json、insufficient_results 只在包装层模拟故障。

  Demo 完全离线：

  - 不访问网络
  - 不依赖 Redis
  - 不读取 model_config.yaml
  - 不读取或输出 API Key
  - 不调用公司真实模型
  - 不消耗模型额度

  ## Windows PowerShell 命令

  从以下目录运行：

  Set-Location "G:\AA_中移在线\AAA知识库\codes\0827\cmos-kb-agent\universal-agent"
  $env:PYTHONPATH = "src"
  $env:PYTHONUTF8 = "1"

  10 条详细演示：

  & "..\.venv\Scripts\python.exe" scripts\run_processing_demo.py `
    --count 10 `
    --model-mode scripted `
    --simulate normal `
    --verbose `
    --show-markdown `
    --output-dir ".\demo-output\10-normal"

  100 条正常演示：

  & "..\.venv\Scripts\python.exe" scripts\run_processing_demo.py `
    --count 100 `
    --model-mode scripted `
    --simulate normal `
    --output-dir ".\demo-output\100-normal"

  timeout 降级：

  & "..\.venv\Scripts\python.exe" scripts\run_processing_demo.py `
    --count 100 `
    --model-mode scripted `
    --simulate timeout `
    --output-dir ".\demo-output\100-timeout"

  invalid JSON 降级：

  & "..\.venv\Scripts\python.exe" scripts\run_processing_demo.py `
    --count 100 `
    --model-mode scripted `
    --simulate invalid_json `
    --output-dir ".\demo-output\100-invalid-json"

  结果不足补位：

  & "..\.venv\Scripts\python.exe" scripts\run_processing_demo.py `
    --count 100 `
    --model-mode scripted `
    --simulate insufficient_results `
    --output-dir ".\demo-output\100-insufficient"

  读取真实检索候选 JSON：

  & "..\.venv\Scripts\python.exe" scripts\run_processing_demo.py `
    --input-json ".\sample_candidates.json" `
    --model-mode scripted `
    --simulate normal `
    --output-dir ".\demo-output\input-json"

  ## 实际日志节选

  [1/7] 初始化输入数据和RunWorkspace | 输入=100
  [2/7] 规范化候选 | 输入=100 | 输出=100
  [3/7] 分析候选 | 候选=100 | 原子=100 | HTML=99 | 表格=1
  [4/7] 适用性过滤和地区例外处理 | 过滤后=95 | 过滤=5
  地区except命中 | 默认正文=全国默认资费说明
                    region_id=0755
                    覆盖正文=深圳地区专享59元含100GB
                    annotation=深圳坐席办理说明
  [5/7] 生成Markdown | 输入=95 | 输出=95
  批内重排 #1 | 输入=20 | 最终选出=['E003','E001','E002','E004','E005']
  ...
  全局复排 | 池大小=25 | 最终编号=['E037','E068','E094']
  E编号映射 | E037→REAL-KNOWLEDGE-042
               E068→REAL-KNOWLEDGE-073
               E094→REAL-KNOWLEDGE-099
  [6/7] mode=model | degraded=False | fallback_count=0
  [7/7] 输出Top3并写回RunWorkspace | Top3=3

  ## 100 条正常模式 Top3

   排名    knowledge_id          名称          retrieval_rank    retrieval_score
  ━━━━━━  ━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━
      1    REAL-KNOWLEDGE-042    5G流量套餐                42               0.59
  ──────  ────────────────────  ────────────  ────────────────  ─────────────────
      2    REAL-KNOWLEDGE-073    5G流量套餐                73               0.28
  ──────  ────────────────────  ────────────  ────────────────  ─────────────────
      3    REAL-KNOWLEDGE-099    5G流量套餐                99               0.02

  当前生产输出对象没有候选级 rerank_score 和 rerank_reason，因此 Demo 明确显示：

  rerank_score = null
  rerank_reason = 当前接口未提供候选级原因

  没有伪造分数或理由。

  ## 降级模式结果

  - normal
      - mode=model
      - degraded=false
      - fallback_count=0
      - Top3：042、073、099

  - timeout
      - mode=fallback
      - degraded=true
      - fallback_count=3
      - 原因：rerank_timeout、retrieval_rank_supplement、global_model_failed
      - Top3：001、002、003

  - invalid_json
      - mode=fallback
      - degraded=true
      - fallback_count=3
      - 原因：rerank_invalid_json、retrieval_rank_supplement、global_model_failed
      - Top3：001、002、003

  - insufficient_results
      - mode=model_with_fallback
      - degraded=true
      - fallback_count=1
      - 原因：rerank_wrong_count、retrieval_rank_supplement、incomplete_model_result、batch_fallback_used
      - Top3：001、002、003

  ## Prompt 验证

  实际 06_rerank_prompt.json 中：

  顶层：
  query, context, retrieval_query, top_k, candidates

  context：
  region_id, region_name, channel_code,
  request_time, audience, customer_type

  candidate：
  evidence_id, title, content_md

  确认：

  - 没有结构化 knowledge_id
  - 没有 evidence_map
  - 没有 raw、metadata、attributes
  - E001 等临时编号正常存在
  - “100元、30GB、ID、1、E001”自然业务文本原样保留
  - 不进行任何业务文本 ID 扫描或替换
  - 工程侧映射仅保存在重排结果及 Workspace 的 rerank_evidence_map

  ## RunWorkspace 最终字段

  adapter_warnings
  normalized_knowledge_candidates
  knowledge_candidate_analysis
  filtered_knowledge_candidates
  knowledge_filter_reasons
  processed_knowledge_candidates
  processing_warnings
  rerank_evidence_map
  rerank_details
  top3_candidates
  processing_meta

  原始 knowledge_candidates 会在运行前后进行深复制比较。使用 --input-json 时还会比较输入文件字节；若输出目录可能覆盖输入文件，Demo 会直
  接拒绝运行。

  ## 输出文件

  指定 --output-dir 后生成：

  - 01_mock_input.json：输入快照及来源
  - 02_normalized_candidates.json：Adapter 标准候选
  - 03_filtered_candidates.json：过滤结果、决策及告警
  - 04_processed_candidates.json：Markdown 后标准结果
  - 05_sample_content.md：地区 except 代表候选 Markdown
  - 06_rerank_prompt.json：实际模型调用的白名单 Payload
  - 07_top3_result.json：Top3、完整候选和重排元数据
  - processing_demo.log：UTF-8 完整日志

  JSON 均为 UTF-8、保留中文、格式化缩进。

  --input-json 支持：

  [
    {"knowledgeId": "K001", "knowledgeName": "示例"}
  ]

  或：

  {
    "candidates": [
      {"knowledgeId": "K001", "knowledgeName": "示例"}
    ]
  }

  ## 验证结果

  - 五个要求的 CLI 场景：全部退出码 0
  - Demo 专属测试：4/4 通过
  - tests/test_knowledge_processing.py：42/42 通过
  - rerank/orchestrator：13/13 通过
  - 原有 tests/test_processing.py：33/33 通过
  - compileall -q src tests scripts：通过
  - git diff --check：通过，仅有现存 LF/CRLF 提示
  - 按要求未运行 Redis 全仓测试
  - 验证输出和临时目录已清理，无残留测试产物

  没有修改 Retrieval、Answer、MainAgent、Workspace、Config 或底层 Agent 框架；没有修改 Processing 核心业务规则；没有提交 Git。

  当前已知限制是第一版仅支持 scripted 模式，Demo 使用固定查询及深圳/10086 上下文。真实检索结果可直接通过 --input-json 接入；未来真实模型
  应在脚本 _build_model() 扩展点通过项目现有模型工厂接入，不应在 Demo 中读取或硬编码 Key。