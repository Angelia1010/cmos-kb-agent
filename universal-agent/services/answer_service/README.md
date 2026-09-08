# answer_service — 答案子智能体测试服务(真实大模型)

只跑 `src/kbagent/answer` 模块(**AnswerSubAgent**)的 FastAPI 测试服务:
请求直接携带 `query + chunks`(即检索/处理阶段的输出),跳过检索与处理,
专门用于验证**真实大模型**下的答案组织与逐句锚定校验效果。

```
请求(query + chunks)
  → select_fragments 片段精选(取前 4,同文档最多 2)
  → LLM 组织答案([TASK:answer],内联引用,输出 JSON)
  → 逐句锚定校验([TASK:anchor_check],硬事实锚定失败直接删句)
  → FinalAnswer(业务说明 + 办理建议 + 句子级锚定明细 + 知识溯源 + 全链路 trace)
```

模型经 `config.yaml` 的 `models[].use` 解析,生产路径为
`kbagent.shared.lingxi_provider:LingxiSSLChatOpenAI`(灵犀 SSL 策略,已 vendor)。
**配置缺失/解析失败时启动即报错**,不回退离线 ScriptedChatModel。

## 启动

前置:

1. 在 `universal-agent` 项目根目录下运行(热重载按相对路径找 `config.yaml`)
2. 注入网关密钥:`set QWEN_API_KEY=sk-xxxx`(Linux:`export QWEN_API_KEY=...`)

```bash
cd universal-agent

# Windows
set PYTHONPATH=src;services
set QWEN_API_KEY=sk-xxxx
.venv\Scripts\python -m uvicorn answer_service.app:app --host 0.0.0.0 --port 8001

# Linux
PYTHONPATH=src:services QWEN_API_KEY=sk-xxxx \
    python -m uvicorn answer_service.app:app --host 0.0.0.0 --port 8001
```

可选环境变量:

| 变量 | 说明 | 默认 |
| --- | --- | --- |
| `ANSWER_SERVICE_BASE_PATH` | 业务路由前缀 | `/api/answer-service/prod` |
| `ANSWER_SERVICE_APP_IDS` | appId 白名单,逗号分隔;空=全部放行 | 空 |

## 端点

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 探活 |
| GET | `/diag` | 实际加载的模型类、密钥注入状态(脱敏) |
| GET | `/diag/gateway` | 用当前 key 实测大模型网关(返回网关原始响应) |
| POST | `{base}/answer` | 答案生成(见下) |

默认完整路径:`POST /api/answer-service/prod/answer`

---

## 输入示例

```bash
curl -X POST http://127.0.0.1:8001/api/answer-service/prod/answer \
  -H "Content-Type: application/json" \
  -d @- <<'EOF'
{
  "params": {
    "appId": "test-app-001",
    "requestId": "req_20260902_0001",
    "sessionId": "sess_demo_001",
    "query": "5G畅享套餐59元档包含多少流量和通话?",
    "chunks": [
      {
        "chunkId": "chk_5g_59_001",
        "docId": "doc_5g_taocan",
        "docTitle": "5G畅享套餐资费说明",
        "content": "5G畅享套餐59元档:每月包含国内流量20GB、国内通话300分钟,超出后流量按5元/GB计费,通话0.15元/分钟。",
        "category": "套餐",
        "updatedAt": "2026-06-10",
        "score": 0.93
      },
      {
        "chunkId": "chk_5g_59_002",
        "docId": "doc_5g_taocan",
        "docTitle": "5G畅享套餐资费说明",
        "content": "5G畅享套餐各档位均可叠加宽带融合优惠,具体以营业厅受理结果为准。",
        "category": "套餐",
        "updatedAt": "2026-06-10",
        "score": 0.81
      },
      {
        "chunkId": "chk_4g_58_001",
        "docId": "doc_4g_taocan",
        "docTitle": "4G飞享套餐(已停售)",
        "content": "4G飞享套餐58元档包含流量15GB、通话200分钟,该套餐已于2024年停止新办。",
        "category": "套餐",
        "updatedAt": "2023-11-02",
        "score": 0.62
      }
    ]
  }
}
EOF
```

字段说明(请求):

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `params.appId` | 是 | 调用方应用ID(可配白名单 `ANSWER_SERVICE_APP_IDS`) |
| `params.requestId` | 是 | 请求ID |
| `params.sessionId` | 否 | 对话ID,仅回传 |
| `params.query` | 是 | 用户问题 |
| `params.chunks[]` | 是(≥1) | 候选知识片段,**按检索得分排序传入**;服务取前 4 个精选(同文档最多 2 个) |
| `chunks[].chunkId` | 是 | 片段ID,答案引用与溯源的锚点 |
| `chunks[].docId` | 是 | 所属文档ID |
| `chunks[].docTitle` | 是 | 文档标题 |
| `chunks[].content` | 是 | 片段正文 |
| `chunks[].category` | 否 | 业务类目,默认空 |
| `chunks[].updatedAt` | 否 | 更新日期 `yyyy-MM-dd`;空/非法按疑似过旧(`stale=true`) |
| `chunks[].score` | 否 | 检索得分,仅展示参考(精选按输入顺序) |

---

## 输出示例(成功)

> 以下为**示例值**,实际内容由真实大模型生成;`sentences` 为锚定校验后保留的句子。

```json
{
  "rtnCode": "0",
  "rtnMsg": "success",
  "object": {
    "requestId": "req_20260902_0001",
    "sessionId": "sess_demo_001",
    "traceId": "trace_3f9a1c2d4e5b",
    "requestArrivedTime": "2026-09-02 10:15:32.481",
    "elapsedMs": 8642,
    "businessExplanation": "5G畅享套餐59元档每月包含国内流量20GB、国内通话300分钟;超套后流量按5元/GB计费,通话0.15元/分钟。",
    "handlingSuggestion": "可引导客户通过中国移动APP或营业厅办理;如客户有宽带需求,可提示叠加融合优惠,以营业厅受理结果为准。",
    "renderedText": "【业务说明】\n5G畅享套餐59元档每月包含国内流量20GB、国内通话300分钟;超套后流量按5元/GB计费,通话0.15元/分钟。\n\n【办理建议】\n可引导客户通过中国移动APP或营业厅办理;如客户有宽带需求,可提示叠加融合优惠,以营业厅受理结果为准。\n\n【知识来源】\n  1. 5G畅享套餐资费说明 [chk_5g_59_001] 更新于 2026-06-10\n     摘录: 5G畅享套餐59元档:每月包含国内流量20GB、国内通话300分钟,超出后流量按5元/GB计费,通话0.15元/分钟。",
    "sentences": [
      {
        "text": "5G畅享套餐59元档每月包含国内流量20GB、国内通话300分钟。",
        "citations": ["chk_5g_59_001"],
        "hardFact": true,
        "anchored": true,
        "note": ""
      },
      {
        "text": "超出套餐后流量按5元/GB计费,通话按0.15元/分钟计费。",
        "citations": ["chk_5g_59_001"],
        "hardFact": true,
        "anchored": true,
        "note": ""
      },
      {
        "text": "该套餐可叠加宽带融合优惠,具体以营业厅受理结果为准。",
        "citations": ["chk_5g_59_002"],
        "hardFact": false,
        "anchored": true,
        "note": ""
      }
    ],
    "sources": [
      {
        "chunkId": "chk_5g_59_001",
        "docTitle": "5G畅享套餐资费说明",
        "snippet": "5G畅享套餐59元档:每月包含国内流量20GB、国内通话300分钟,超出后流量按5元/GB计费,通话0.15元/分钟。",
        "updatedAt": "2026-06-10",
        "stale": false
      },
      {
        "chunkId": "chk_5g_59_002",
        "docTitle": "5G畅享套餐资费说明",
        "snippet": "5G畅享套餐各档位均可叠加宽带融合优惠,具体以营业厅受理结果为准。",
        "updatedAt": "2026-06-10",
        "stale": false
      }
    ],
    "trace": {
      "trace_id": "trace_3f9a1c2d4e5b",
      "started_ms": 1788321332481,
      "elapsed_ms": 8642,
      "events": [
        {"ts_ms": 1788321332481, "stage": "request", "event": "received",
         "payload": {"requestId": "req_20260902_0001", "query": "5G畅享套餐59元档包含多少流量和通话?", "chunk_count": 3}},
        {"ts_ms": 1788321332490, "stage": "answer", "event": "materials",
         "payload": {"chunk_ids": ["chk_5g_59_001", "chk_5g_59_002", "chk_4g_58_001"]}},
        {"ts_ms": 1788321336120, "stage": "answer", "event": "anchor_check",
         "payload": {"text": "5G畅享套餐59元档每月包含国内流量20GB、国内通话300分钟。", "citations": ["chk_5g_59_001"], "hard_fact": true, "anchored": true, "dropped": false}},
        {"ts_ms": 1788321341123, "stage": "finalize", "event": "done",
         "payload": {"elapsed_ms": 8642, "sentence_count": 3, "source_count": 2}}
      ]
    }
  }
}
```

要点:

- `sentences[].anchored=false` 且 `note="建议核实"`:软性表述锚定失败,保留但提醒核实
- 硬事实(`hardFact=true`)锚定失败会被**直接删除**,不会出现在 `sentences` 里
  (如需排查被删句子,看 `trace.events` 中 `anchor_check` 的 `dropped=true` 记录)
- `sources[].stale=true`:知识更新日期超过溯源天数(默认 365 天)或日期非法
- `trace`:全链路 trace,badcase 回放用;生产化时可去掉

## 输出示例(错误)

```json
// chunks 为空 / query 为空 → 40001
{
  "rtnCode": "40001",
  "rtnMsg": "参数错误: params.chunks: List should have at least 1 item after validation, not 0",
  "object": {}
}

// 网关调用异常等未预期错误 → 50001
{ "rtnCode": "50001", "rtnMsg": "服务内部错误", "object": {} }

// 超过端到端超时(默认 300s)→ 50002
{ "rtnCode": "50002", "rtnMsg": "答案生成超时", "object": {} }
```

## 排障

```bash
# 1. 探活
curl http://127.0.0.1:8001/health

# 2. 检查模型类与密钥注入(留意 api_key 是否显示"占位符未展开")
curl http://127.0.0.1:8001/diag

# 3. 实测大模型网关连通性
curl http://127.0.0.1:8001/diag/gateway
```

常见问题:

- **启动报 `models.use ... 解析失败`**:检查 `config.yaml` 的 `use` 路径,
  灵犀 SSL Provider 实际位于 `kbagent.shared.lingxi_provider`
- **`/diag` 显示 `占位符未展开`**:启动进程未注入 `QWEN_API_KEY` 环境变量
- **`/diag` 模型类是 `ScriptedChatModel`**:说明不是本服务
  (本服务强制真实模型,启动即失败而非回退);检查是否起成了 `kbagent_service`
