# llm_service — 灵犀大模型直通服务(/llm_397b_api)

灵犀大模型网关直通透传的最小 FastAPI 封装:**不做任何智能体编排**,
`POST /llm_397b_api` 接收用户问题(`query`),调用真实大模型后一次性返回
预测 token 文本。

```
请求(query)
  → HumanMessage 单轮调用灵犀网关
    (config.yaml models[].use → kbagent.shared.lingxi_provider:LingxiSSLChatOpenAI,
     模型 Qwen3.5-397B-A17B-FP8)
  → 预测文本(answer)
```

模型经 `config.yaml` 的 `models[].use` 解析。
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
.venv\Scripts\python -m uvicorn llm_service.app:app --host 0.0.0.0 --port 8002

# Linux
PYTHONPATH=src:services QWEN_API_KEY=sk-xxxx \
    python -m uvicorn llm_service.app:app --host 0.0.0.0 --port 8002
```

## 端点

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 探活 |
| POST | `/llm_397b_api` | 大模型直通调用(见下) |

---

## 输入示例

```bash
curl -X POST http://127.0.0.1:8002/llm_397b_api \
  -H "Content-Type: application/json" \
  -d '{"query": "5G畅享套餐59元档包含多少流量和通话?"}'
```

字段说明(请求):

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `query` | 是 | 用户问题 |

## 输出示例(成功)

```json
{
  "rtnCode": "0",
  "rtnMsg": "success",
  "object": {
    "requestId": "3f2b8c1d4e5f6a7b8c9d0e1f2a3b4c5d",
    "answer": "5G畅享套餐59元档每月包含国内流量20GB、国内通话300分钟……",
    "model": "Qwen3.5-397B-A17B-FP8",
    "elapsedMs": 4321
  }
}
```

要点:

- `requestId` 为服务端生成(uuid4 hex),日志按它检索
- `answer` 为大模型预测 token 文本(单轮、无系统提示词、无工具编排)
- token 用量(`usage_metadata`)只写入服务日志,不进响应契约

## 输出示例(错误)

```json
// query 缺失/为空 → 40001
{
  "rtnCode": "40001",
  "rtnMsg": "参数错误: query: String should have at least 1 character",
  "object": {}
}

// 网关调用异常等未预期错误 → 50001
{ "rtnCode": "50001", "rtnMsg": "大模型网关调用失败", "object": {} }

// 超过端到端超时(默认 660s)→ 50002
{ "rtnCode": "50002", "rtnMsg": "大模型调用超时", "object": {} }
```

## 排障

```bash
# 1. 探活
curl http://127.0.0.1:8002/health

# 2. 最小请求实测(网关不通时返回 50001,看服务日志里的异常栈)
curl -X POST http://127.0.0.1:8002/llm_397b_api \
  -H "Content-Type: application/json" -d '{"query": "你好"}'
```

常见问题:

- **启动报 `models.use ... 解析失败`**:检查 `config.yaml` 的 `use` 路径,
  灵犀 SSL Provider 实际位于 `kbagent.shared.lingxi_provider`
- **50001 且日志显示 401/403**:启动进程未注入 `QWEN_API_KEY` 环境变量
  (`api_key: "${QWEN_API_KEY}"` 占位符未展开)
- **调用挂起**:内网网关单次调用可达分钟级;服务层默认 660s 兜底超时,
  模型层超时见 `config.yaml` 的 `models[].kwargs.timeout`

## 单元测试

```bash
# 在 universal-agent 根目录(离线,注入 mock 模型,不依赖网关)
PYTHONPATH=src;services python -m unittest tests.test_llm_service -v
```
