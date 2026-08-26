# State 字段结构说明

本文档详细说明 uniagent / kbagent 中两套"状态"系统的字段定义、数据格式与存储机制。

---

## 概述：两套状态系统

| 系统 | 类 | 生命周期 | 存储位置 |
|------|----|----------|----------|
| **ThreadState** | `uniagent.state.ThreadState` | 单次 agent 执行期间（内存） | LangGraph checkpointer（Memory / SQLite） |
| **StateBackend** | `LocalFileBackend` / `RedisBackend` | 跨请求持久化（可选） | 本地 JSON 文件 或 Redis 字符串 |

> kbagent 当前架构中，每次请求相互独立，**不需要跨请求持久化**。
> StateBackend 作为基础设施提供，供未来多轮会话或断点续传场景接入。

---

## 一、ThreadState —— 运行时状态字段

`ThreadState` 继承自 LangGraph 的 `AgentState`，是子智能体执行期间的完整状态快照。

### 1.1 基类字段（来自 `AgentState`）

```python
messages: Annotated[Sequence[BaseMessage], add_messages]
remaining_steps: NotRequired[Annotated[int, RemainingStepsManager]]
```

| 字段 | 类型 | 归约策略 | 说明 |
|------|------|----------|------|
| `messages` | `list[BaseMessage]` | 追加（LangGraph `add_messages`） | 对话消息历史，包含 Human / AI / Tool 三种消息 |
| `remaining_steps` | `int` | LangGraph 内部管理 | 剩余可执行步数，防止无限循环（默认从 25 递减） |

### 1.2 扩展字段（来自 `ThreadState`）

```python
artifacts: Annotated[dict[str, Any], idempotent_merge]
promoted_tools: Annotated[list[str], dedup_list_merge]
todos: Annotated[list[dict[str, Any]], dedup_list_merge]
summary: Annotated[str, last_wins]
token_usage: Annotated[dict[str, int], idempotent_merge]
current_task: Annotated[str | None, last_wins]
loop_iteration: Annotated[int, last_wins]
verification_result: Annotated[dict[str, Any] | None, last_wins]
```

| 字段 | 类型 | 默认值 | 归约策略 | 说明 |
|------|------|--------|----------|------|
| `artifacts` | `dict[str, Any]` | `{}` | `idempotent_merge`（新键覆盖，旧键保留） | 执行过程中产生的任意键值制品，如检索结果、处理结果 |
| `promoted_tools` | `list[str]` | `[]` | `dedup_list_merge`（内容去重追加） | 由延迟工具搜索机制动态提升的工具名称 |
| `todos` | `list[dict]` | `[]` | `dedup_list_merge` | 执行期间跟踪的任务/待办事项 |
| `summary` | `str` | `""` | `last_wins`（最新值覆盖） | 由 SummaryMiddleware 生成的对话摘要（长会话压缩） |
| `token_usage` | `dict[str, int]` | `{}` | `idempotent_merge` | 累计 Token 使用量计数器 |
| `current_task` | `str \| None` | `None` | `last_wins` | 当前激活的功能/任务 ID |
| `loop_iteration` | `int` | `0` | `last_wins` | 当前循环迭代编号（由 GoalLoop / TurnLoop 设置） |
| `verification_result` | `dict \| None` | `None` | `last_wins` | 最新一次充分性验证结果 |

### 1.3 归约器说明

| 归约器 | 行为 |
|--------|------|
| `last_wins` | 直接用新值覆盖旧值 |
| `idempotent_merge` | `{**existing, **new}`，新值覆盖同名键，旧键保留 |
| `dedup_list_merge` | 按内容去重追加（dict 类型用 JSON 序列化作为去重 key） |

---

## 二、ThreadState 完整 JSON 示例

以下为检索子智能体执行完毕后 ThreadState 的典型快照：

```json
{
  "messages": [
    {
      "type": "human",
      "content": "我想办理30元流量套餐",
      "id": "msg-001"
    },
    {
      "type": "ai",
      "content": "",
      "tool_calls": [
        {
          "name": "query_understanding",
          "args": { "query": "30元流量套餐" },
          "id": "call-001",
          "type": "tool_call"
        }
      ],
      "id": "msg-002"
    },
    {
      "type": "tool",
      "content": "{\"intent\": \"办理套餐\", \"category\": \"taocan\", \"sub_intent\": \"查询\"}",
      "name": "query_understanding",
      "tool_call_id": "call-001",
      "id": "msg-003"
    },
    {
      "type": "ai",
      "content": "",
      "tool_calls": [
        {
          "name": "coarse_recall",
          "args": {
            "keywords": ["30元", "流量套餐"],
            "retrieval_mode": "hybrid",
            "filters": { "category": "taocan" }
          },
          "id": "call-002",
          "type": "tool_call"
        }
      ],
      "id": "msg-004"
    },
    {
      "type": "tool",
      "content": "已召回 6 条候选知识片段，top3 相关度：0.91 / 0.87 / 0.76",
      "name": "coarse_recall",
      "tool_call_id": "call-002",
      "id": "msg-005"
    },
    {
      "type": "ai",
      "content": "已完成检索，召回6条候选片段，top3分数均超过阈值，充分性通过。",
      "id": "msg-006"
    }
  ],

  "remaining_steps": 23,

  "artifacts": {
    "retrieval_chunks": [
      {
        "chunk_id": "doc001_0",
        "doc_id": "doc001",
        "doc_title": "30元流量套餐说明",
        "content": "月费30元，包含10GB国内流量，超出后1元/GB，有效期1个月。",
        "category": "taocan",
        "score": 0.91,
        "updated_at": "2025-06-01",
        "position": { "page": 1, "section": "套餐详情" },
        "version": "v1.0",
        "source_chunk_ids": [],
        "extra": {}
      },
      {
        "chunk_id": "doc002_1",
        "doc_id": "doc002",
        "doc_title": "套餐变更须知",
        "content": "套餐变更需在营业厅、APP或拨打10086办理，当月生效。",
        "category": "taocan",
        "score": 0.76,
        "updated_at": "2025-05-15",
        "position": {},
        "version": "v1.0",
        "source_chunk_ids": [],
        "extra": {}
      }
    ],
    "processing_result": [
      {
        "chunk_id": "doc001_0",
        "doc_title": "30元流量套餐说明",
        "content": "月费30元，包含10GB国内流量，超出后1元/GB，有效期1个月。",
        "category": "taocan",
        "score": 0.91,
        "updated_at": "2025-06-01"
      }
    ]
  },

  "promoted_tools": [],

  "todos": [],

  "summary": "",

  "token_usage": {
    "prompt_tokens": 1240,
    "completion_tokens": 380,
    "total_tokens": 1620
  },

  "current_task": null,

  "loop_iteration": 2,

  "verification_result": {
    "passed": true,
    "confidence": 0.87,
    "layer": "rule",
    "evidence": ["top3_score_ok", "min_chunks_ok"],
    "hint": ""
  }
}
```

---

## 三、`artifacts` 字段详解

`artifacts` 是贯穿三个子智能体的主要制品传递容器。

### 3.1 检索子智能体写入

**key**: `retrieval_chunks`

```json
"retrieval_chunks": [
  {
    "chunk_id": "doc001_0",       // 片段唯一 ID，格式：{doc_id}_{position}
    "doc_id": "doc001",           // 文档 ID
    "doc_title": "30元流量套餐说明",
    "content": "月费30元，包含...", // 片段正文
    "category": "taocan",         // 业务类目：taocan / kuandai / zhangdan / tousu
    "score": 0.91,                // 相关度分数（RRF 融合后）
    "updated_at": "2025-06-01",   // 文档更新日期（用于过期检测）
    "position": { "page": 1 },    // 在原文档中的位置信息（可选）
    "version": "v1.0",            // 文档版本
    "source_chunk_ids": [],       // 归并溯源：原始 chunk_id 列表（去重后为空）
    "extra": {}                   // 扩展字段（provider 自定义）
  }
]
```

### 3.2 处理子智能体写入

**key**: `processing_result`（覆盖 `retrieval_chunks` 处理后的结果）

结构与 `retrieval_chunks` 相同，差异在于：
- 已去重（按 `doc_id` 去重，每文档最多保留 2 条）
- 已过滤下架内容（含"已下线"/"已停售"的片段被移除）
- 已降噪（去除纯格式/说明性片段）
- 已按 `score` 降序排列
- 若技能包匹配，内容可能经过业务归一处理

---

## 四、`verification_result` 字段详解

由 `SufficiencyVerifier`（充分性验证器）写入，`GoalLoop` 读取以决定是否继续迭代。

```json
"verification_result": {
  "passed": true,          // 是否通过充分性检验
  "confidence": 0.87,      // 置信度 0.0～1.0（纯规则时等于 top3 均值）
  "layer": "rule",         // 验证层：rule（规则优先）/ llm（LLM 补充）
  "evidence": [            // 通过的规则列表
    "top3_score_ok",       //   top3 片段平均分 >= threshold（0.4）
    "min_chunks_ok"        //   候选片段数 >= min_chunk_count（3）
  ],
  "hint": ""               // 验证失败时的改进提示（注入负例反馈）
}
```

验证失败时的 `hint` 示例：
```json
{
  "passed": false,
  "confidence": 0.23,
  "layer": "rule",
  "evidence": [],
  "hint": "top3 平均分 0.23 低于阈值 0.40，建议放宽关键词或扩展检索范围。"
}
```

---

## 五、StateBackend —— 持久化存储格式

### 5.1 存储内容

StateBackend 存储**任意 JSON 对象**（`dict[str, Any]`）。
在 kbagent 中，典型用法是将 ThreadState 的关键字段序列化后写入，
供断点续传或跨进程共享使用。

### 5.2 键命名规范

| 后端 | 逻辑键（logical key） | 实际存储位置 |
|------|----------------------|-------------|
| `LocalFileBackend` | `agent:thread:abc123` | `{state_dir}/{SHA256[:32]}.json` |
| `RedisBackend` | `agent:thread:abc123` | Redis key: `{key_prefix}:agent:thread:abc123` |

推荐键格式：`{scope}:{thread_id}`，例如：
- `session:thread-abc123ef` — 单次会话状态
- `agent:retrieval:thread-abc` — 检索子智能体专属状态

### 5.3 Redis 键示例

```
uniagent:state:agent:thread-abc123   →  { "loop_iteration": 2, "chunks": [...] }
uniagent:state:session:xyz           →  { "history": [...], "token_usage": {...} }
```

### 5.4 LocalFileBackend 目录结构

```
.uniagent/state/
├── _keymap.json                  ← 逻辑键 → SHA256 hash 映射（用于 list_keys）
├── a1b2c3d4e5f6789012345678.json ← 对应逻辑键 "agent:thread:abc123"
└── f9e8d7c6b5a4321098765432.json ← 对应逻辑键 "session:xyz"
```

`_keymap.json` 示例：
```json
{
  "agent:thread:abc123": "a1b2c3d4e5f6789012345678",
  "session:xyz": "f9e8d7c6b5a4321098765432"
}
```

### 5.5 StateBackend 存储值示例

```json
{
  "loop_iteration": 2,
  "verification_result": {
    "passed": true,
    "confidence": 0.87,
    "layer": "rule",
    "evidence": ["top3_score_ok", "min_chunks_ok"],
    "hint": ""
  },
  "token_usage": {
    "prompt_tokens": 1240,
    "completion_tokens": 380,
    "total_tokens": 1620
  },
  "retrieval_chunks": [
    {
      "chunk_id": "doc001_0",
      "doc_title": "30元流量套餐说明",
      "score": 0.91
    }
  ],
  "saved_at": "2026-08-25T10:30:00"
}
```

---

## 六、StateConfig 配置字段

```yaml
# config.yaml
state:
  backend: local           # "local" | "redis" | 自定义点分路径
  state_dir: .uniagent/state   # 本地后端：存储目录

  # Redis 后端（取消注释使用）
  # backend: redis
  # redis_url: "${REDIS_URL:redis://localhost:6379/0}"
  # key_prefix: "uniagent:state"    # Redis 键命名空间
  # ttl: 0                          # 0=永不过期；3600=1小时自动淘汰
```

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `backend` | `str` | `"local"` | 后端类型：`local` / `redis` / 自定义导入路径 |
| `state_dir` | `str` | `".uniagent/state"` | 本地后端存储目录 |
| `redis_url` | `str` | `"redis://localhost:6379/0"` | Redis 连接 URL，支持 `${ENV_VAR}` 替换 |
| `key_prefix` | `str` | `"uniagent:state"` | Redis 键命名空间前缀 |
| `ttl` | `int` | `0` | Redis 键 TTL（秒），0 = 永不过期，`ge=0` 约束 |

---

## 七、kbagent 实际状态流转

kbagent 使用 **RunWorkspace**（ContextVar）作为请求内跨工具共享状态，
**不依赖** StateBackend 进行请求内状态传递。

```
用户 Query
    ↓
MainAgent.arun()
    ├── set_workspace(RunWorkspace)   ← 创建请求级 Workspace（ContextVar）
    │
    ├── RetrievalSubAgent.run()
    │   ├── coarse_recall() → ws.data["chunks"] = [Chunk, ...]
    │   └── GoalLoop 验证 → verification_result 写入 ThreadState
    │
    ├── ProcessingSubAgent.run()
    │   ├── 读 ws.data["chunks"]
    │   ├── 工具处理后 → ws.data["chunks"] = [处理后 Chunk, ...]
    │   └── 返回 processed: List[Chunk]
    │
    └── AnswerSubAgent.run()
        ├── 读 processed: List[Chunk]
        ├── LLM 组织答案 → FinalAnswer
        └── 逐句锚定校验 → hard_fact 句删除 / soft_fact 句标注

RunWorkspace 生命周期 = 单次 arun() 调用
StateBackend 生命周期 = 跨请求（多轮会话 / 断点续传）
```

---

## 八、相关源码位置

| 文件 | 说明 |
|------|------|
| `src/uniagent/state/thread_state.py` | ThreadState 定义（字段 + 归约器注解） |
| `src/uniagent/state/reducers.py` | `last_wins` / `idempotent_merge` / `dedup_list_merge` |
| `src/uniagent/state/backend.py` | `StateBackend` ABC + `LocalFileBackend` + `get_backend()` |
| `src/uniagent/state/redis_backend.py` | `RedisBackend` 实现 |
| `src/uniagent/config/sub_configs.py` | `StateConfig` 配置模型 |
| `src/kbagent/shared/models.py` | `Chunk` / `FinalAnswer` / `SourceRef` 数据结构 |
| `src/kbagent/shared/workspace.py` | `RunWorkspace` ContextVar 状态容器 |
| `tests/test_state_redis.py` | RedisBackend 单元测试（30 项，AsyncMock） |
