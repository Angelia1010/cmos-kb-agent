# 03 - state 模块：线程状态与归约器

## 文件

- `uniagent/state/thread_state.py` — ThreadState 扩展状态
- `uniagent/state/reducers.py` — 通用归约器工厂
- `uniagent/state/backend.py` — 状态持久化后端协议 + LocalFileBackend

## 功能说明

### ThreadState

继承 LangGraph 的 `AgentState`，使用 `Annotated[type, reducer]` 声明每个字段的合并策略，使 LangGraph 在并行分支或 `Command(update=...)` 时知道如何合并状态。

| 字段 | 类型 | 归约器 | 用途 |
|------|------|--------|------|
| `artifacts` | `dict[str, Any]` | `idempotent_merge` | 执行中产生的键值制品 |
| `promoted_tools` | `list[str]` | `dedup_list_merge` | 延迟工具提升名称 |
| `todos` | `list[dict]` | `dedup_list_merge` | 任务/待办追踪 |
| `summary` | `str` | `last_wins` | 对话摘要 |
| `token_usage` | `dict[str, int]` | `idempotent_merge` | 累计Token用量 |
| `current_task` | `str | None` | `last_wins` | 当前活动任务ID |
| `loop_iteration` | `int` | `last_wins` | 循环迭代编号 |
| `verification_result` | `dict | None` | `last_wins` | 最新验证结果 |

### 归约器

| 归约器 | 行为 |
|--------|------|
| `last_wins(old, new)` | 直接用新值覆盖 |
| `idempotent_merge(old, new)` | 字典浅合并，新键覆盖旧键 |
| `dedup_list_merge(old, new)` | 列表追加去重，保持顺序 |

### StateBackend

抽象基类定义 `load/save/delete/list_keys` 四个异步方法。`LocalFileBackend` 用本地 JSON 文件实现，每个 key 对应一个 `.json` 文件，可替换为 Redis/S3 等。
