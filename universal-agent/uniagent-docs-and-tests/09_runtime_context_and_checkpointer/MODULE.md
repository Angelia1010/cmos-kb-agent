# 09 - runtime/context + checkpointer 模块

## 文件

- `uniagent/runtime/context.py` — 用户上下文 IoC
- `uniagent/runtime/checkpointer.py` — 检查点工厂
- `uniagent/runtime/protocols.py` — 跨层类型契约

## 功能说明

### context（用户上下文 IoC）

基于 `ContextVar` 的请求级用户身份传递，无需参数透传：

```python
set_current_user(user)    # 在请求入口设置
user = get_current_user() # 在任意深度获取
```

`CurrentUser` 协议：`user_id: str` + `display_name: str`。

### checkpointer（检查点工厂）

`create_checkpointer(backend)` 根据后端名称创建 LangGraph 兼容的检查点保存器：

| backend | 实现 |
|---------|------|
| `"memory"` | `MemorySaver`（内存，默认） |
| `"sqlite"` | `SqliteSaver`（需安装 `langgraph-checkpoint-sqlite`） |
| 点分路径 | 通过反射加载自定义后端 |

### protocols（跨层类型契约）

`LoopSignalBase` 和 `LoopHookBase` 是零实现依赖的 Protocol 定义。底层模块（如中间件）可从此处导入，无需引入完整的 `runtime/hooks` 模块，避免循环依赖。
