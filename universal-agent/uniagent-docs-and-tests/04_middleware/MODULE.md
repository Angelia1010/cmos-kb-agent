# 04 - middleware 模块：中间件基类、链组装、排序装饰器

## 文件

- `uniagent/middleware/base.py` — `Middleware` 基类
- `uniagent/middleware/chain.py` — `assemble_middleware_chain()` 链组装
- `uniagent/middleware/positioning.py` — `@after` / `@before` 排序装饰器

## 功能说明

### Middleware 基类

所有中间件的基类，提供四个扩展点（默认透传）：

| 方法 | 调用时机 | 执行顺序 |
|------|---------|---------|
| `before_agent(state)` | Agent节点推理前 | 正序 |
| `after_agent(state)` | Agent节点推理后 | 逆序（洋葱模型） |
| `loop_hooks()` | 返回循环层钩子 | — |

**适配说明**：原实现继承 `langgraph.prebuilt.chat_agent_executor.AgentMiddleware`，该符号在 langgraph 0.6/1.x 不存在。适配后由 `BaseLoop._invoke_agent()` 显式执行中间件链。

### assemble_middleware_chain()

基于锚点的拓扑排序链组装：

1. 以 `built_in` 列表为基础顺序。
2. 按 `@after` / `@before` 锚点约束插入 `extras` 中间件。
3. 可选 `tail_type` 确保某类中间件排在末尾。
4. 循环依赖检测，冲突时抛出 `MiddlewareChainError`。

### @after / @before 装饰器

类装饰器，在中间件类上存储排序元数据：

```python
@after(DanglingToolCallMiddleware)
@before(TokenUsageMiddleware)
class MyMiddleware(Middleware): ...
```

确保 `MyMiddleware` 在 `DanglingToolCallMiddleware` 之后、`TokenUsageMiddleware` 之前执行。
