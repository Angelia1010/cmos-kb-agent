# 12 - tools 模块：工具注册表

## 文件

- `uniagent/tools/registry.py`

## 功能说明

### get_available_tools()

从三个来源加载工具并按名称去重（先出现者优先）：

1. **配置定义的工具**：遍历 `config.tools` 列表，通过 `resolve_variable()` 反射加载。若是类则尝试实例化（传入 `kwargs`）。未启用（`enabled=False`）或加载失败的工具跳过。
2. **extra_tools**：代码中直接传入的工具（如框架内置工具）。
3. **mcp_tools**：从 MCP 服务器发现的工具（KB 适配版已移除延迟发现，但接口保留）。

### 工具名称提取

支持三种工具对象形式：
- `BaseTool` 实例 → `.name`
- 含 `.name` 属性的对象 → `str(.name)`
- 可调用对象 → `.__name__`
- 其他 → `str(id(obj))`

### 设计要点

- **去重策略**：按名称先到先得，防止同名工具冲突。
- **excluded_names**：可传入需完全排除的工具名称集合。
- **失败宽容**：单个工具加载失败不影响其余工具，仅记录 warning。
