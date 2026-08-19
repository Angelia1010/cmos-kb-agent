# 01 - imports 模块：动态导入解析器

## 文件

- `uniagent/imports/resolvers.py`

## 功能说明

本模块提供运行时动态导入能力，将 `"pkg.mod:name"` 格式的字符串解析为真实的 Python 对象。
是整个框架配置驱动的基石 —— 配置文件中的 LLM 类、工具类、中间件类等全部通过此模块在运行时按需加载。

### 核心API

| 函数 | 用途 |
|------|------|
| `resolve_variable(path, expected_type?)` | 导入任意对象（变量、函数、类实例），可选类型校验 |
| `resolve_class(path, base_class?)` | 导入**类**，可选基类校验（确保是子类） |

### 异常

- `ResolveError`：路径格式错误（缺少 `:`）、模块不存在、属性不存在、类型不匹配时抛出。

### 设计要点

1. **冒号分隔约定**：`"module.path:AttrName"` —— 冒号左侧是模块路径（可 `importlib.import_module`），右侧是 `getattr` 属性名。
2. **安全友好的错误提示**：模块不存在时提示 `pip install`，属性不存在时指明模块。
3. **无缓存**：每次调用都做真实的 `importlib.import_module`，依赖 Python 自身的模块缓存。
