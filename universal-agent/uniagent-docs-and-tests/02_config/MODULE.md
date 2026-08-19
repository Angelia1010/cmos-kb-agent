# 02 - config 模块：配置系统

## 文件

- `uniagent/config/app_config.py` — 根配置模型 + YAML热重载 + ContextVar栈
- `uniagent/config/sub_configs.py` — 各子配置数据类（Model/Tool/Loop/State/Verification/Skill）
- `uniagent/config/reload_boundary.py` — 热重载边界安全检查

## 功能说明

### AppConfig（根配置）

基于 Pydantic BaseModel 的应用配置，支持：

1. **YAML 加载**：`AppConfig.from_yaml(path)` 从 YAML 文件解析。
2. **环境变量展开**：YAML 中的 `${VAR}` 或 `${VAR:default}` 自动替换为环境变量值。
3. **热重载**：`get_app_config()` 基于文件签名（mtime + size + sha256）检测变更，变更时自动重新加载。
4. **ContextVar 栈隔离**：`push_current_app_config()` / `pop_current_app_config()` 实现请求级配置覆盖，栈顶配置优先于文件配置。

### sub_configs（子配置模型）

| 子配置 | 用途 |
|--------|------|
| `ModelConfig` | LLM提供商配置（use路径、model名、temperature） |
| `ToolConfig` | 工具配置（use路径、enabled开关、kwargs） |
| `LoopConfig` | 循环引擎（max_iterations、max_tokens、max_time_seconds） |
| `StateConfig` | 状态持久化后端（local/自定义） |
| `VerificationConfig` | 验证策略（none/llm/composite） |
| `SkillConfig` | 技能子系统（enabled、directories、max_active） |

### reload_boundary（热重载边界）

注册"启动期字段"（如 `sandbox`），这些字段在首次加载后冻结——热重载时若变更则忽略并发出警告。防止运行时状态被配置变更破坏。
