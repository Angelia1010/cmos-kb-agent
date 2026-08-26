"""子配置模型——所有轻量级配置数据类均集中于此文件。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """单个 LLM 提供商的配置。"""

    name: str = "default"
    use: str = Field(
        description="聊天模型类的点分导入路径，例如 'langchain_openai:ChatOpenAI'。",
    )
    model: str = Field(description="模型标识符，例如 'gpt-4o'。")
    temperature: float = 0.0
    # ── API 接入配置 ──
    api_key: str = Field(default="", description="API 密钥，支持 ${ENV_VAR} 环境变量替换。")
    base_url: str = Field(default="", description="自定义端点（代理 / 私有化部署）。")
    timeout: float = Field(default=60.0, ge=0.0, description="请求超时（秒），0 表示不限制。")
    max_retries: int = Field(default=2, ge=0, description="SDK 层自动重试次数。")
    extra_headers: dict[str, str] = Field(default_factory=dict, description="额外 HTTP 请求头。")
    kwargs: dict[str, object] = Field(default_factory=dict, description="兜底：任意 provider 专属参数，可覆盖上述字段。")



class ToolConfig(BaseModel):
    """单个工具的配置。"""

    name: str = Field(description="工具的唯一名称。")
    use: str = Field(
        description="工具可调用对象的点分导入路径，例如 'mytools.search:web_search_tool'。",
    )
    group: str = ""
    enabled: bool = True
    kwargs: dict[str, object] = Field(default_factory=dict)



class LoopConfig(BaseModel):
    """循环引擎的配置。"""

    max_iterations: int = Field(default=25, ge=1)
    max_tokens: int = Field(default=0, ge=0)
    max_time_seconds: float = Field(default=0, ge=0)
    verify_every: int = Field(default=1, ge=1)
    checkpoint_every: int = Field(default=1, ge=1)


class StateConfig(BaseModel):
    """外部状态持久化的配置。

    支持两种内置后端（通过 ``backend`` 字段选择）：

    - ``"local"``：本地 JSON 文件（开发/单机部署，默认）；
    - ``"redis"``：Redis 字符串（多进程/分布式部署）。

    也可填写点分导入路径（如 ``"myapp.state:CustomBackend"``）接入自定义后端，
    构造函数将接收当前 ``StateConfig`` 实例作为唯一参数。
    """

    backend: str = Field(
        default="local",
        description="状态后端：'local' / 'redis' 或自定义后端的点分导入路径。",
    )
    # ── 本地文件后端 ──
    state_dir: str = Field(
        default=".uniagent/state",
        description="本地文件后端的存储目录（backend='local' 时有效）。",
    )
    # ── Redis 后端 ──
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description=(
            "Redis 连接 URL（backend='redis' 时有效）。"
            "明文：redis://[:pass@]host[:port][/db]；"
            "TLS：rediss://...；"
            "支持 ${ENV_VAR} 环境变量替换。"
        ),
    )
    key_prefix: str = Field(
        default="uniagent:state",
        description=(
            "Redis 键命名空间前缀（backend='redis' 时有效）。"
            "实际键格式：{key_prefix}:{logical_key}。"
        ),
    )
    ttl: int = Field(
        default=0,
        ge=0,
        description=(
            "Redis 键生存时间（秒，backend='redis' 时有效）。"
            "0 = 永不过期；>0 = 自动淘汰（适合会话级状态）。"
        ),
    )


class VerificationConfig(BaseModel):
    """目标验证的配置。"""

    strategy: str = Field(default="none", description="验证策略：'none'、'command'、'llm' 或 'composite'。")
    command: str = Field(default="", description="CommandVerifier 使用的 Shell 命令。")
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    evaluator_model: str = Field(default="", description="评估器 LLM 的点分导入路径。")


class SkillConfig(BaseModel):
    """技能子系统的配置。"""

    enabled: bool = Field(default=False, description="是否启用技能子系统。")
    directories: list[str] = Field(
        default_factory=lambda: ["skills"],
        description="扫描技能包的目录列表。",
    )
    max_active: int = Field(default=3, ge=1, description="同时激活的最大技能数量。")
    auto_match: bool = Field(
        default=True,
        description="是否根据用户输入的触发词自动匹配技能。",
    )
