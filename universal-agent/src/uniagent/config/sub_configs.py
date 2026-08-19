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
    kwargs: dict[str, object] = Field(default_factory=dict)



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
    """外部状态持久化的配置。"""

    backend: str = Field(default="local", description="状态后端：'local' 或点分导入路径。")
    state_dir: str = Field(default=".uniagent/state", description="本地文件后端的存储目录。")


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
