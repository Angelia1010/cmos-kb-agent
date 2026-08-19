# 10 - verification 模块：验证器协议与内置验证器

## 文件

- `uniagent/verification/verifier.py` — `Verifier` 协议 + `VerificationResult`
- `uniagent/verification/builtins/always_pass.py` — 空操作验证器
- `uniagent/verification/builtins/llm_verifier.py` — LLM 独立评估验证器
- `uniagent/verification/builtins/composite_verifier.py` — 链式多层验证器

## 功能说明

### Verifier 协议

`runtime_checkable` 的 Protocol，强制生成器/评估器分离原则：

```python
class Verifier(Protocol):
    async def verify(self, goal: str, state: dict) -> VerificationResult: ...
```

### VerificationResult（验证结果）

| 字段 | 说明 |
|------|------|
| `passed: bool` | 是否通过 |
| `evidence: str` | 人类可读的依据 |
| `confidence: float` | 0.0–1.0 置信度 |
| `layer: str` | 产生结果的层（lint/test/e2e/llm/noop...） |
| `details: dict` | 任意结构化详情 |

### 内置验证器

| 验证器 | 功能 |
|--------|------|
| `AlwaysPassVerifier` | 始终返回通过，用于测试或 TurnLoop 场景 |
| `LLMVerifier` | 用独立 LLM 评估目标是否达成。将对话摘要+制品+目标发送给评估 LLM，解析返回的 JSON `{passed, confidence, evidence}`。支持置信度阈值（低于阈值则判为未通过） |
| `CompositeVerifier` | 链式多验证器（lint → test → e2e），在第一个失败层快速失败。所有层通过才算整体通过 |

### LLMVerifier 的生成器/评估器分离

评估器 LLM 是独立实例，不是生成解决方案的 LLM。这确保了评估的客观性——生成者不能"自我评分"。
