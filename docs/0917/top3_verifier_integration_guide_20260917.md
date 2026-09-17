# Top3 Verifier 提交清单与联调指引（2026-09-17）

## 1. 功能简介

`Top3AnswerabilityVerifier` 位于重排之后，用于判断当前 Top3 是否已经包含回答用户原始
query 所必需的信息。

- `passed`：Top3 足以回答，检索模块结束内部 loop，并把可回答的 Top3 返回给 MainAgent；
- `failed`：Verifier 正常完成判断，但 Top3 不足，需要检索模块参考
  `retrieval_feedback` 发起下一轮检索；
- `unknown`：Verifier 自身发生超时、模型调用异常或模型输出非法，由检索模块内部执行异常策略，
  不能直接当成普通检索失败。

整体边界是：MainAgent 只调用检索模块；检索模块内部组织检索、数据处理、重排、Verifier
和重试 loop。Verifier 本身不实现 loop，也不持有轮次、当前 Top3 或最大重试次数，这些
状态由检索模块内部的循环控制逻辑负责；这部分当前尚未实现，由检索同事联调时补充。

## 2. 核心提交文件

### 2.1 生产联调必须提交

1. `universal-agent/src/kbagent/processing/verifier.py`
   - Verifier 主实现；
   - 调用注入的模型；
   - 处理空候选、Evidence 映射、结构解析和异常收敛。

2. `universal-agent/src/kbagent/processing/prompts.py`
   - 新增 `[TASK:top3_answerability]` 系统 Prompt。

3. `universal-agent/src/kbagent/shared/knowledge_processing/models.py`
   - 三个 Literal：`VerificationStatus`、`VerificationReasonCode`、`RetryStrategy`；
   - 两个结果模型：`RetrievalFeedback`、`Top3VerificationResult`；
   - 字段一致性和原因/策略冲突校验。

4. `universal-agent/src/kbagent/shared/knowledge_processing/__init__.py`
   - 导出新增 Literal 和结果模型。

5. `universal-agent/src/kbagent/processing/__init__.py`
   - 导出 `Top3AnswerabilityVerifier`。

### 2.2 建议和核心实现一起提交

6. `universal-agent/tests/test_top3_answerability_verifier.py`
   - 覆盖三种状态、异常解析、字段一致性、Evidence 映射、超时、模型异常和空候选。

7. `universal-agent/src/kbagent/scripted_model.py`
   - 增加 `[TASK:top3_answerability]` 离线模拟响应；
   - 只用于本地和测试，不替代生产模型。

8. `docs/0916/top3_verifier_integration_guide_20260917.md`
   - 本联调说明。

## 3. 在检索模块内部调用 Verifier

下面只展示仓库中已经存在、可以直接调用的 API 拼接方式，不包含尚未实现的 Retrieval
loop：

```python
from kbagent.processing.agent import ProcessingSubAgent
from kbagent.processing import Top3AnswerabilityVerifier
from kbagent.shared.knowledge_processing.adapter import normalize_processing_context
from kbagent.shared.workspace import get_workspace

ws = get_workspace()

# ProcessingSubAgent.run() 是当前真实接口，返回 List[ProcessedKnowledge]，
# 同时向 Workspace 写入 top3_candidates 和 processed_chunks。
top3_candidates = await ProcessingSubAgent(model).run()

verification = await Top3AnswerabilityVerifier(model).verify(
    query=ws.query,
    candidates=top3_candidates,
    retrieval_query=ws.data.get("retrieval_query"),
    context=normalize_processing_context(ws.data.get("processing_context")),
)
```

上述调用应由检索模块在每一轮召回之后执行，不应放在 MainAgent。`ProcessingSubAgent.run()`
要求当前 `RunWorkspace` 已包含：

- `chunks`：本轮检索得到的 `List[Chunk]`；
- `knowledge_candidates`：通过现有 `retrieval_to_candidates(chunks=...)` 生成的候选。

`processing_context` 和 `retrieval_query` 当前都是 Workspace 中的可选数据；缺少上下文时，
现有 `normalize_processing_context()` 会使用项目默认行为。

`candidates` 中每条候选至少需要：

- 稳定且非空的 `chunk_id`；
- 非空 `content_md`；
- 建议保留 `name`、`knowledge_id`、`retrieval_rank` 等原字段。

模型只会看到 E001、E002、E003 临时编号、标题和 Markdown，不会看到真实 `chunk_id`、
metadata 或 raw。Verifier 会将模型返回的 Evidence 编号安全映射回本轮真实 `chunk_id`。

当前提供的是 Python API，没有新增生产 HTTP 接口。

## 4. 当前代码与目标联调边界

### 4.1 当前代码现状

当前仓库还没有实现目标架构，实际代码是：

- `MainAgent.arun()` 先调用
  `RetrievalSubAgent(...).run(query, region_code)`；
- `RetrievalSubAgent.run()` 当前只执行一次 `intergrate_all`，失败时回退
  `keyword_extraction + coarse_recall`，返回 `List[Chunk]`，没有 loop；
- MainAgent 随后调用 `retrieval_to_candidates(...)`；
- MainAgent 再调用 `self._processing.run()`；
- Processing 完成后，MainAgent 从 `ws.data["processed_chunks"]` 取结果交给答案模块。

因此，检索同事联调时需要调整的是 `RetrievalSubAgent` 及其内部编排；本次 Verifier 提交
没有修改这些文件。

### 4.2 目标边界

目标调用关系为：

```text
MainAgent
  └─ 调用 RetrievalSubAgent
       └─ 每轮：检索 → retrieval_to_candidates → ProcessingSubAgent.run
                → Top3AnswerabilityVerifier.verify
                  ├─ passed：结束内部 loop
                  ├─ failed：使用 retrieval_feedback 发起下一轮检索
                  └─ unknown：执行检索模块内部的技术异常策略
  └─ 获取检索模块最终产物后调用 AnswerSubAgent
```

上图是目标架构流程图，不是当前已经存在的代码。

### 4.3 联调前需要与检索同事确认

当前 `RetrievalSubAgent.run()` 的返回类型是 `List[Chunk]`，项目还没有定义“检索模块内部
完成 Processing 和 Verifier 后”的新对外结果契约。联调前需要确认：

1. 最大检索轮次耗尽后，是返回当前最优 Top3，还是抛出异常触发 MainAgent 现有降级；
2. `unknown` 是否先在检索模块内部重试 Verifier，以及允许重试几次；
3. 下一轮如何把 `retrieval_feedback` 映射到现有 `intergrate_all` 或
   `keyword_extraction / coarse_recall` 的参数。

这些契约当前代码中尚不存在，本文不预设新的类名、方法名或返回对象。

检索模块内部的循环控制逻辑需要持有：

- 当前轮次和最大检索轮次；
- 原始 query；
- 当前 retrieval query；
- 历史检索反馈；
- 当前 Top3；
- 重复 query / 重复结果检测；
- `unknown` 的重试、降级或告警策略。

这些运行信息不要写回 `Top3VerificationResult`。

## 5. 检索模块内部如何使用 failed 反馈

`retrieval_feedback` 包含：

- `suggested_query`：下一轮建议检索语句；
- `missing_aspects`：当前结果具体缺少的内容；
- `suggested_keywords`：建议补充或重点使用的关键词；
- `retry_strategy`：下一轮检索调整方向。

策略含义：

- `supplement_missing_aspects`：保留当前方向，针对缺失点补充召回；
- `replace_off_topic_results`：当前候选偏题，重新改写并替换无关结果；
- `broaden_semantic_recall`：当前召回为空或过窄，扩大语义范围和同义表达；
- `narrow_to_business_dimension`：围绕业务对象、地区、渠道、时间或资费类型缩小范围。

检索模块可以综合使用四个字段，不要求机械地把 `suggested_query` 原样作为最终检索语句。
这些反馈只在检索模块内部流转，正常情况下不需要暴露给 MainAgent。

## 6. 特殊状态处理

### no_valid_candidates

Top3 为空，或前三条候选都缺少有效 `chunk_id` / `content_md` 时，Verifier 不调用模型，
确定性返回：

- `status="failed"`；
- `reason_codes=["no_valid_candidates"]`；
- `retry_strategy="broaden_semantic_recall"`；
- 可用的 suggested query、missing aspects 和 keywords。

### unknown

可能原因：

- `verifier_timeout`；
- `verifier_model_error`；
- `verifier_invalid_output`。

`unknown` 的 `retrieval_feedback` 始终为 `None`。推荐由检索模块内部处理：

1. 对模型超时或暂时性服务错误做有限次数 Verifier 重试；
2. 持续异常时记录告警并按产品策略降级；
3. 不要把技术故障当作检索质量差，直接消耗普通 Retrieval loop 次数。

## 7. 联调重点检查

1. MainAgent 只调用检索模块，不直接调用 Processing 或 Verifier；
2. 检索模块内部每轮按“检索 → 数据处理 → 重排 → Verifier”执行；
3. Verifier 使用用户原始 query，`retrieval_query` 传当前轮检索语句；
4. 输入候选确实是本轮重排后的当前 Top3；
5. 每条候选的 `chunk_id` 和 `content_md` 已正确保留；
6. `passed` 时检索模块结束 loop，并向 MainAgent 返回可回答的 Top3；
7. `failed` 时 feedback 只在检索模块内部驱动下一轮检索；
8. `unknown` 在检索模块内部进入技术异常策略，不消耗普通业务检索重试；
9. 检索模块内部设置最大轮次、重复 query 和重复 Top3 防护；
10. MainAgent 收到成功结果后再调用答案模块，必要时使用 `evidence_chunk_ids` 缩小证据；
11. 真实模型继续由现有配置和依赖注入创建，不在 Verifier 内写死配置。
