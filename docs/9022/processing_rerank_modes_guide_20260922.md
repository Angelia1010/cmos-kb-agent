# Processing Rerank模式解说文档

日期：2026-09-22  
代码范围：`universal-agent/src/kbagent/processing` 及Processing调试UI  
依据：当前工作区最新代码

## 1. Rerank在Processing中的位置

Processing固定执行：

```text
analyze → filter → build_markdown → rerank
```

Rerank接收Build Markdown阶段产生的 `ProcessedKnowledge`，对可参与重排的候选执行两阶段排序：

```text
有效候选
  ↓ 按retrieval_rank稳定排序并分配临时Evidence ID
Batch粗排：默认每批20条，每批选Top5
  ↓ 汇总各批入围结果
Global终排：最多25条，最终选Top3
  ↓
top3_candidates → processed_chunks
```

默认Top100候选时，正常路径是5次Batch调用加1次Global调用，共6次模型调用。候选不超过3条时不调用重排模型；Prompt最小载荷超预算时，对应阶段也不会调用模型。

## 2. 通用Prompt结构

无论哪种模式，模型都只使用临时Evidence ID，不接触真实knowledge_id或chunk_id。

基础结构为：

```json
{
  "query": "用户原始Query",
  "top_k": 5,
  "candidates": [
    {
      "evidence_id": "E001",
      "title": "知识标题"
    }
  ]
}
```

需要正文投影的模式会额外出现：

```json
"content_md": "发送给模型的Markdown投影"
```

通用约束：

- 使用原始用户 `query`，不向模型发送 `retrieval_query`；
- 不发送 `retrieval_rank`；
- 不发送真实knowledge_id、chunk_id、metadata、raw、上下文或Evidence映射；
- 候选标题投影上限默认200字符；
- 模型只能返回严格JSON：`{"ranked_ids":["E001"]}`；
- 模型返回后，程序校验ID范围、类型、重复项和数量。

## 3. 当前四种rerank_input_mode

默认值：

```text
title_then_content
```

当前支持：

- `title_only`
- `headings_and_intro`
- `title_then_content`
- `title_and_content`

### 3.1 模式总表

| 模式 | Batch粗排 | Global终排 | 适用方向 |
|---|---|---|---|
| `title_only` | Evidence ID、title | Evidence ID、title | 最低输入量基线 |
| `headings_and_intro` | Evidence ID、title、H1-H3大纲、可选简介正文 | Evidence ID、title、H1-H3大纲、可选简介正文 | 结构化轻量重排 |
| `title_then_content` | Evidence ID、title | Evidence ID、title、确定性精简正文 | 当前默认，成本与语义平衡 |
| `title_and_content` | Evidence ID、title、正文投影 | Evidence ID、title、正文投影 | 高输入量对照模式 |

## 4. title_only

### Batch发送

```json
{
  "evidence_id": "E001",
  "title": "流量年包60GB版"
}
```

### Global发送

与Batch相同，只发送Evidence ID和标题。

### 特点

- Prompt最小；
- 不发送 `content_md`；
- 模型调用轮次与其他模式相同；
- 标题信息不足或多个知识标题相似时，排序质量可能下降；
- 不存在title与Markdown H1重复问题。

## 5. headings_and_intro

### 当前发送内容

Batch和Global都发送：

```json
{
  "evidence_id": "E001",
  "title": "流量年包60GB版",
  "content_md": "# 流量年包60GB版\n\n## 基础信息\n\n### 套餐对象\n\n### 套餐简介\n\n### 套餐简介\n\n360元/年，包含60GB国内通用流量。"
}
```

### 标题提取

- 从最终 `content_md` 解析ATX格式标题；
- 保留代码围栏外的H1、H2、H3；
- 按原Markdown顺序排列；
- 普通H4-H6不进入大纲；
- 代码围栏中的 `###` 不会被误识别成标题；
- 无可识别标题时仍保留独立 `title` 字段。

### 简介正文白名单

只有标题规范化后精确匹配以下名称，才附带章节正文：

- 业务简介
- 套餐简介
- 产品简介
- 业务概述
- 套餐概述
- 产品概述

规范化处理尾部Markdown关闭标记、连续空格和中英文冒号，不做模糊语义匹配。

以下标题会进入大纲，但不会附带正文：

- 业务介绍
- 套餐介绍
- 产品介绍
- 有效期
- 上线日期
- 下线日期
- 其他普通H1-H3章节。

### 简介章节边界

- H1简介：到下一个H1前结束；
- H2简介：到下一个H1或H2前结束；
- H3简介：到下一个H1、H2或H3前结束；
- 简介内部的H4-H6可以作为简介正文保留；
- 多个简介章节按原文顺序处理。

### 正文完整性

简介正文按以下完整单元处理：

- 完整段落；
- 完整代码围栏；
- 完整表格行；
- 表格数据行会携带完整表头和分隔行。

预算不足时省略完整单元，不从段落或表格单元格中间截断。

### 当前已知问题

当前代码同时发送独立 `title` 和Markdown H1。标准Processing链路中，H1就是同一个知识标题，因此该模式当前一定存在标题重复。

这还可能让H1绕过独立title的200字符投影限制。推荐的后续修正是：保留独立 `title`，大纲只保留H2/H3；如果Markdown H1与候选name不一致才保留H1。

此外，简介标题本身已经在大纲中出现，而每个简介正文单元又会携带一次简介标题作为上下文。因此一个简介包含多个段落或表格行时，类似 `### 业务简介` 的标题可能重复多次。这样可以保证任一独立正文单元仍有章节语境，但也会增加Prompt占用，属于当前实现需要进一步优化的重复信息。

## 6. title_then_content

这是当前默认模式。

### Batch发送

只发送：

```text
evidence_id + title
```

不发送 `content_md`。

### Global发送

发送：

```text
evidence_id + title + 确定性精简content_md
```

### Global正文选择顺序

1. `matched_atom_ids` 能精确对应当前候选Atom时，命中Atom优先；
2. 其余Markdown正文按Query与章节标题、正文关键词重合度排序；
3. 标题关键词重合权重大于正文重合；
4. 无明显相关性时按原文顺序兜底；
5. 只选择完整段落、完整表格行等不可分单元。

如果候选完整 `content_md` 的JSON序列化字符数不超过单候选Global上限1000字符，当前代码会直接保留完整Markdown和原顺序。

### 特点

- Batch阶段输入小；
- 只有各批入围、最多25条候选在Global发送正文；
- 当前最均衡，继续作为默认模式；
- Global发送短篇完整Markdown时可能同时出现独立title和相同H1；
- 长文经过单元切分后，正文前缀会排除H1，通常不会重复知识标题。

## 7. title_and_content

### Batch发送

发送：

```text
evidence_id + title + content_md正文投影
```

Batch单候选正文目标上限使用 `prompt_max_chars_per_candidate`，默认6000字符，同时受Batch Prompt总上限10000字符约束。

### Global发送

与 `title_then_content` 的Global逻辑相同，使用最多1000字符的确定性精简正文，并受30000字符总预算约束。

### 特点

- 所有参与Batch的候选都可能向模型发送正文；
- 不增加调用轮次，但输入字符、Token、耗时和模型接收正文的候选范围明显增加；
- 适合作为高输入量对照，不是默认模式；
- Batch和Global发送可完整容纳的短Markdown时，都可能出现独立title和相同H1重复；
- 长正文走单元切分后通常不重复H1。

## 8. 正文投影的通用确定性规则

非 `headings_and_intro` 的正文模式使用以下逻辑：

### 8.1 精确Atom命中

如果 `matched_atom_ids` 能在候选 `atoms` 中找到同ID Atom：

- 读取Atom标题、正文和单位；
- 渲染成Markdown单元；
- 将命中Atom排在普通Markdown单元之前；
- Trace记录 `matched_atom_ids_used`。

如果当前结构无法精确找到Atom，不猜测映射关系，直接使用Markdown关键词和原文顺序。

### 8.2 Markdown切分

Markdown正文按：

- 标题上下文；
- 完整段落；
- 完整表格；
- 完整表格数据行

切分。表格数据行会重复携带表头和分隔行，确保单独投影时仍是合法、可理解的表格片段。

### 8.3 相关度排序

Query和正文按英文/数字词及中文词片段提取关键词：

- 标题重合计更高权重；
- 正文重合计普通权重；
- Query完整出现在单元中有额外加分；
- 分数相同时保持原文顺序。

## 9. Prompt字符预算

默认配置：

| 配置 | 默认值 |
|---|---:|
| `prompt_max_chars_per_title` | 200 |
| `batch_prompt_max_chars` | 10000 |
| `prompt_max_chars_per_candidate` | 6000 |
| `global_prompt_max_chars_per_candidate` | 1000 |
| `global_prompt_max_chars` | 30000 |
| `rerank_timeout_seconds` | 15秒 |

字符预算统计包含：

- System Prompt；
- Query；
- top_k；
- Evidence ID；
- 候选title；
- JSON字段和结构；
- JSON转义后的实际字符；
- 可选 `content_md`。

Global有最多256字符安全余量。

### Global公平分配

Global先计算最小载荷固定开销，从30000字符中扣除固定开销和安全余量，剩余正文预算按候选数量公平初配：

```text
初始单候选额度 = min(1000, 正文总预算 // 实际候选数)
```

初配后有剩余预算，再按候选顺序轮询加入仍可容纳的完整正文单元。

### headings_and_intro的Batch公平分配

该模式的Batch同样先计算最小载荷，再在本批候选之间公平分配大纲/简介预算。候选内部优先级是：

```text
H1/H2/H3标题大纲 → 简介正文
```

### 最终复检

Prompt完成JSON序列化后再次检查实际总字符数。若仍超预算：

- 从当前正文占用最多的候选开始；
- 每次移除最后一个完整投影单元；
- 不删除Evidence ID、title、Query或top_k；
- 最小必要载荷仍超预算时不调用模型。

## 10. 模型输出解析

模型必须返回：

```json
{"ranked_ids":["E003","E001","E007"]}
```

程序会校验：

- 顶层只能有 `ranked_ids`；
- `ranked_ids` 必须是列表；
- 每项必须是字符串；
- ID必须属于本阶段候选；
- 不能重复；
- 有效结果数量必须等于本阶段top_k。

常见warning：

- `rerank_invalid_json`
- `rerank_invalid_schema`
- `rerank_non_string_id`
- `rerank_duplicate_id`
- `rerank_unknown_id`
- `rerank_wrong_count`
- `rerank_timeout`
- `rerank_model_error`
- `rerank_prompt_budget_exceeded`

## 11. retrieval_rank的内部作用

`retrieval_rank` 不发送给模型，但仍在程序内部用于：

- 重排前稳定排序；
- 模型异常或超时时降级；
- Prompt最小载荷超预算时降级；
- 模型返回数量不足时补位；
- Batch入围池的确定性补齐；
- Global完全失败时从全部有效候选中选取Top3；
- 最终排序稳定性。

## 12. 降级与补位

### Batch

每批期望最多5条。模型返回不足、无效或调用失败时：

- 先保留模型返回的有效Evidence ID；
- 再按本批内部稳定顺序补足Top5；
- 补位顺序本质上来自 `retrieval_rank` 排序；
- 批次结果继续进入Global候选池。

### Global

Global期望最多3条：

- 有部分有效结果时，保留有效结果，再从全部有效候选稳定顺序补足；
- 完全没有有效模型结果时，从全部有效候选中按稳定顺序取Top3，而不是只从Global池补位。

### 最终降级状态

常见 `fallback_reasons`：

- `retrieval_rank_supplement`
- `incomplete_model_result`
- `batch_fallback_used`
- `global_model_failed`
- 上述模型解析、超时、异常或预算warning code。

只要最终不是完整、无补位的Global模型结果，通常会标记为degraded。

## 13. 候选数量和调用次数

默认参数：

```text
batch_size = 20
batch_top_k = 5
global_pool_size = 25
final_top_k = 3
```

典型调用次数：

| 有效候选数 | Batch调用 | Global调用 | 正常总调用 |
|---:|---:|---:|---:|
| 0～3 | 0 | 0 | 0 |
| 4～20 | 1 | 1 | 2 |
| 21～40 | 2 | 1 | 3 |
| 41～60 | 3 | 1 | 4 |
| 61～80 | 4 | 1 | 5 |
| 81～100 | 5 | 1 | 6 |

Prompt超预算时对应阶段会跳过模型，所以实际调用数可能少于表中数量，但不会因为模式切换而额外增加调用阶段。

## 14. 原始数据和最终输出

所有标题截取、正文精简、大纲提取都只作用于发送给模型的Prompt投影：

- 不修改原始候选；
- 不修改原始 `content_md`；
- 不改变最终 `top3_candidates` 结构；
- 不改变最终 `processed_chunks` 结构；
- 最终Chunk仍返回完整Markdown；
- `rerank_rank` 在最终候选副本上连续设置为1、2、3。

## 15. Trace记录

Rerank Trace记录：

- 实际 `input_mode`；
- 每个Batch的输入ID、模型ID、补位后ID；
- Global候选池和最终ID；
- 实际System/User Prompt；
- 模型原始输出和解析结果；
- 原始、初始、最终Prompt字符数；
- 固定开销、安全余量和正文预算；
- 每候选原始/发送标题字符数；
- 每候选原始/发送正文字符数；
- 选择方式、单元数量和省略数量；
- 命中的Atom ID；
- fallback和degradation原因。

Trace只在 `/process/ui/run` 调试请求中采集，普通正式 `/process` 不自动记录完整调试Trace。

## 16. UI中的模式控制

内部调试页面可以选择四种模式，选择只影响当前一次 `/process/ui/run`：

```text
title_then_content（推荐）
title_only（低输入量）
headings_and_intro（标题大纲+简介）
title_and_content（高输入量）
```

UI后端为每次请求单独创建 `KnowledgeProcessingOptions`，不会修改共享全局Options，不会导致并发请求串模式。

正式 `/process` 当前仍使用原 `ProcessingRequest`，HTTP请求体不接受 `rerank_input_mode`；主服务默认使用 `title_then_content`。

## 17. 模式选择建议

### 生产默认

推荐继续使用：

```text
title_then_content
```

理由是Batch输入较小，Global又能获得正文语义。

### 低成本基线

使用：

```text
title_only
```

适合验证正文是否真正改善排序。

### 结构化轻量实验

使用：

```text
headings_and_intro
```

适合标题层级清晰、Atom字段名有语义、简介质量稳定的数据。当前存在重复H1问题，建议修正后再做正式效果比较。

### 高输入量对照

使用：

```text
title_and_content
```

适合排查Batch阶段缺少正文是否影响召回排序，但需要注意输入成本和正文发送范围。

## 18. 当前已知限制和待处理项

1. `headings_and_intro` 当前固定重复独立title和相同H1；
2. `headings_and_intro` 的简介标题会在大纲和各简介正文单元中重复；
3. `title_then_content` Global短正文可能重复title和H1；
4. `title_and_content` Batch/Global短正文可能重复title和H1；
5. `headings_and_intro` 只识别ATX标题，不识别Setext标题；
6. 纯数字H2仍会进入大纲；
7. 简介采用精确白名单，未列出的近义标题不会附带正文；
8. “有效期”只保留标题，不附带日期正文；
9. 模型只负责排序，程序不会获取或展示模型隐藏推理过程；
10. 模式不同可能产生不同Top3，比较效果时应同时记录Query、候选集和实际模式。

## 19. 相关代码位置

- 模式和默认配置：`universal-agent/src/kbagent/shared/knowledge_processing/models.py`
- 两阶段重排与Prompt投影：`universal-agent/src/kbagent/processing/rerank.py`
- Batch/Global System Prompt：`universal-agent/src/kbagent/processing/prompts.py`
- 调试接口模式白名单：`universal-agent/services/processing_service/debug_ui.py`
- 调试页面模式选择：`universal-agent/services/processing_service/static/processing_debug/index.html`
- 调试页面回放逻辑：`universal-agent/services/processing_service/static/processing_debug/app.js`

本文只说明当前实现，没有修改业务代码。
