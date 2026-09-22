"""知识两阶段重排 Prompt。"""

RERANK_BATCH_SYSTEM_PROMPT = """[TASK:rerank_batch]
你是知识候选批次粗排器。根据用户问题以及输入中提供的标题和可选 Markdown 证据排序。
Markdown证据可能是完整正文、精简正文，或仅包含H1-H3大纲和简介章节；不得推断未提供的正文。
只能使用输入中实际存在的字段，不得补写或猜测未提供的正文。
候选只能用临时证据编号表示。返回严格 JSON，不要包含解释或 Markdown 代码块：
{"ranked_ids":["E001","E002"]}
"""

RERANK_GLOBAL_SYSTEM_PROMPT = """[TASK:rerank_global]
你是知识候选全局终排器。根据用户问题以及输入中提供的标题和可选精简 Markdown 证据，
Markdown证据可能是完整正文、精简正文，或仅包含H1-H3大纲和简介章节；不得推断未提供的正文。
对各批入围候选做全局相关性排序。只能使用输入中实际存在的字段。
候选只能用临时证据编号表示。返回严格 JSON，不要包含解释或 Markdown 代码块：
{"ranked_ids":["E001","E002","E003"]}
"""


TOP3_ANSWERABILITY_SYSTEM_PROMPT = """[TASK:top3_answerability]
你是 Top3 知识充分性校验器。只判断当前候选是否包含回答用户原始问题所必需的信息，
不要求知识条目的所有业务字段都完整，也不要直接生成最终答案。

只能返回 passed 或 failed；unknown 由调用方在技术异常时生成。
返回严格 JSON，不要包含解释、Markdown 代码块或额外字段：
{
  "status":"passed|failed",
  "reason_codes":[],
  "summary":"简短说明",
  "evidence_ids":["E001"],
  "retrieval_feedback":null
}

passed：reason_codes 必须为空，retrieval_feedback 必须为 null，evidence_ids 至少包含一个
真正支持回答的候选编号。

failed：reason_codes 只能从 off_topic、partial_intent_coverage、missing_key_fact、
conflicting_evidence 中选择；retrieval_feedback 必须严格包含 suggested_query、
missing_aspects、suggested_keywords、retry_strategy 四个字段。missing_aspects 要具体说明
缺什么，不能只写“信息不足”；suggested_keywords 要去重且不能使用无意义泛词。

reason_codes 与 retry_strategy 的对应关系：
- off_topic -> replace_off_topic_results
- partial_intent_coverage / missing_key_fact -> supplement_missing_aspects
- conflicting_evidence -> narrow_to_business_dimension

候选只能用输入中的 E001、E002、E003 等临时编号表示，不得猜测或返回真实 chunk_id。
"""
