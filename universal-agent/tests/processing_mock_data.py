"""独立知识处理流水线的确定性 Mock 候选。"""
from __future__ import annotations


def make_top100_candidates(count: int = 100) -> list[dict]:
    """生成指定数量的稳定候选；默认保持原 Top100 契约。"""
    if count < 1:
        raise ValueError("count 必须大于 0")
    candidates = []
    for index in range(1, count + 1):
        relevant = index in {42, 73, 99}
        candidates.append({
            "knowledge_id": f"REAL-KNOWLEDGE-{index:03d}",
            "knowledge_name": "5G流量套餐" if relevant else f"普通业务{index}",
            "retrieval_rank": index,
            "retrieval_score": round((count - index + 1) / count, 4),
            "applicability": {"status": "有效", "regions": ["全国"]},
            "content": (
                "<p>59元含100GB流量，适用5G套餐咨询。</p>"
                if relevant else f"<p>这是与查询无关的普通业务 {index}。</p>"
            ),
            "atoms": [{
                "atom_id": f"ATOM-{index:03d}",
                "group_id": "资费说明",
                "arrange_seq_number": 1,
                "param_name": "月费",
                "content": 59 if relevant else index,
                "wkuntt": "元",
                "annotation": "以当地系统显示为准",
            }],
        })

    # 固定索引的代表性样例供单测和终端 Demo 共同使用，不使用随机数据。
    if count >= 1:
        candidates[0]["demoScenario"] = "region_except"
        candidates[0]["atoms"][0].update({
            "param_name": "地区资费",
            "content": "全国默认资费说明",
            "except_rules": {
                "region_id": "0755",
                "channel_code": "10086",
                "value": "深圳地区专享59元含100GB",
                "annotation": {
                    "visibility": "agent",
                    "content": "深圳坐席办理说明",
                },
            },
        })
    if count >= 2:
        candidates[1]["demoScenario"] = "html_table"
        candidates[1]["content"] = (
            "<section><p>办理渠道说明</p><table>"
            "<tr><th>渠道</th><th>说明</th></tr>"
            "<tr><td><strong>热线</strong></td><td>10086 | 人工</td></tr>"
            "</table><p>自然业务文本：100元、30GB、ID、1、E001。</p></section>"
        )
    if count >= 3:
        candidates[2]["demoScenario"] = "structured_list"
        candidates[2]["content"] = {
            "blocks": [{
                "type": "ul",
                "children": [
                    {"type": "li", "content": "携带有效证件"},
                    {"type": "li", "content": "确认套餐资费"},
                ],
            }]
        }
        candidates[2]["atoms"][0]["annotation"] = {
            "visibility": "public",
            "content": "办理前请确认客户需求",
        }
    if count >= 4:
        candidates[3]["demoScenario"] = "inactive_status"
        candidates[3]["applicability"]["status"] = "下架"
    if count >= 5:
        candidates[4]["demoScenario"] = "invalid_start_time"
        candidates[4]["applicability"]["effective_start"] = "不是有效时间"
    if count >= 6:
        candidates[5]["demoScenario"] = "region_mismatch"
        candidates[5]["applicability"]["region_ids"] = ["9999"]
    if count >= 7:
        candidates[6]["demoScenario"] = "channel_mismatch"
        candidates[6]["applicability"]["channel_codes"] = ["OTHER"]
    if count >= 8:
        candidates[7]["demoScenario"] = "expired"
        candidates[7]["applicability"]["effective_end"] = "2020-01-01"
    if count >= 9:
        candidates[8]["demoScenario"] = "invalid_end_time"
        candidates[8]["applicability"]["effective_end"] = "不是有效时间"
    if count >= 10:
        candidates[9]["demoScenario"] = "empty_rendered_content"
        candidates[9]["content"] = "<p> </p><script>不应渲染</script>"
        candidates[9]["atoms"][0]["content"] = ""
        candidates[9]["atoms"][0]["annotation"] = None
    return candidates
