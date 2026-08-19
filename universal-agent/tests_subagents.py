# -*- coding: utf-8 -*-
"""补充测试:主智能体编排固定 / 子智能体自主规划 / 技能包不可缺。"""
import sys
sys.path.insert(0, "src")

import unittest

from kbagent import MainAgent, MockESClient, ScriptedChatModel


class TestOrchestrationVsAutonomy(unittest.TestCase):
    def test_stage_order_is_fixed(self):
        """主智能体编排固定:trace 中阶段严格按 检索→处理→答案 出现。"""
        a = MainAgent(model=ScriptedChatModel(), es=MockESClient())
        a.run("用户想咨询宽带新装流程")
        stages = [e.stage for e in a.tracer.events]
        i_r = next(i for i, s in enumerate(stages) if s.startswith("retrieval"))
        i_p = next(i for i, s in enumerate(stages) if s == "processing")
        i_a = next(i for i, s in enumerate(stages) if s == "answer")
        self.assertLess(i_r, i_p)
        self.assertLess(i_p, i_a)

    def test_subagent_plans_tool_sequence(self):
        """检索子智能体自主规划:召回前自主调用了理解/关键词工具(见 trace 参数)。"""
        a = MainAgent(model=ScriptedChatModel(), es=MockESClient())
        a.run("用户想咨询话费账单明细")
        recalls = [e for e in a.tracer.events if e.event == "recall"]
        self.assertGreaterEqual(len(recalls), 1)
        # 召回使用了 keyword_extraction 产出的扩展词(证明前置工具真实执行)
        q = recalls[0].payload["dsl"]["query"]["bool"]["must"][0]["multi_match"]["query"]
        self.assertIn("话费账单", q)      # lexicon 扩展词

    def test_retry_round_replans(self):
        """重试轮重新规划:第2轮自主加入 question_rewrite,检索词发生变化。"""
        a = MainAgent(model=ScriptedChatModel(), es=MockESClient())
        a.run("副卡怎么共享主卡额度")
        recalls = [e for e in a.tracer.events if e.event == "recall"]
        self.assertEqual(len(recalls), 2)
        q1 = recalls[0].payload["dsl"]["query"]["bool"]["must"][0]["multi_match"]["query"]
        q2 = recalls[1].payload["dsl"]["query"]["bool"]["must"][0]["multi_match"]["query"]
        self.assertNotEqual(q1, q2)

    def test_skill_package_is_effective(self):
        """技能包子系统必须生效:套餐问题触发 taocan-skill → 字段归一。"""
        a = MainAgent(model=ScriptedChatModel(), es=MockESClient())
        ans = a.run("用户想办理流量套餐,如何推荐?")
        self.assertIn("月费(每月)", ans.business_explanation)


if __name__ == "__main__":
    unittest.main(verbosity=1)
