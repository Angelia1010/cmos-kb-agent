"""Regression tests for answer repair and evidence/date safeguards (offline)."""
import json
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from kbagent.answer.generate import _check_answer
from tests.test_answer import _answer, _chunk


def draft(script="您好，请进入App搜索挂失停复机。", suggestion=""):
    return {"script": script, "handling_suggestion": suggestion,
            "usability": {"level": "directly_usable", "reasons": [], "uncovered": []}}


class SequenceModel:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(content=json.dumps(response, ensure_ascii=False))


OK = {"consistent": True, "issues": []}
BAD = {"consistent": False, "issues": ["建议中的7天自动解挂无依据"]}


class TestAnswerQuality(unittest.TestCase):
    def chunk(self, date=None):
        return _chunk(content="进入App搜索挂失停复机。", updated_at=date or
                      datetime.now(timezone.utc).isoformat())

    def test_repair_removes_unsupported_rule_and_checks_both_fields(self):
        model = SequenceModel(draft(suggestion="7天自动解挂"), BAD, draft(), OK)
        ans = _answer("我的卡掉了，怎么挂失？", model, [self.chunk()])
        self.assertNotIn("7天", ans.script + ans.handling_suggestion)
        self.assertEqual(ans.usability.level, "directly_usable")
        self.assertEqual(len(model.calls), 4)
        self.assertIn("办理建议:7天自动解挂", model.calls[1][1].content)
        self.assertIn("建议中的7天自动解挂无依据", model.calls[2][1].content)

    def test_failed_repair_preserves_original_content_and_sources(self):
        model = SequenceModel(draft(suggestion="原始办理建议"), BAD,
                              draft("修订话术", "修订建议"), BAD)
        ans = _answer("挂失", model, [self.chunk()])
        self.assertEqual((ans.script, ans.handling_suggestion),
                         (draft()["script"], "原始办理建议"))
        self.assertEqual(ans.usability.level, "verify_first")
        self.assertTrue(any("一致性" in r for r in ans.usability.reasons))
        self.assertEqual(len(ans.sources), 1)
        self.assertEqual(len(model.calls), 4)

    def test_suggestion_only_is_checked(self):
        model = SequenceModel(draft("", "7天自动解挂"), BAD, {},)
        ans = _answer("挂失", model, [self.chunk()])
        self.assertEqual(ans.handling_suggestion, "7天自动解挂")
        self.assertEqual(ans.usability.level, "verify_first")

    def test_malformed_or_failed_repair_preserves_original_draft(self):
        for result in ({}, draft("", ""), TimeoutError("gateway")):
            with self.subTest(result=result):
                model = SequenceModel(draft(suggestion="原始办理建议"), BAD, result)
                ans = _answer("挂失", model, [self.chunk()])
                self.assertEqual(ans.script, draft()["script"])
                self.assertEqual(ans.handling_suggestion, "原始办理建议")
                self.assertEqual(ans.usability.level, "verify_first")
                self.assertTrue(ans.sources)

    def test_checker_requires_boolean_and_valid_issues(self):
        for check in ({"consistent": "false", "issues": []},
                      {"consistent": "true", "issues": []},
                      {"consistent": True},
                      {"consistent": True, "issues": "bad"},
                      {"consistent": True, "issues": ["无依据"]},
                      {"consistent": True, "issues": [1]}, {}):
            with self.subTest(check=check):
                passed, issues = _check_answer(SequenceModel(check), "话术", "建议", "原文")
                self.assertFalse(passed)
                self.assertTrue(issues)

    def test_checker_exception_fails_closed(self):
        passed, issues = _check_answer(SequenceModel(TimeoutError()), "话术", "建议", "原文")
        self.assertFalse(passed)
        self.assertTrue(issues)

    def test_no_materials_does_not_display_invented_answer(self):
        model = SequenceModel(draft())
        ans = _answer("挂失", model, [])
        self.assertEqual(ans.script, "")
        self.assertEqual(ans.usability.level, "not_usable")

    def test_recent_date_and_timestamp_formats(self):
        now = datetime.now(timezone.utc)
        for date in (now.date().isoformat(), now.strftime("%Y-%m-%d %H:%M:%S"),
                     now.isoformat(), now.isoformat().replace("+00:00", "Z")):
            with self.subTest(date=date):
                ans = _answer("挂失", SequenceModel(draft(), OK), [self.chunk(date)])
                self.assertFalse(ans.sources[0].stale)
                self.assertEqual(ans.usability.level, "directly_usable")

    def test_old_timestamp_requires_review(self):
        date = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
        ans = _answer("挂失", SequenceModel(draft(), OK), [self.chunk(date)])
        self.assertTrue(ans.sources[0].stale)
        self.assertEqual(ans.usability.level, "verify_first")

    def test_unknown_date_is_not_claimed_to_be_old(self):
        for date in ("unknown", " "):
            ans = _answer("挂失", SequenceModel(draft(), OK), [self.chunk(date)])
            self.assertFalse(ans.sources[0].stale)
            self.assertEqual(ans.usability.level, "verify_first")
            self.assertTrue(any("无法识别" in r for r in ans.usability.reasons))


if __name__ == "__main__":
    unittest.main()
