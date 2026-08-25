# -*- coding: utf-8 -*-
"""kbagent 演示入口 —— PYTHONPATH=src python main.py

4 个场景（完全离线：ScriptedChatModel + MockES）：
  1. 正常链路（套餐推荐，GoalLoop 1轮验证通过 + taocan-skill 技能包触发）
  2. 循环重试（冷门问题，验证失败→框架注入反馈→第2轮改写重召→轮次耗尽携最优退出）
  3. 缓存命中快速通道
  4. LLM 故障 → 降级兜底

uniagent 框架演示见 test_uniagent_e2e.py（7个场景，不依赖 kbagent）。
"""
import sys
sys.path.insert(0, "src")

from kbagent import MainAgent, MockESClient, ScriptedChatModel


def divider(t):
    print("\n" + "=" * 70 + f"\n  {t}\n" + "=" * 70)


def show(p, q, dump=False):
    ans = p.run(q)
    print(f"\n>> 用户问题: {q}\n>> trace_id: {ans.trace_id} | 耗时: {ans.elapsed_ms}ms")
    print(ans.render())
    if dump:
        print("\n---- 关键 trace 事件 ----")
        for e in p.tracer.events:
            print(f"  [{e.stage}] {e.event} {str(e.payload)[:110]}")


class BrokenModel(ScriptedChatModel):
    def _generate(self, *a, **kw):
        raise TimeoutError("LLM gateway timeout")


def main():
    p = MainAgent(model=ScriptedChatModel(), es=MockESClient())

    divider("场景1: 正常链路(检索子智能体自主规划,1轮验证通过;taocan-skill 技能包触发)")
    show(p, "用户想办理流量套餐,如何推荐?", dump=True)

    divider("场景2: 循环重试(冷门问题;验证失败反馈→第2轮改写;轮次上限=2 硬退出)")
    show(p, "副卡怎么共享主卡额度", dump=True)

    divider("场景3: 缓存命中快速通道")
    show(p, "请问用户想办理流量套餐,如何推荐")

    divider("场景4: LLM 故障 → 降级兜底")
    broken = MainAgent(model=BrokenModel(), es=MockESClient())
    show(broken, "宽带新装怎么办理")


if __name__ == "__main__":
    main()
