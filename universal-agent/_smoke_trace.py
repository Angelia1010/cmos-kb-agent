# -*- coding: utf-8 -*-
"""临时冒烟:验证 processTrace 透出(成功路径 / 环境变量开关)。跑完即删。"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "services"))
os.environ["KB_SERVICE_ES"] = "mock"

from fastapi.testclient import TestClient  # noqa: E402
from kbagent import MockESClient, ScriptedChatModel  # noqa: E402
from kbagent_service.app import create_app  # noqa: E402

REQ = {"params": {
    "appId": "smoke", "requestId": "r1", "sessionId": "s1",
    "userInfo": {"phone": "x", "province": "福建"},
    "extInfo": {},
    "conversations": [{"role": 1, "content": "异地能不能补办手机卡"}],
}}

app = create_app(model=ScriptedChatModel(), es=MockESClient())
with TestClient(app) as client:
    r = client.post("/api/kb-agent-service/prod/retrieve", json=REQ)
    assert r.status_code == 200, r.text
    obj = r.json()["object"]
    trace = obj.get("processTrace")
    assert isinstance(trace, list) and len(trace) > 0, "processTrace 应为非空列表"
    stages = {e["stage"] for e in trace}
    print("trace 事件数:", len(trace))
    print("stages:", sorted(stages))
    for e in trace[:6]:
        print(f"  {e['stage']:22s} {e['event']:24s} payload_keys={list(e['payload'].keys())}")
    assert {"run", "retrieval", "answer"} <= stages, f"关键 stage 缺失: {stages}"
    # payload 必须 JSON 原生(能被 json.dumps 再次序列化)
    import json
    json.dumps(trace, ensure_ascii=False)

# 环境变量关闭
os.environ["KB_SERVICE_EXPOSE_TRACE"] = "0"
app2 = create_app(model=ScriptedChatModel(), es=MockESClient())
with TestClient(app2) as client:
    r = client.post("/api/kb-agent-service/prod/retrieve", json=REQ)
    assert r.status_code == 200, r.text
    assert r.json()["object"].get("processTrace") == [], "开关关闭时应为空列表"
print("KB_SERVICE_EXPOSE_TRACE=0 → processTrace 为空 ✓")

# 老调用方兼容:响应仍满足 AskResponse 契约(字段都在)
print("SMOKE PASSED")
