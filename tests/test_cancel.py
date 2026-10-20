"""前端"随时停止生成"能力回归测试。

覆盖：
- call_llm 在取消标记置位时抛 GenerationCancelled（逐字检查）
- /chat/stream 收到该异常 → 发 event: cancel（而不是 error），且先发 session 事件
- /chat/cancel 端点的归属校验（404/403）+ 置位 + 幂等
全部用假对象，不依赖真实网络 / LLM / 数据库。
"""
import threading

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

from agent.nodes import GenerationCancelled, call_llm
from app.auth import get_current_user
from app.main import app
from core import Dispatcher, ToolRegistry

TEST_UID = "00000000-0000-0000-0000-000000000000"


# ---------- call_llm 逐字检查取消 ----------

def test_call_llm_raises_cancelled_when_event_set():
    from tests.test_graph import FakeLLM

    llm = FakeLLM([AIMessage(content="回答")])
    node = call_llm(llm, Dispatcher(ToolRegistry()))
    ev = threading.Event()
    ev.set()
    with pytest.raises(GenerationCancelled):
        node({"messages": [HumanMessage("hi")]}, {"configurable": {"cancel_event": ev}})


def test_call_llm_ignores_cancel_when_event_not_set():
    from tests.test_graph import FakeLLM

    llm = FakeLLM([AIMessage(content="回答")])
    node = call_llm(llm, Dispatcher(ToolRegistry()))
    out = node({"messages": [HumanMessage("hi")]})
    assert out["messages"][0].content == "回答"


# ---------- /chat/stream：session 事件 + cancel 事件 ----------

class _FakeCursor:
    def __init__(self, rows):
        self._rows = list(rows)
        self._i = 0

    def fetchone(self):
        if self._i < len(self._rows):
            row = self._rows[self._i]
            self._i += 1
            return row
        return None

    def fetchall(self):
        return self._rows


class _FakeConn:
    def __init__(self, owner_map):
        self._owner_map = owner_map

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def execute(self, sql, params=None):
        tid = params[0] if params else None
        uid = self._owner_map.get(tid)
        if uid is None:
            return _FakeCursor([])
        return _FakeCursor([{"id": tid, "user_id": uid}])


class _FakePool:
    def __init__(self, owner_map=None):
        self._owner_map = owner_map or {}

    def connection(self):
        return _FakeConn(self._owner_map)

    def close(self):
        pass


@pytest.fixture()
def client(monkeypatch):
    def fake_build_agent():
        return None, None, None, _FakePool(), None, None

    monkeypatch.setattr("app.main.build_agent", fake_build_agent)
    app.dependency_overrides[get_current_user] = lambda: {
        "id": TEST_UID, "username": "tester", "disabled": False,
    }
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    app.dependency_overrides.pop(get_current_user, None)


def test_chat_stream_emits_session_then_cancel(client):
    class _CancelGraph:
        def get_state(self, config):
            return None

        def stream(self, input, config=None, stream_mode=None):
            raise GenerationCancelled()
            yield  # pragma: no cover

    app.state.graph = _CancelGraph()
    r = client.post("/api/chat/stream", json={"query": "hi"})
    assert r.status_code == 200
    # 先发 session（前端据此拿 thread_id 才能取消），再发 cancel，且不是 error
    assert "event: session" in r.text
    assert "event: cancel" in r.text
    assert "event: error" not in r.text


# ---------- /chat/cancel 端点 ----------

def _pool_client(monkeypatch, owner_map):
    def fake_build_agent():
        return None, None, None, _FakePool(owner_map), None, None

    monkeypatch.setattr("app.main.build_agent", fake_build_agent)
    app.dependency_overrides[get_current_user] = lambda: {
        "id": TEST_UID, "username": "tester", "disabled": False,
    }
    return TestClient(app, raise_server_exceptions=False)


def test_cancel_endpoint_sets_event(monkeypatch):
    owner = {"t-1": TEST_UID}
    with _pool_client(monkeypatch, owner) as c:
        ev = threading.Event()
        app.state.cancels["t-1"] = ev
        r = c.post("/api/chat/cancel", json={"thread_id": "t-1"})
        assert r.status_code == 200
        assert r.json()["status"] == "cancelled"
        assert ev.is_set()
    app.dependency_overrides.pop(get_current_user, None)


def test_cancel_endpoint_no_active_generation_is_idempotent(monkeypatch):
    owner = {"t-1": TEST_UID}
    with _pool_client(monkeypatch, owner) as c:
        r = c.post("/api/chat/cancel", json={"thread_id": "t-1"})
        assert r.status_code == 200
        assert r.json()["status"] == "no_active_generation"
    app.dependency_overrides.pop(get_current_user, None)


def test_cancel_endpoint_forbidden_other_user(monkeypatch):
    owner = {"t-other": "11111111-1111-1111-1111-111111111111"}
    with _pool_client(monkeypatch, owner) as c:
        r = c.post("/api/chat/cancel", json={"thread_id": "t-other"})
        assert r.status_code == 403
    app.dependency_overrides.pop(get_current_user, None)


def test_cancel_endpoint_not_found(monkeypatch):
    with _pool_client(monkeypatch, {}) as c:
        r = c.post("/api/chat/cancel", json={"thread_id": "no-such"})
        assert r.status_code == 404
    app.dependency_overrides.pop(get_current_user, None)
