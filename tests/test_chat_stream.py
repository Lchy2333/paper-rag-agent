"""SSE 流式对话端点（/api/chat/stream）回归测试。

用假 graph 直接产出 (mode, chunk) 事件流，验证：stage / token / done / approval /
error 事件按协议正确打包成 SSE，且不用真实 LLM / RAG / 数据库。
"""
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessageChunk, ToolMessage

from app.auth import get_current_user
from app.main import app


class _FakeGraph:
    """假 graph：get_state 无挂起；stream（同步）按预设 chunk 序列逐个产出。"""

    def __init__(self, chunks):
        self._chunks = list(chunks)

    def get_state(self, config):
        return None

    def stream(self, input, config=None, stream_mode=None):
        for item in self._chunks:
            yield item


class _FakeCursor:
    def fetchone(self):
        return None

    def fetchall(self):
        return []


class _FakeConn:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def execute(self, *args, **kwargs):
        return _FakeCursor()


class _FakePool:
    def connection(self):
        return _FakeConn()

    def close(self):
        pass


@pytest.fixture()
def client(monkeypatch):
    def fake_build_agent():
        return None, None, None, _FakePool(), None, None

    monkeypatch.setattr("app.main.build_agent", fake_build_agent)
    app.dependency_overrides[get_current_user] = lambda: {
        "id": "00000000-0000-0000-0000-000000000000",
        "username": "tester",
        "disabled": False,
    }
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    app.dependency_overrides.pop(get_current_user, None)


def _interrupt(payload):
    return type("Interrupt", (), {"value": payload})()


def test_chat_stream_normal_flow(client):
    """stage → token（拼接）→ done 完整链路；messages 模式按元组 (msg, metadata) 下放。"""
    app.state.graph = _FakeGraph([
        ("custom", {"label": "正在理解问题、优化检索词…"}),
        ("messages", (AIMessageChunk(content="回"), {"langgraph_node": "call_llm"})),
        ("messages", (AIMessageChunk(content="答"), {"langgraph_node": "call_llm"})),
        ("values", {"messages": []}),
    ])

    r = client.post("/api/chat/stream", json={"query": "hi"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    text = r.text
    assert "event: stage" in text and "正在理解问题" in text
    assert "event: token" in text and "回" in text and "答" in text
    assert "event: done" in text


def test_chat_stream_filters_tool_message_tokens(client):
    """call_tool 的 ToolMessage（工具结果）不应被当 token 流给前端。"""
    app.state.graph = _FakeGraph([
        ("messages", (ToolMessage(content="工具结果不应该出现", tool_call_id="c1"),
                      {"langgraph_node": "call_tool"})),
        ("messages", (AIMessageChunk(content="回答"), {"langgraph_node": "call_llm"})),
        ("values", {"messages": []}),
    ])

    r = client.post("/api/chat/stream", json={"query": "hi"})
    assert r.status_code == 200
    assert "工具结果不应该出现" not in r.text
    assert "回答" in r.text
    assert "event: done" in r.text


def test_chat_stream_approval_interrupt(client):
    """图挂起 → 推送 approval 事件后结束（不发 done）。"""
    app.state.graph = _FakeGraph([
        ("custom", {"label": "正在调用工具 delete_doc…"}),
        ("values", {"__interrupt__": [_interrupt({"tool": "delete_doc", "args": {"id": "A"}})]}),
    ])

    r = client.post("/api/chat/stream", json={"query": "删A"})
    assert r.status_code == 200
    text = r.text
    assert "event: approval" in text
    assert '"tool": "delete_doc"' in text
    assert "event: done" not in text


def test_chat_stream_error_event(client):
    """图内部抛错 → 用 SSE error 事件收尾，不裸 500。"""
    class _RaisingGraph:
        def get_state(self, config):
            return None

        def stream(self, input, config=None, stream_mode=None):
            raise ValueError("boom")
            yield  # pragma: no cover

    app.state.graph = _RaisingGraph()
    r = client.post("/api/chat/stream", json={"query": "hi"})
    assert r.status_code == 200
    assert "event: error" in r.text
    assert '"error_code": "internal_error"' in r.text
