"""全局异常处理回归测试：未捕获异常 → 分类响应（不裸 500）。

用假 graph 直接抛各类异常（模拟 LLM 调用失败冒泡到路由层），
验证 exception_handler 的分类映射 + 内置错误路径（422）不受影响。
"""
import openai
import pytest
from fastapi.testclient import TestClient

from app.main import app


# 绕过 openai 异常对 httpx.Response 的构造要求，仅用于测试分类判定
class _FakeConnectionError(openai.APIConnectionError):
    def __init__(self):
        pass


class _FakeRateLimitError(openai.RateLimitError):
    def __init__(self):
        pass


class _FakeAuthError(openai.AuthenticationError):
    def __init__(self):
        pass


class _FailingGraph:
    """假 graph：invoke 直接抛指定异常（模拟 graph.invoke 冒泡）。"""

    def __init__(self, exc):
        self._exc = exc

    def invoke(self, *args, **kwargs):
        raise self._exc


@pytest.fixture()
def client():
    # raise_server_exceptions=False：异常由我们的 handler 消费，专注断言响应体
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _post_chat_with_error(client, exc):
    app.state.graph = _FailingGraph(exc)
    return client.post("/api/chat", json={"query": "hi"})


def test_llm_unreachable_returns_502(client):
    r = _post_chat_with_error(client, _FakeConnectionError())
    assert r.status_code == 502
    body = r.json()
    assert body["status"] == "error"
    assert body["error_code"] == "llm_unreachable"
    assert "LLM" in body["message"]


def test_llm_rate_limited_returns_429(client):
    r = _post_chat_with_error(client, _FakeRateLimitError())
    assert r.status_code == 429
    assert r.json()["error_code"] == "llm_rate_limited"


def test_llm_config_error_returns_500_with_code(client):
    r = _post_chat_with_error(client, _FakeAuthError())
    assert r.status_code == 500
    body = r.json()
    assert body["error_code"] == "llm_config_error"
    # 对外脱敏：不泄露 key/模型等细节
    assert "api_key" not in body["message"].lower()


def test_unknown_error_returns_500_internal(client):
    r = _post_chat_with_error(client, ValueError("boom"))
    assert r.status_code == 500
    assert r.json()["error_code"] == "internal_error"


def test_validation_error_still_422(client):
    """FastAPI 内置的请求校验（422）不被全局 handler 抢走。"""
    app.state.graph = _FailingGraph(ValueError("不应被触发"))
    r = client.post("/api/chat", json={"query": 12345})  # query 类型错
    assert r.status_code == 422
