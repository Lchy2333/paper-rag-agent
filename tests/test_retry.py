"""重试机制回归测试。

覆盖两处修复：
- L1：call_llm 节点对 LLM 瞬态失败（ARK 超时/限流/5xx）指数退避重试
- L2：工具链路重试真正生效（HttpTool 不再吞异常；Dispatcher 对瞬态失败重试）

全部用假对象/假异常，不依赖真实网络与真实 LLM。
"""
import openai
import pytest
import requests
from langchain_core.messages import AIMessage, HumanMessage

from agent.nodes import call_llm
from core import Dispatcher, ToolRegistry
from core.retry import is_transient, retry_call
from core.tool import ToolBase, ToolResult
from tools.http_tool import HttpTool


class _FakeRateLimitError(openai.RateLimitError):
    """绕过 openai 异常对 httpx.Response 的构造要求，仅用于测试瞬态判定。"""

    def __init__(self):
        pass


def _no_sleep(*args, **kwargs):
    """测试注入：重试退避不真实等待。"""
    return None


# ---------- is_transient 判定 ----------

def test_is_transient_matches_langchain_openai_errors():
    """langchain-openai 1.7 自定义异常保留 openai 继承链，is_transient 依然匹配。"""
    from langchain_openai.chat_models.base import (
        OpenAIConnectionError,
        OpenAIRateLimitError,
        OpenAITimeoutError,
    )

    assert issubclass(OpenAIRateLimitError, openai.RateLimitError)
    assert issubclass(OpenAITimeoutError, openai.APITimeoutError)
    assert issubclass(OpenAIConnectionError, openai.APIConnectionError)
    assert is_transient(_FakeRateLimitError())


def test_is_transient_flags_requests_timeout_connection():
    assert is_transient(requests.Timeout())
    assert is_transient(requests.ConnectionError())


def test_is_transient_does_not_flag_nontransient():
    assert not is_transient(ValueError("bad args"))


# ---------- retry_call 本身 ----------

def test_retry_call_retries_transient_then_succeeds():
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise _FakeRateLimitError()
        return "ok"

    assert retry_call(flaky, retryable=is_transient, max_attempts=3,
                      sleep=_no_sleep) == "ok"
    assert len(calls) == 2


def test_retry_call_does_not_retry_nontransient():
    calls = []

    def boom():
        calls.append(1)
        raise ValueError("bad args")

    with pytest.raises(ValueError):
        retry_call(boom, retryable=is_transient, max_attempts=3, sleep=_no_sleep)
    assert len(calls) == 1


def test_retry_call_gives_up_after_max_attempts():
    calls = []

    def always_fail():
        calls.append(1)
        raise _FakeRateLimitError()

    with pytest.raises(openai.RateLimitError):
        retry_call(always_fail, retryable=is_transient, max_attempts=3,
                   sleep=_no_sleep)
    assert len(calls) == 3


# ---------- L1：LLM 调用重试 ----------

class _FlakyLLM:
    """模拟 ChatOpenAI：前 N 次 invoke 抛瞬态异常，之后返回正常回复。"""

    def __init__(self, fail_times, reply, exc):
        self.fail_times = fail_times
        self.reply = reply
        self.exc = exc
        self.calls = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.exc
        return self.reply


class _EchoTool(ToolBase):
    def name(self):
        return "echo"

    def build_schema(self):
        return {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "回显",
                "parameters": {"type": "object", "properties": {}},
            },
        }

    def execute(self, args, user_id=None):
        return ToolResult(success=True, data={"echo": args.get("msg")})


def _dispatcher():
    registry = ToolRegistry()
    registry.register(_EchoTool())
    return Dispatcher(registry, sleep=_no_sleep)


def test_call_llm_retries_transient_then_succeeds():
    dispatcher = _dispatcher()
    llm = _FlakyLLM(fail_times=2, reply=AIMessage(content="最终回答"),
                    exc=_FakeRateLimitError())
    node = call_llm(llm, dispatcher)

    out = node({"messages": [HumanMessage(content="hi")]})

    assert llm.calls == 3  # 2 次失败 + 1 次成功
    assert out["messages"][0].content == "最终回答"


def test_call_llm_raises_after_max_attempts():
    dispatcher = _dispatcher()
    llm = _FlakyLLM(fail_times=99, reply=AIMessage(content="x"),
                    exc=_FakeRateLimitError())
    node = call_llm(llm, dispatcher)

    with pytest.raises(openai.RateLimitError):
        node({"messages": [HumanMessage(content="hi")]})
    assert llm.calls == 3  # 最多重试 3 次后放弃


# ---------- L2：工具链路重试 ----------

class _FakeTimeoutSession:
    def request(self, method, url, json=None, timeout=30, headers=None):
        raise requests.Timeout("boom")


def test_http_tool_raises_timeout_instead_of_swallowing():
    """修复前：HttpTool 把超时吞成 ToolResult；修复后：异常冒泡交给 retry_call。"""
    tool = HttpTool(session=None, base_url="http://localhost:8080",
                    config={"path": "/api/v1/ask", "method": "POST",
                            "schema": {}},
                    name="ask")
    tool.session = _FakeTimeoutSession()

    with pytest.raises(requests.Timeout):
        tool.execute({"query": "hi"})


class _FlakyTool(ToolBase):
    """第一次调用抛瞬态异常，之后成功。"""

    def __init__(self):
        self.calls = 0

    def name(self):
        return "flaky"

    def build_schema(self):
        return {
            "type": "function",
            "function": {
                "name": "flaky",
                "description": "",
                "parameters": {"type": "object", "properties": {}},
            },
        }

    def execute(self, args, user_id=None):
        self.calls += 1
        if self.calls == 1:
            raise requests.Timeout("boom")
        return ToolResult(success=True, data={"ok": 1})


class _BoomTool(ToolBase):
    """抛非瞬态异常，不应重试。"""

    def __init__(self):
        self.calls = 0

    def name(self):
        return "boom"

    def build_schema(self):
        return {
            "type": "function",
            "function": {
                "name": "boom",
                "description": "",
                "parameters": {"type": "object", "properties": {}},
            },
        }

    def execute(self, args, user_id=None):
        self.calls += 1
        raise ValueError("bad args")


def test_dispatch_retries_transient_tool_failure():
    tool = _FlakyTool()
    registry = ToolRegistry()
    registry.register(tool)
    dispatcher = Dispatcher(registry, sleep=_no_sleep)

    result = dispatcher.dispatch("flaky", "{}")

    assert result.success
    assert tool.calls == 2  # 1 次失败 + 1 次成功


def test_dispatch_no_retry_on_nontransient_tool_failure():
    tool = _BoomTool()
    registry = ToolRegistry()
    registry.register(tool)
    dispatcher = Dispatcher(registry, sleep=_no_sleep)

    result = dispatcher.dispatch("boom", "{}")

    assert not result.success
    assert tool.calls == 1  # 非瞬态错误不重试
    assert "calling tool failed" in result.error
