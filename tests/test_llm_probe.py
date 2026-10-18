"""LLM 就绪探针测试：check_llm 逻辑 + LLMHealthTracker 状态判定。

不依赖真实 LLM / 网络：用假 tracker / 注入假 check_llm 覆盖状态分支。
"""
import time

import pytest

from core.llm_probe import LLMHealthTracker, check_llm, PROBE_THROTTLE_SECONDS


class _FakeLLM:
    """模拟 ChatOpenAI：可配置成功 / 抛错。"""

    def __init__(self, *, base_url=None, api_key=None, model=None, timeout=10, max_retries=0, max_tokens=None):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries

    def invoke(self, messages):
        if self.model == "boom":
            raise ConnectionError("connection refused")
        return type("Msg", (), {"content": "pong"})


def test_check_llm_missing_config(monkeypatch):
    """配置不完整 → ok=False，不真正调用。"""
    monkeypatch.setattr("core.llm_probe.ChatOpenAI", _FakeLLM)
    result = check_llm("", "key", "model")
    assert result["ok"] is False
    assert "不完整" in result["error"]


def test_check_llm_ok(monkeypatch):
    monkeypatch.setattr("core.llm_probe.ChatOpenAI", _FakeLLM)
    result = check_llm("http://x", "key", "gpt-4o")
    assert result["ok"] is True
    assert "latency_ms" in result


def test_check_llm_failure(monkeypatch):
    monkeypatch.setattr("core.llm_probe.ChatOpenAI", _FakeLLM)
    result = check_llm("http://x", "key", "boom")
    assert result["ok"] is False
    assert "connection refused" in result["error"]


# ---------- LLMHealthTracker 状态判定 ----------

@pytest.fixture()
def tracker():
    return LLMHealthTracker()


def test_tracker_ok_after_success(tracker):
    tracker.record_success()
    s = tracker.status(("", "", ""))
    assert s["status"] == "ok"


def test_tracker_error_after_failures(tracker):
    tracker.record_failure("err1")
    tracker.record_failure("err2")
    s = tracker.status(("", "", ""))
    assert s["status"] == "error"
    assert "err2" in s["error"]


def test_tracker_stale_success_unknown(tracker):
    """有历史成功但已闲置 → unknown，不主动探针（避免空闲烧钱）。"""
    tracker.record_success()
    tracker._last_success_at = time.time() - 3600  # 直接改内部，模拟很久没调用
    s = tracker.status(("", "", ""))
    assert s["status"] == "unknown"


def test_tracker_cold_start_triggers_probe(tracker, monkeypatch):
    """从未成功过（冷启动）→ 触发节流主动探针。"""
    calls = []

    def fake_check(base, key, model):
        calls.append((base, key, model))
        return {"ok": True, "latency_ms": 10}

    monkeypatch.setattr("core.llm_probe.check_llm", fake_check)
    s = tracker.status(("http://base", "key", "model"))
    assert s["status"] == "ok"
    assert calls == [("http://base", "key", "model")]


def test_tracker_probe_throttled(tracker, monkeypatch):
    """探针结果在缓存窗口内 → 不重复调用。"""
    tracker._last_probe_at = time.time()  # 刚探过
    tracker._last_probe_result = {"ok": True, "latency_ms": 5}
    calls = []

    def fake_check(*a):
        calls.append(a)
        return {"ok": False}

    monkeypatch.setattr("core.llm_probe.check_llm", fake_check)
    s = tracker.status(("http://base", "key", "model"))
    assert s["status"] == "ok"  # 用缓存结果
    assert calls == []  # 未触发新探针


def test_tracker_probe_failure_after_throttle_elapsed(tracker, monkeypatch):
    """冷启动 + 探针失败 → error。"""
    tracker._last_probe_at = time.time() - (PROBE_THROTTLE_SECONDS + 1)
    tracker._last_probe_result = {"ok": True}
    calls = []

    def fake_check(*a):
        calls.append(a)
        return {"ok": False, "error": "auth failed"}

    monkeypatch.setattr("core.llm_probe.check_llm", fake_check)
    s = tracker.status(("http://base", "key", "model"))
    assert s["status"] == "error"
    assert "auth failed" in s["error"]
