"""分层健康检查测试：status（存活=agent+db）与 checks（就绪=db/rag/llm）解耦。

不依赖真实 DB / RAG / LLM：全部用假对象注入 app.state。
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app


class _FakePool:
    """DB 探活可配置：ok / 抛错。"""

    def __init__(self, ok=True):
        self.ok = ok

    def connection(self):
        if not self.ok:
            raise RuntimeError("db connection refused")
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql):
        return None

    def close(self):
        pass


class _FakeRagProxy:
    """RAG 探活可配置。"""

    def __init__(self, health_ok=True):
        self.health_ok = health_ok

    def check_health(self):
        if not self.health_ok:
            raise RuntimeError("rag unreachable")
        return {"status": "ok"}


class _FakeTracker:
    """LLM 状态可配置。"""

    def __init__(self, status="ok"):
        self._status = status

    def status(self, global_llm):
        return {"status": self._status}


@pytest.fixture()
def fake_env(monkeypatch):
    monkeypatch.setattr(
        "app.main.build_agent",
        lambda: (None, None, None, _FakePool(), _FakeRagProxy(), _FakeTracker()),
    )
    with TestClient(app) as c:
        # lifespan 已挂 build_agent 假对象；再显式覆盖 state（health 路由直接读 state）
        c.app.state.pool = _FakePool(ok=True)
        c.app.state.rag_proxy = _FakeRagProxy(health_ok=True)
        c.app.state.llm_tracker = _FakeTracker(status="ok")
        c.app.state.settings = type("S", (), {
            "llm_base_url": "http://x", "llm_api_key": "k", "llm_model": "m",
        })()
        yield c


def test_health_all_ok(fake_env):
    r = fake_env.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["checks"]["database"]["status"] == "ok"
    assert body["checks"]["rag"]["status"] == "ok"
    assert body["checks"]["llm"]["status"] == "ok"


def test_health_db_error_affects_status(fake_env):
    fake_env.app.state.pool = _FakePool(ok=False)
    r = fake_env.get("/health")
    body = r.json()
    assert body["status"] == "error"          # 存活判定失败
    assert body["checks"]["database"]["status"] == "error"


def test_health_rag_error_does_not_affect_status(fake_env):
    """RAG 挂 → 就绪明细 error，但存活 status 仍 ok（编排不误重启）。"""
    fake_env.app.state.rag_proxy = _FakeRagProxy(health_ok=False)
    r = fake_env.get("/health")
    body = r.json()
    assert body["status"] == "ok"             # 存活不受影响
    assert body["checks"]["rag"]["status"] == "error"


def test_health_llm_status_reflected(fake_env):
    fake_env.app.state.llm_tracker = _FakeTracker(status="error")
    r = fake_env.get("/health")
    body = r.json()
    assert body["status"] == "ok"             # LLM 异常不影响存活
    assert body["checks"]["llm"]["status"] == "error"
