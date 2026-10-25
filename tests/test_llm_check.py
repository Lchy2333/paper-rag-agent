"""LLM 就绪验证接口测试：POST /api/llm/check（BYOK 复用点）。"""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.auth import get_current_user


class _FakePool:
    def connection(self):
        return None

    def close(self):
        pass


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(
        "app.main.build_agent",
        lambda: (None, None, None, _FakePool(), None, None),
    )
    app.dependency_overrides[get_current_user] = lambda: {
        "id": "11111111-2222-3333-4444-555555555555",
        "username": "tester",
    }
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_llm_check_ok(client, monkeypatch):
    # check_llm_full 在 llm_check 模块内被 import 绑定，patch 模块内引用
    monkeypatch.setattr(
        "app.api.llm_check.check_llm_full",
        lambda base_url, api_key, model: {"ok": True, "latency_ms": 12},
    )
    r = client.post("/api/llm/check", json={
        "base_url": "https://api.example.com/v1",
        "api_key": "user-key",
        "model": "some-model",
    })
    assert r.status_code == 200
    assert r.json() == {"ok": True, "latency_ms": 12}


def test_llm_check_failure_400(client, monkeypatch):
    monkeypatch.setattr(
        "app.api.llm_check.check_llm_full",
        lambda base_url, api_key, model: {"ok": False, "error": "auth failed"},
    )
    r = client.post("/api/llm/check", json={
        "base_url": "https://api.example.com/v1",
        "api_key": "bad-key",
        "model": "some-model",
    })
    assert r.status_code == 400
    assert r.json()["detail"] == "auth failed"


def test_llm_check_requires_auth(client, monkeypatch):
    monkeypatch.setattr(
        "app.api.llm_check.check_llm_full",
        lambda base_url, api_key, model: {"ok": True},
    )
    app.dependency_overrides.pop(get_current_user, None)  # 移除假认证
    r = client.post("/api/llm/check", json={
        "base_url": "https://api.example.com/v1",
        "api_key": "k",
        "model": "m",
    })
    assert r.status_code in (401, 403)
