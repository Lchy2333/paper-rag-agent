"""认证测试：注册 / 登录 / JWT 验签 / get_current_user 依赖。

全部 mock 数据库访问（app.db 模块函数），不依赖真实 PG；
app 通过 monkeypatch build_agent 避免 lifespan 连库。
"""
import uuid
from datetime import datetime, timedelta, timezone

import jwt
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.auth import create_access_token, get_current_user
from app.config import settings
from app.main import app

FAKE_USER_ID = str(uuid.uuid4())
TEST_SECRET = "test-secret-" * 6  # ≥32 字节，避免 PyJWT 短密钥警告
FAKE_USER = {
    "id": FAKE_USER_ID,
    "username": "admin",
    "password_hash": "$2b$12$abcdefghijklmnopqrstuv",  # 仅占位，verify 走 mock
    "disabled": False,
}


@pytest.fixture()
def client(monkeypatch):
    """真实 app + 假 build_agent（不连库）；DB 访问由各测试 monkeypatch app.db。"""
    class _FakePool:
        def close(self):
            pass

    monkeypatch.setattr("app.main.build_agent", lambda: (None, None, None, _FakePool(), None, None))
    monkeypatch.setattr(settings, "jwt_secret", TEST_SECRET)
    monkeypatch.setattr(settings, "jwt_expire_minutes", 60)
    with TestClient(app) as c:
        yield c


# ---------- 注册 ----------

def test_register_ok(client, monkeypatch):
    monkeypatch.setattr("app.db.get_user_by_username", lambda pool, u: None)
    monkeypatch.setattr(
        "app.db.create_user",
        lambda pool, u, h: {"id": FAKE_USER_ID, "username": u, "disabled": False},
    )
    r = client.post("/api/auth/register", json={"username": "alice", "password": "secret123"})
    assert r.status_code == 201
    body = r.json()
    assert body["username"] == "alice"
    assert body["user_id"] == FAKE_USER_ID
    assert "password" not in str(body)  # 绝不回传密码相关字段


def test_register_conflict(client, monkeypatch):
    monkeypatch.setattr("app.db.get_user_by_username", lambda pool, u: FAKE_USER)
    r = client.post("/api/auth/register", json={"username": "admin", "password": "secret123"})
    assert r.status_code == 409
    assert r.json()["detail"] == "用户名已存在"


def test_register_validation_error(client):
    r = client.post("/api/auth/register", json={"username": "ab", "password": "short"})
    assert r.status_code == 422  # 长度校验失败走 FastAPI 内置 422


# ---------- 登录 ----------

def test_login_ok_returns_jwt(client, monkeypatch):
    monkeypatch.setattr("app.db.get_user_by_username", lambda pool, u: FAKE_USER)
    monkeypatch.setattr("app.api.auth.verify_password", lambda p, h: True)
    r = client.post("/api/auth/login", json={"username": "admin", "password": "Lucifer49368293!"})
    assert r.status_code == 200
    body = r.json()
    assert body["token_type"] == "bearer"
    assert body["user_id"] == FAKE_USER_ID
    payload = jwt.decode(body["access_token"], TEST_SECRET, algorithms=["HS256"])
    assert payload["sub"] == FAKE_USER_ID  # JWT 里就是 user_id


def test_login_wrong_password(client, monkeypatch):
    monkeypatch.setattr("app.db.get_user_by_username", lambda pool, u: FAKE_USER)
    monkeypatch.setattr("app.api.auth.verify_password", lambda p, h: False)
    r = client.post("/api/auth/login", json={"username": "admin", "password": "wrong-pass"})
    assert r.status_code == 401


def test_login_unknown_user(client, monkeypatch):
    monkeypatch.setattr("app.db.get_user_by_username", lambda pool, u: None)
    r = client.post("/api/auth/login", json={"username": "nobody", "password": "secret123"})
    assert r.status_code == 401  # 与密码错误同响应，不暴露"用户是否存在"


# ---------- 受保护接口 401 ----------

def test_chat_requires_auth(client):
    r = client.post("/api/chat", json={"query": "hi"})
    assert r.status_code == 401


def test_chat_invalid_token_401(client):
    r = client.post(
        "/api/chat",
        json={"query": "hi"},
        headers={"Authorization": "Bearer not.a.jwt"},
    )
    assert r.status_code == 401


# ---------- get_current_user 依赖（独立 mini app，聚焦依赖本身） ----------

def _make_me_app():
    mini = FastAPI()
    mini.state.pool = object()  # get_current_user 不触碰它（db 走 mock）

    @mini.get("/me")
    def me(user=Depends(get_current_user)):
        return {"id": user["id"], "username": user["username"]}

    return mini


def test_current_user_ok(monkeypatch):
    monkeypatch.setattr("app.db.get_user_by_id", lambda pool, uid: FAKE_USER)
    monkeypatch.setattr(settings, "jwt_secret", TEST_SECRET)
    token = create_access_token(uuid.UUID(FAKE_USER_ID), "admin")
    with TestClient(_make_me_app()) as c:
        r = c.get("/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert r.json()["username"] == "admin"


def test_current_user_missing_header_401(monkeypatch):
    monkeypatch.setattr("app.db.get_user_by_id", lambda pool, uid: FAKE_USER)
    monkeypatch.setattr(settings, "jwt_secret", TEST_SECRET)
    with TestClient(_make_me_app()) as c:
        r = c.get("/me")
    assert r.status_code == 401


def test_current_user_disabled_401(monkeypatch):
    disabled = {**FAKE_USER, "disabled": True}
    monkeypatch.setattr("app.db.get_user_by_id", lambda pool, uid: disabled)
    monkeypatch.setattr(settings, "jwt_secret", TEST_SECRET)
    token = create_access_token(uuid.UUID(FAKE_USER_ID), "admin")
    with TestClient(_make_me_app()) as c:
        r = c.get("/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 401
    assert "禁用" in r.json()["detail"]


def test_current_user_expired_token_401(monkeypatch):
    monkeypatch.setattr("app.db.get_user_by_id", lambda pool, uid: FAKE_USER)
    monkeypatch.setattr(settings, "jwt_secret", TEST_SECRET)
    now = datetime.now(timezone.utc)
    expired = jwt.encode(
        {"sub": FAKE_USER_ID, "username": "admin", "iat": now - timedelta(hours=2),
         "exp": now - timedelta(hours=1)},
        TEST_SECRET,
        algorithm="HS256",
    )
    with TestClient(_make_me_app()) as c:
        r = c.get("/me", headers={"Authorization": f"Bearer {expired}"})
    assert r.status_code == 401
    assert "过期" in r.json()["detail"]
