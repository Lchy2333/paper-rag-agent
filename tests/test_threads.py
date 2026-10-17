"""B2 测试：线程归属绑定 + 改名/置顶 + 删除 + 审计。

全部 mock 数据库访问（app.db 模块函数）与 graph，不依赖真实 PG / LLM。
多用户隔离核心断言：跨用户访问 /chat /approve /threads → 403。
"""
from datetime import datetime, timezone
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from app.auth import get_current_user
from app.main import app

USER_A = UUID("11111111-1111-1111-1111-111111111111")
USER_B = UUID("22222222-2222-2222-2222-222222222222")
THREAD = "abc123thread"
TS = datetime(2026, 10, 17, 12, 0, 0, tzinfo=timezone.utc)


class _Msg:
    def __init__(self, content):
        self.content = content


class _FakeGraph:
    def __init__(self):
        self.invoke_calls = []
        self.deleted = []

    def get_state(self, config):
        class _Snap:
            interrupts = []
            values = {"messages": []}

        return _Snap()

    def invoke(self, *args, **kwargs):
        self.invoke_calls.append((args, kwargs))
        # 真实 LangGraph：无中断时不带 __interrupt__ key（而非空列表）
        return {"messages": [_Msg("ok")]}

    @property
    def checkpointer(self):
        return self

    def delete_thread(self, thread_id):
        self.deleted.append(thread_id)


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
def ctx(monkeypatch):
    """app + 假 build_agent + FakeGraph；graph 挂在 app.state 供路由使用。"""
    monkeypatch.setattr("app.main.build_agent", lambda: (None, None, None, _FakePool()))
    g = _FakeGraph()
    with TestClient(app) as c:
        app.state.graph = g
        yield c, g


def _as_user(user_id):
    app.dependency_overrides[get_current_user] = lambda: {
        "id": user_id, "username": "u", "disabled": False,
    }
    return lambda: None


def _clear_user():
    app.dependency_overrides.pop(get_current_user, None)


def _thread_row(owner=USER_A):
    return {
        "id": THREAD, "user_id": owner, "title": "t", "pinned": False,
        "created_at": TS, "updated_at": TS,
    }


# ---------- /api/chat 归属校验 ----------

def test_chat_new_thread_creates(ctx, monkeypatch):
    c, g = ctx
    _as_user(USER_A)
    created = []
    monkeypatch.setattr("app.db.create_thread",
                        lambda pool, tid, uid: created.append((tid, uid)))
    try:
        r = c.post("/api/chat", json={"query": "hi"})
    finally:
        _clear_user()
    assert r.status_code == 200
    assert len(created) == 1
    assert created[0][1] == USER_A  # 新会话绑定当前用户
    assert r.json()["thread_id"] == created[0][0]
    assert g.invoke_calls  # 走到了 graph


def test_chat_unknown_thread_404(ctx, monkeypatch):
    c, g = ctx
    _as_user(USER_A)
    monkeypatch.setattr("app.db.get_thread", lambda pool, tid: None)
    try:
        r = c.post("/api/chat", json={"query": "hi", "thread_id": THREAD})
    finally:
        _clear_user()
    assert r.status_code == 404


def test_chat_other_users_thread_403(ctx, monkeypatch):
    c, g = ctx
    _as_user(USER_A)
    monkeypatch.setattr("app.db.get_thread", lambda pool, tid: _thread_row(owner=USER_B))
    try:
        r = c.post("/api/chat", json={"query": "hi", "thread_id": THREAD})
    finally:
        _clear_user()
    assert r.status_code == 403


def test_chat_own_thread_ok(ctx, monkeypatch):
    c, g = ctx
    _as_user(USER_A)
    monkeypatch.setattr("app.db.get_thread", lambda pool, tid: _thread_row(owner=USER_A))
    try:
        r = c.post("/api/chat", json={"query": "hi", "thread_id": THREAD})
    finally:
        _clear_user()
    assert r.status_code == 200
    assert g.invoke_calls
    assert g.invoke_calls[0][1]["config"]["configurable"]["thread_id"] == THREAD


# ---------- /api/approve 归属校验 + 审计 ----------

def test_approve_other_users_thread_403(ctx, monkeypatch):
    c, g = ctx
    _as_user(USER_A)
    monkeypatch.setattr("app.db.get_thread", lambda pool, tid: _thread_row(owner=USER_B))
    try:
        r = c.post("/api/approve", json={"thread_id": THREAD, "approved": True})
    finally:
        _clear_user()
    assert r.status_code == 403
    assert not g.invoke_calls  # 归属不过，绝不执行审批恢复


def test_approve_unknown_thread_404(ctx, monkeypatch):
    c, g = ctx
    _as_user(USER_A)
    monkeypatch.setattr("app.db.get_thread", lambda pool, tid: None)
    try:
        r = c.post("/api/approve", json={"thread_id": THREAD, "approved": True})
    finally:
        _clear_user()
    assert r.status_code == 404


def test_approve_ok_writes_audit(ctx, monkeypatch):
    c, g = ctx
    _as_user(USER_A)
    audit = []
    monkeypatch.setattr("app.db.get_thread", lambda pool, tid: _thread_row(owner=USER_A))
    monkeypatch.setattr(
        "app.db.add_audit_log",
        lambda pool, uid, tid, action, payload: audit.append((uid, tid, action, payload)),
    )
    try:
        r = c.post("/api/approve", json={"thread_id": THREAD, "approved": True})
    finally:
        _clear_user()
    assert r.status_code == 200
    assert r.json()["status"] == "done"
    assert audit == [(str(USER_A), THREAD, "approve", {"approved": True})]


# ---------- GET /api/threads ----------

def test_list_threads(ctx, monkeypatch):
    c, g = ctx
    _as_user(USER_A)
    rows = [
        {"id": "t1", "title": "置顶", "pinned": True, "created_at": TS, "updated_at": TS},
        {"id": "t2", "title": "", "pinned": False, "created_at": TS, "updated_at": TS},
    ]
    monkeypatch.setattr("app.db.list_threads", lambda pool, uid: rows)
    try:
        r = c.get("/api/threads")
    finally:
        _clear_user()
    assert r.status_code == 200
    body = r.json()
    assert [t["thread_id"] for t in body] == ["t1", "t2"]
    assert body[0]["pinned"] is True


# ---------- PATCH /api/threads/{id}（改名 + 置顶） ----------

def test_update_thread_rename_and_pin(ctx, monkeypatch):
    c, g = ctx
    _as_user(USER_A)
    called = {}
    monkeypatch.setattr("app.db.get_thread", lambda pool, tid: _thread_row(owner=USER_A))
    monkeypatch.setattr(
        "app.db.update_thread",
        lambda pool, tid, title=None, pinned=None: called.update(tid=tid, title=title, pinned=pinned)
        or {**_thread_row(owner=USER_A), "title": title, "pinned": pinned},
    )
    try:
        r = c.patch(f"/api/threads/{THREAD}", json={"title": "新标题", "pinned": True})
    finally:
        _clear_user()
    assert r.status_code == 200
    body = r.json()
    assert body["title"] == "新标题"
    assert body["pinned"] is True
    assert called == {"tid": THREAD, "title": "新标题", "pinned": True}


def test_update_thread_other_user_403(ctx, monkeypatch):
    c, g = ctx
    _as_user(USER_A)
    monkeypatch.setattr("app.db.get_thread", lambda pool, tid: _thread_row(owner=USER_B))
    try:
        r = c.patch(f"/api/threads/{THREAD}", json={"title": "hack"})
    finally:
        _clear_user()
    assert r.status_code == 403


def test_update_thread_no_fields_422(ctx, monkeypatch):
    c, g = ctx
    _as_user(USER_A)
    monkeypatch.setattr("app.db.get_thread", lambda pool, tid: _thread_row(owner=USER_A))
    try:
        r = c.patch(f"/api/threads/{THREAD}", json={})
    finally:
        _clear_user()
    assert r.status_code == 422


# ---------- DELETE /api/threads/{id} ----------

def test_delete_thread_ok_cleans_checkpoints(ctx, monkeypatch):
    c, g = ctx
    _as_user(USER_A)
    monkeypatch.setattr("app.db.get_thread", lambda pool, tid: _thread_row(owner=USER_A))
    try:
        r = c.delete(f"/api/threads/{THREAD}")
    finally:
        _clear_user()
    assert r.status_code == 200
    assert g.deleted == [THREAD]  # checkpointer.delete_thread 被调用，快照已清理
    assert r.json()["status"] == "deleted"


def test_delete_thread_other_user_403(ctx, monkeypatch):
    c, g = ctx
    _as_user(USER_A)
    monkeypatch.setattr("app.db.get_thread", lambda pool, tid: _thread_row(owner=USER_B))
    try:
        r = c.delete(f"/api/threads/{THREAD}")
    finally:
        _clear_user()
    assert r.status_code == 403
    assert g.deleted == []
