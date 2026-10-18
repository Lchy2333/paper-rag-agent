"""文档管理转发层测试：验证路由注入可信身份 + 错误映射 + 删除审计。

不依赖真实 RAG / DB：
- build_agent 被 monkeypatch 成假对象（不连库）；
- app.state.rag_proxy 由 fixture 注入假 RagProxy（记录调用参数）；
- app.db.add_audit_log 被 monkeypatch 记录审计调用。
"""
import pytest
from fastapi.testclient import TestClient
import io

from app import db
from app.auth import get_current_user
from app.main import app


class _FakePool:
    def connection(self):
        return None

    def close(self):
        pass


class _FakeProxy:
    """记录调用、按预置结果返回的假 RagProxy。"""

    def __init__(self):
        self.calls = []
        self.responses = {}
        self.errors = {}

    def _record(self, method, path, user_id, **kw):
        self.calls.append({"method": method, "path": path, "user_id": user_id, **kw})
        err = self.errors.get((method, path))
        if err:
            raise err
        return self.responses.get((method, path))

    def list_documents(self, user_id):
        return self._record("GET", "/api/v1/documents", user_id)

    def upload_documents(self, user_id, files, data):
        return self._record("POST", "/api/v1/documents", user_id, files=files, data=data)

    def get_document(self, user_id, document_id):
        return self._record("GET", f"/api/v1/documents/{document_id}", user_id)

    def rechunk_document(self, user_id, document_id, body):
        return self._record("POST", f"/api/v1/documents/{document_id}/rechunk", user_id, body=body)

    def delete_document(self, user_id, document_id):
        return self._record("DELETE", f"/api/v1/documents/{document_id}", user_id)


@pytest.fixture()
def fake_env(monkeypatch):
    """真实 app + 假 build_agent + 假 RagProxy + 假审计。"""
    proxy = _FakeProxy()
    audit_calls = []

    monkeypatch.setattr("app.main.build_agent", lambda: (None, None, None, _FakePool(), None, None))
    monkeypatch.setattr(db, "add_audit_log", lambda pool, uid, tid, action, payload: audit_calls.append(
        {"user_id": uid, "action": action, "payload": payload}
    ))
    # 覆盖认证依赖：固定假用户身份
    app.dependency_overrides[get_current_user] = lambda: {
        "id": "11111111-2222-3333-4444-555555555555",
        "username": "tester",
    }
    with TestClient(app) as c:
        c.app.state.rag_proxy = proxy
        yield c, proxy, audit_calls
    app.dependency_overrides.clear()


def test_list_documents_injects_trusted_user(fake_env):
    client, proxy, _ = fake_env
    proxy.responses[("GET", "/api/v1/documents")] = {"documents": []}

    r = client.get("/api/documents")
    assert r.status_code == 200
    assert r.json() == {"documents": []}
    # 注入的身份 = JWT 里的 user_id（不是前端传的）
    assert proxy.calls[-1]["user_id"] == "11111111-2222-3333-4444-555555555555"


def test_delete_document_forwards_and_audits(fake_env):
    client, proxy, audit_calls = fake_env
    proxy.responses[("DELETE", "/api/v1/documents/doc-1")] = {"status": "deleted"}

    r = client.delete("/api/documents/doc-1")
    assert r.status_code == 200
    assert proxy.calls[-1]["method"] == "DELETE"
    assert proxy.calls[-1]["user_id"] == "11111111-2222-3333-4444-555555555555"
    # 审计留痕
    assert audit_calls == [{
        "user_id": "11111111-2222-3333-4444-555555555555",
        "action": "document_delete",
        "payload": {"document_id": "doc-1"},
    }]


def test_get_document_404_maps_error(fake_env):
    client, proxy, _ = fake_env
    from core.rag_proxy import RagProxyError
    proxy.errors[("GET", "/api/v1/documents/nope")] = RagProxyError(404, {"error": "document not found"})

    r = client.get("/api/documents/nope")
    assert r.status_code == 404
    assert r.json()["detail"] == "document not found"


def test_rechunk_body_passthrough(fake_env):
    client, proxy, _ = fake_env
    proxy.responses[("POST", "/api/v1/documents/doc-1/rechunk")] = {"document": {"id": "doc-1"}}

    r = client.post("/api/documents/doc-1/rechunk", json={"chunk_size": 400})
    assert r.status_code == 200
    assert proxy.calls[-1]["body"] == {"chunk_size": 400}


def test_rag_unreachable_502(fake_env):
    client, proxy, _ = fake_env
    from core.rag_proxy import RagProxyError
    proxy.errors[("GET", "/api/v1/documents")] = RagProxyError(502, None, "无法连接 RAG 服务")

    r = client.get("/api/documents")
    assert r.status_code == 502
    assert r.json()["detail"] == "无法连接 RAG 服务"


def test_upload_requires_field_name_file(fake_env):
    """前端 FormData 字段名是 `file`（与 RAG 侧 form.File["file"] 一致）。

    回归：参数曾误命名 `files`，FastAPI 按参数名匹配 multipart 字段 → 422。
    字段名必须叫 `file`，否则前端上传直接 422 Unprocessable Entity。
    """
    client, proxy, _ = fake_env
    proxy.responses[("POST", "/api/v1/documents")] = {
        "success_count": 1, "duplicate_count": 0, "failed_count": 0, "results": []
    }

    pdf = b"%PDF-1.4 fake pdf content"
    # 正确字段名 file
    r = client.post(
        "/api/documents",
        files=[("file", ("paper.pdf", io.BytesIO(pdf), "application/pdf"))],
    )
    assert r.status_code == 200
    assert proxy.calls[-1]["method"] == "POST"
    assert proxy.calls[-1]["user_id"] == "11111111-2222-3333-4444-555555555555"
    # 字段名必须叫 file：转发到 RAG 的 files 结构第一项 field_name 也是 file
    assert proxy.calls[-1]["files"][0][0] == "file"

    # 错误字段名 files（老 bug）：应 422 校验失败
    r_bad = client.post(
        "/api/documents",
        files=[("files", ("paper.pdf", io.BytesIO(pdf), "application/pdf"))],
    )
    assert r_bad.status_code == 422
