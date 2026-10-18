"""RAG 转发器：Agent 作为 BFF 转发文档管理 REST 调用到 RAG 服务。

设计动机（见 architecture.md §12）：
- 前端不再直连 RAG，统一走 agent 中转；
- X-User-ID 由调用方（路由）从 JWT 验签得到后传入（可信身份，前端无法伪造）；
- X-API-Key 只在本进程持有（build_agent 注入），浏览器永远接触不到；
- 上传等大请求用独立超时（对齐 RAG 侧 write_timeout 300s），不复用工具链的 30s。
"""
import logging
from typing import Any, Dict

import requests

logger = logging.getLogger(__name__)

# 普通接口超时（秒）：列表/详情/删除/重分块
DEFAULT_TIMEOUT = 30
# 上传接口超时（秒）：大文件传输 + RAG 接收落盘，对齐 RAG write_timeout_sec=300
UPLOAD_TIMEOUT = 300


class RagProxyError(RuntimeError):
    """RAG 转发失败的统一异常：携带 HTTP 状态码 + 响应体。"""

    def __init__(self, status_code: int, body: Any, message: str = ""):
        self.status_code = status_code
        self.body = body
        super().__init__(message or f"RAG 请求失败: HTTP {status_code}")


class RagProxy:
    """转发器：把 agent 的文档管理 REST 调用转发到 RAG，注入可信身份与鉴权头。"""

    def __init__(self, session: requests.Session, base_url: str, api_key: str):
        self.session = session
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key  # 只在服务端持有，不随请求下发

    # ---------- 内部：构造请求 ----------

    def _headers(self, user_id: str) -> Dict[str, str]:
        h = {"X-User-ID": user_id}
        if self.api_key:
            h["X-API-Key"] = self.api_key
        return h

    def _url(self, path: str) -> str:
        return self.base_url + path

    def _send(
        self,
        method: str,
        path: str,
        user_id: str,
        *,
        json: Any = None,
        files: Any = None,
        data: Any = None,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> Any:
        """执行请求；非 2xx 抛 RagProxyError（含状态码 + body），由路由转 HTTP 响应。"""
        try:
            resp = self.session.request(
                method,
                self._url(path),
                headers=self._headers(user_id),
                json=json,
                files=files,
                data=data,
                timeout=timeout,
            )
        except requests.Timeout as e:
            raise RagProxyError(504, None, "RAG 服务响应超时") from e
        except requests.ConnectionError as e:
            raise RagProxyError(502, None, "无法连接 RAG 服务") from e

        if not resp.ok:
            raise RagProxyError(resp.status_code, self._safe_body(resp))

        # 204 等无 body 场景
        if resp.content:
            try:
                return resp.json()
            except ValueError:
                return resp.text
        return None

    @staticmethod
    def _safe_body(resp: requests.Response) -> Any:
        """尽力解析错误响应体（RAG 错误多为 {error: "..."}）。"""
        try:
            return resp.json()
        except ValueError:
            return resp.text

    # ---------- 对外方法（与 RAG 文档管理接口一一对应） ----------

    def check_health(self) -> Any:
        """RAG 健康检查：/health 豁免鉴权（不要求 X-API-Key / X-User-ID），短超时。

        失败不吞异常，由调用方（health.py）捕获并转成真实错误信息。
        """
        resp = self.session.get(self._url("/api/v1/health"), timeout=3)
        resp.raise_for_status()
        return resp.json()

    def list_documents(self, user_id: str) -> Any:
        return self._send("GET", "/api/v1/documents", user_id)

    def upload_documents(self, user_id: str, files: list, data: Dict[str, str]) -> Any:
        """上传：files 为 [(field_name, (filename, fileobj, content_type)), ...]，大超时。"""
        return self._send(
            "POST", "/api/v1/documents", user_id, files=files, data=data, timeout=UPLOAD_TIMEOUT
        )

    def get_document(self, user_id: str, document_id: str) -> Any:
        return self._send("GET", f"/api/v1/documents/{document_id}", user_id)

    def rechunk_document(self, user_id: str, document_id: str, body: Dict[str, Any]) -> Any:
        return self._send(
            "POST", f"/api/v1/documents/{document_id}/rechunk", user_id, json=body
        )

    def delete_document(self, user_id: str, document_id: str) -> Any:
        return self._send("DELETE", f"/api/v1/documents/{document_id}", user_id)
