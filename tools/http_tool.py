from typing import Any, Dict

import requests
import time

from core.tool import ToolBase, ToolResult


class HttpTool(ToolBase):
    """通用 HTTP 工具：行为完全由 RAGtools.json 中的配置驱动。一个配置 = 一个工具。无需再写子类。"""
    def __init__(self, session: requests.Session, base_url: str, config: Dict[str, Any], name: str):
        self.session = session
        self.base_url = base_url.rstrip("/")
        self.config = config
        self._name = name

    def name(self) -> str:
        return self._name

    @property
    def requires_approval(self) -> bool:
        """是否需要人工审批（RAGtools.json 中 requires_approval: true 的工具）。"""
        return self.config.get("requires_approval", False)

    def build_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self._name,
                "description": self.config.get("description", ""),
                "parameters": self.config.get("schema", {}),
            },
        }

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        path = self.config["path"]
        method = self.config.get("method", "GET").upper()

        # 路径参数替换：/documents/:id -> /documents/{id}
        url, body_args = self._build_url(path, args)

        # 网络错误（超时/连接/HTTP 状态/非 JSON 响应）直接抛出，
        # 由 Dispatcher 的 retry_call 统一判定瞬态错误并重试。
        start = time.time()
        if method == "GET":
            resp = self.session.get(url, params=body_args, timeout=30)
        else:
            resp = self.session.request(method, url, json=body_args, timeout=30)
        resp.raise_for_status()
        return ToolResult(success=True, data=resp.json(), latency_ms=(time.time() - start) * 1000)

    def _build_url(self, path: str, args: Dict[str, Any]) -> tuple:
        """替换 :param 路径参数，剩余参数作为 query/body。"""
        url = self.base_url + path
        body_args = dict(args)
        segments = url.split("/")
        for i, seg in enumerate(segments):
            if seg.startswith(":"):
                key = seg[1:]
                if key not in body_args:
                    raise ValueError(f"missing path parameter: {key}")
                segments[i] = str(body_args.pop(key))
        return "/".join(segments), body_args
