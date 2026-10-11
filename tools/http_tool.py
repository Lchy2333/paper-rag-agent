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

        try:
            start = time.time()
            if method == "GET":
                resp = self.session.get(url, params=body_args, timeout=30)
            else:
                resp = self.session.request(method, url, json=body_args, timeout=30)
            resp.raise_for_status()
            return ToolResult(success=True, data=resp.json(), latency_ms=(time.time() - start) * 1000)
        except requests.Timeout:
            return ToolResult(success=False, error=f"request timeout: {url}")
        except requests.RequestException as e:
            return ToolResult(success=False, error=str(e))

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
