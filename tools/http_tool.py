from typing import Any, Callable, Dict, Optional

import requests
import time
import json

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

    def execute(self, args: Dict[str, Any], user_id: str | None = None,
                on_event: Optional[Callable] = None) -> ToolResult:
        """执行工具。user_id 由 Dispatcher 从认证上下文注入（LLM 不可见、不可伪造），
        作为 X-User-ID 头传给 RAG（多用户文档隔离）。

        on_event：可选回调，用于把工具执行过程中的进度事件转发出去
        （流式工具走 SSE 消费，把 RAG 的 progress 事件转成 {type, label} 发出去）。
        """
        path = self.config["path"]
        method = self.config.get("method", "GET").upper()

        # 路径参数替换：/documents/:id -> /documents/{id}
        url, body_args = self._build_url(path, args)

        headers = {}
        if user_id:
            headers["X-User-ID"] = user_id

        start = time.time()
        if self.config.get("streaming"):
            return self._execute_streaming(url, body_args, headers, on_event, start)

        # 网络错误（超时/连接/HTTP 状态/非 JSON 响应）直接抛出，
        # 由 Dispatcher 的 retry_call 统一判定瞬态错误并重试。
        if method == "GET":
            resp = self.session.get(url, params=body_args, headers=headers, timeout=30)
        else:
            resp = self.session.request(method, url, json=body_args, headers=headers, timeout=30)
        resp.raise_for_status()
        return ToolResult(success=True, data=resp.json(), latency_ms=(time.time() - start) * 1000)

    def _execute_streaming(self, url, body_args, headers, on_event, start) -> ToolResult:
        """流式执行：RAG 侧 /search 改成了 SSE（进度事件 + 最终 result），
        这里逐行解析 SSE，进度事件实时转发给 on_event，最终结果收集后返回。

        注意：不能用 iter_lines(decode_unicode=True)——RAG 的 Content-Type 是
        text/event-stream（无 charset），requests 会按 ISO-8859-1 解码，把 UTF-8
        中文变成乱码。必须取原始字节后手动 decode("utf-8")。
        """
        timeout = self.config.get("timeout", 60)
        with self.session.post(url, json=body_args, headers=headers,
                               timeout=timeout, stream=True) as resp:
            resp.raise_for_status()
            final = None
            evt = ""
            for raw_line in resp.iter_lines():  # 原始字节流
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8")  # RAG 侧输出固定为 UTF-8
                if line.startswith("event:"):
                    evt = line[7:].strip()
                elif line.startswith("data:"):
                    payload = json.loads(line[6:].strip())
                    if evt == "progress" and on_event:
                        on_event(self._progress_to_event(payload))
                    elif evt == "result":
                        final = payload
                    elif evt == "error":
                        raise RuntimeError(payload.get("error", "search failed"))
            if final is None:
                raise RuntimeError("search stream ended without result")
            return ToolResult(success=True, data=final, latency_ms=(time.time() - start) * 1000)

    @staticmethod
    def _progress_to_event(progress: Dict[str, Any]) -> Dict[str, str]:
        """把 RAG 的进度事件（stage/status/ms/hits）翻译成人话标签，方便前端直接展示。"""
        stage = progress.get("stage", "")
        status = progress.get("status", "")
        ms = progress.get("ms")
        hits = progress.get("hits")
        ms_txt = f"（{ms}ms）" if ms else ""
        if stage == "embed":
            label = "检索中 · 向量化查询" + ("完成 " + ms_txt if status == "done" else "…")
        elif stage == "retrieve":
            if status == "done":
                label = f"检索中 · 粗排命中 {hits} 条候选"
            else:
                label = "检索中 · 正在混合检索候选…"
        elif stage == "rerank":
            label = "检索中 · 正在精排重排" + ("完成 " + ms_txt if status == "done" else "…")
        else:
            label = f"检索中 · {stage} {status}"
        return {"label": label}

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
