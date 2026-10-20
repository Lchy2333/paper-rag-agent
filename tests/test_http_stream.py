"""HttpTool 流式 SSE 消费的回归测试。

关键：RAG 返回的 Content-Type 是 text/event-stream（无 charset），requests 会按
ISO-8859-1 解码，若用 decode_unicode=True 会把 UTF-8 中文变乱码（之前线上就是这么
坏的）。这里用真实本地 HTTP 服务验证：中文内容必须原样保留。
"""
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests

from tools.http_tool import HttpTool


class _SSEHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = (
            'event: progress\ndata: {"stage":"embed","status":"done","ms":10}\n\n'
            'event: result\ndata: {"results":[{"content":"钢管混凝土受力机制"}],"total_chunks":1}\n\n'
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def test_streaming_tool_decodes_utf8_chinese():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _SSEHandler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        tool = HttpTool(
            requests.Session(),
            f"http://127.0.0.1:{srv.server_address[1]}",
            {"path": "/search", "method": "POST", "streaming": True, "timeout": 10},
            "search_rag",
        )
        events = []
        result = tool.execute({"query": "钢管混凝土"}, user_id=None, on_event=events.append)

        assert result.success, f"执行失败: {result.error}"
        # 中文必须原样保留（不能被 ISO-8859-1 转码成乱码）
        assert result.data["results"][0]["content"] == "钢管混凝土受力机制"
        assert result.data["total_chunks"] == 1
        # progress 事件被翻译成人话标签转发
        assert events and "检索中" in events[0]["label"]
    finally:
        srv.shutdown()


def test_streaming_tool_missing_result_raises():
    """SSE 流没等到 result 就结束（如 RAG 返回的是旧版 JSON）→ 抛错（由 Dispatcher 兜底转失败）。"""
    import pytest

    class _JsonHandler(_SSEHandler):
        def do_POST(self):
            body = '{"results":[]}'.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), _JsonHandler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        tool = HttpTool(
            requests.Session(),
            f"http://127.0.0.1:{srv.server_address[1]}",
            {"path": "/search", "method": "POST", "streaming": True, "timeout": 10},
            "search_rag",
        )
        with pytest.raises(RuntimeError, match="ended without result"):
            tool.execute({"query": "q"}, user_id=None)
    finally:
        srv.shutdown()
