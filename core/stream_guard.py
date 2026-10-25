"""把阻塞式同步 stream 变成可控迭代（首 token 守护）。

被两处复用：
- agent/nodes.py 的 call_llm：真实对话流式生成（业务层首 token 守护，FIRST_TOKEN_TIMEOUT=30s）；
- core/llm_probe.check_llm_stream：健康检查 / BYOK 的 stream 通道就绪探针。

动机（见 architecture.md §15）：同步 `stream()` 卡在 __next__() 时外层无法干预——
上游挂起时既没有超时判定，取消检查也进不去（循环体执行不到）。因此把流的消费放进
daemon 工作线程 + 队列，主循环在 q.get(timeout=...) 上等待：
- 首 token：限期内无数据 → 抛 TimeoutError（瞬态，调用方可按 is_transient 重试）；
- 首 token 后不限时：生成中途的哑掉由客户端 read 超时（llm_timeout）兜底；
- 取消：每收到一个 chunk 检查 cancel_event，置位即抛 GenerationCancelled；
- 流内异常：worker 透传到主循环原样重抛。
工作线程 daemon=True，即使被挂起的流卡住也不阻塞进程退出。
"""
import queue
import threading
import time


class GenerationCancelled(Exception):
    """用户主动中断生成：guard 抛出，/chat/stream 捕获后以 cancel 事件收尾。

    必须单独捕获（不能走 classify_exception 兜底），否则会被当成 internal_error。
    """


def iter_stream_with_guard(stream, first_token_timeout: float, cancel_event=None):
    q: queue.Queue = queue.Queue(maxsize=16)

    def worker():
        try:
            for chunk in stream:
                q.put(("chunk", chunk))
        except BaseException as e:  # 流内抛错（含超时/5xx）透传给主循环
            q.put(("error", e))
        finally:
            q.put(("done", None))

    threading.Thread(target=worker, daemon=True, name="llm-stream-worker").start()

    deadline = time.monotonic() + first_token_timeout
    first = True
    while True:
        wait = deadline - time.monotonic() if first else None
        try:
            item = q.get(timeout=max(0.0, wait) if wait is not None else None)
        except queue.Empty:
            raise TimeoutError(
                "首 token 超时（%.0fs 内无数据）" % first_token_timeout
                if first else "LLM 流空闲超时（相邻 token 间隔过长）")
        kind, payload = item
        if kind == "error":
            raise payload
        if kind == "done":
            return
        if cancel_event is not None and cancel_event.is_set():
            raise GenerationCancelled()
        yield payload
        first = False
