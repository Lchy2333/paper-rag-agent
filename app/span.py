"""span：请求内关键阶段的秒表，一行日志一个 span。

行格式：[SPAN] name=<阶段名> dur=<毫秒> ok=<0|1> run_id=<压测批次或-> [附加key=value]

设计约定：
- 聚合脚本按 name 分组算阶段 P95；run_id 切片隔离压测流量（免得按 rid join [ACCESS]）；
- rid 不写在字段里——root handler 的 RequestIdFilter 自动盖（见 app/main.py）；
- dur 一律进程内 perf_counter 本地算（跨服务不相减时间戳）；
- 判定一个阶段要不要 span 的标准：这步慢了我会不会去优化它。
"""
import logging
import time
from contextlib import contextmanager

from app.request_context import run_id_var

logger = logging.getLogger("span")


class _SpanHandle:
    """with span(...) as sp 的返回值：允许调用方把「已捕获的失败」标进 ok。

    默认 ok=1；异常上抛时自动置 0。两类场景需要手动改：
    - SSE 生成器内部 except 捕获异常后转成 error 事件发给前端（进程不崩，但请求失败了）；
    其余场景不需要碰它。
    """

    __slots__ = ("ok",)

    def __init__(self) -> None:
        self.ok = 1


@contextmanager
def span(name: str, **fields):
    """with span("llm_call"): ... → 出作用域时打一行 [SPAN]。

    - 异常不吞不翻译：只把 ok 置 0 后原样上抛，错误处理仍归调用方；
    - 客户端断开时生成器收到 GeneratorExit（BaseException，不是 Exception）：
      ok 保持 1、时长算到断开为止——用户主动取消不算服务失败；
    - fields 拼在行尾（key=value），如 thread_id=xxx。
    """
    handle = _SpanHandle()
    start = time.perf_counter()
    try:
        yield handle
    except Exception:
        handle.ok = 0
        raise
    finally:
        dur_ms = int((time.perf_counter() - start) * 1000)
        extra = "".join(f" {k}={v}" for k, v in fields.items())
        logger.info(
            "[SPAN] name=%s dur=%d ok=%d run_id=%s%s",
            name, dur_ms, handle.ok, run_id_var.get(), extra,
        )
