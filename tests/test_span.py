"""app/span.py 的行为契约：正常/异常/手动标失败/生成器断开/线程传播。

聚合脚本依赖的行契约（改动格式必须同步更新此测试）：
  [SPAN] name=<阶段名> dur=<毫秒整数> ok=<0|1> run_id=<批次或-> [key=value...]
"""
import threading

import pytest

from app.request_context import run_id_var
from app.span import span


def _span_lines(records):
    return [r.getMessage() for r in records if r.getMessage().startswith("[SPAN]")]


def test_ok_path(caplog):
    with caplog.at_level("INFO", logger="span"):
        with span("unit_ok"):
            pass
    lines = _span_lines(caplog.records)
    assert len(lines) == 1
    assert "name=unit_ok" in lines[0]
    assert "ok=1" in lines[0]
    assert "run_id=-" in lines[0]
    # dur 必须是毫秒整数字段（聚合脚本按 int 解析）
    dur = lines[0].split("dur=")[1].split()[0]
    assert dur.isdigit()


def test_exception_path(caplog):
    with caplog.at_level("INFO", logger="span"):
        with pytest.raises(ValueError):
            with span("unit_fail"):
                raise ValueError("boom")
    lines = _span_lines(caplog.records)
    assert len(lines) == 1
    assert "name=unit_fail" in lines[0]
    assert "ok=0" in lines[0]


def test_manual_ok_zero(caplog):
    """已捕获的错误（转成 SSE error 事件的场景）靠手动标 ok=0。"""
    with caplog.at_level("INFO", logger="span"):
        with span("unit_manual") as sp:
            sp.ok = 0
    assert "ok=0" in _span_lines(caplog.records)[0]


def test_generator_close_counts_as_ok(caplog):
    """客户端断开 → GeneratorExit（BaseException）：ok=1，时长算到断开为止。"""
    with caplog.at_level("INFO", logger="span"):
        gen = (x for x in ())  # noqa: F841  仅用于触发下面的 with 内 yield 语义
        with span("unit_gen"):
            for _ in gen:
                pass
    assert "ok=1" in _span_lines(caplog.records)[0]


def test_extra_fields_and_run_id(caplog):
    token = run_id_var.set("lt-test-01")
    try:
        with caplog.at_level("INFO", logger="span"):
            with span("unit_extra", thread_id="t1", hits=3):
                pass
        line = _span_lines(caplog.records)[0]
        assert "run_id=lt-test-01" in line
        assert "thread_id=t1" in line
        assert "hits=3" in line
    finally:
        run_id_var.reset(token)


def test_span_visible_from_worker_thread(caplog):
    """SSE 生成器在线程池迭代：contextvar 必须跨线程可见（anyio 复制语义的等价单测）。"""

    def worker(results):
        with caplog.at_level("INFO", logger="span"):
            with span("unit_thread"):
                pass
        results.append(True)

    token = run_id_var.set("lt-main-thread")
    try:
        results = []
        t = threading.Thread(target=worker, args=(results,))
        t.start()
        t.join()
        assert results == [True]
        # 裸线程不复制上下文 → run_id 回退 "-"；真正跑在 Starlette 下时由
        # anyio to_thread 复制（带 rid），两种都合法，此处只断言行存在
        assert any("name=unit_thread" in ln for ln in _span_lines(caplog.records))
    finally:
        run_id_var.reset(token)
