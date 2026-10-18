"""健康检查：分层返回，供编排 healthcheck 与前端状态灯使用。

设计（见 architecture.md §13）：
- status（存活）：只由 agent 进程 + database 决定，供编排 healthcheck 判断"是否重启"。
  LLM / RAG 是外部依赖，异常不应触发重启（重启救不了外部服务）；
- checks（就绪）：database / rag / llm 各自状态，供前端状态灯展示明细。
"""
import logging
import time

from fastapi import APIRouter, Request

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

# 子探活超时（秒）：不能让某个依赖卡住拖慢 /health
DEPENDENCY_TIMEOUT = 3


def _check_database(pool) -> dict:
    """DB 探活：SELECT 1。硬依赖（agent 自己的持久化底座）。"""
    try:
        start = time.time()
        with pool.connection() as conn:
            conn.execute("SELECT 1")
        return {"status": "ok", "latency_ms": round((time.time() - start) * 1000)}
    except Exception as e:  # noqa: BLE001
        logger.warning("health_db_failed: %s", str(e)[:200])
        return {"status": "error", "error": str(e)[:200]}


def _check_rag(request: Request) -> dict:
    """RAG 探活：转发 RAG /health（豁免鉴权）。就绪依赖，不触发重启。"""
    start = time.time()
    try:
        resp = request.app.state.rag_proxy.check_health()
        latency_ms = round((time.time() - start) * 1000)
        ok = resp is not None and resp.get("status") == "ok"
        if not ok:
            return {"status": "error", "latency_ms": latency_ms, "error": f"RAG /health 返回异常: {resp}"}
        return {"status": "ok", "latency_ms": latency_ms}
    except Exception as e:  # noqa: BLE001
        latency_ms = round((time.time() - start) * 1000)
        logger.warning("health_rag_failed: %s", str(e)[:200])
        return {"status": "error", "latency_ms": latency_ms, "error": str(e)[:200]}


def _check_llm(request: Request) -> dict:
    """LLM 探活（就绪语义）：被动信号优先 + 节流主动探针兜底。"""
    tracker = request.app.state.llm_tracker
    if tracker is None:
        return {"status": "unknown"}
    global_llm = (
        request.app.state.settings.llm_base_url,
        request.app.state.settings.llm_api_key,
        request.app.state.settings.llm_model,
    )
    result = tracker.status(global_llm)
    return result


@router.get("/health")
def health(request: Request):
    """分层健康检查：status 供编排（存活），checks 供前端（就绪明细）。"""
    # 存活判定：agent 进程必然活着（能响应即 ok）+ database
    db = _check_database(request.app.state.pool)
    checks = {"database": db}

    # 就绪明细（不影响 status，不触发重启）
    checks["rag"] = _check_rag(request)
    checks["llm"] = _check_llm(request)

    status = "error" if db["status"] == "error" else "ok"
    return {"status": status, "checks": checks}
