"""FastAPI 应用入口。

用 lifespan 一次性构建 client / registry / dispatcher / graph，
挂到 app.state，路由通过依赖注入按需取用（进程生命周期内只建一次）。
全局 exception_handler 兜底未捕获异常：分类响应 + 结构化日志（工具调用失败
已被 Dispatcher 转成 ToolMessage，不进这里）。
"""
import json
import logging
import sys
import time
from contextlib import asynccontextmanager

import psycopg
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from agent.graph import build_graph
from app.api import approve, auth, chat, documents, health, llm_check, threads
from app.config import settings
from app.errors import classify_exception
from core import Dispatcher, ToolRegistry
from core.llm_probe import LLMHealthTracker
from core.rag_proxy import RagProxy
from tools.http_tool import HttpTool

sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("paper-rag-agent")

# 访问日志：与 RAG（Gin）侧统一格式，走独立 logger 输出原始行（不被 basicConfig 前缀污染）：
# [ACCESS] 2026/10/25 - 06:17:40 | 200 | 17.851µs | 127.0.0.1 | GET "/health"
access_logger = logging.getLogger("access")
_access_handler = logging.StreamHandler(sys.stdout)
_access_handler.setFormatter(logging.Formatter("%(message)s"))
access_logger.addHandler(_access_handler)
access_logger.setLevel(logging.INFO)
access_logger.propagate = False


def _format_duration(seconds: float) -> str:
    """把秒格式化为 Go time.Duration 风格（17.851µs / 1.234ms / 2.345s）。"""

    def _trim(s: str) -> str:
        return s.rstrip("0").rstrip(".") if "." in s else s

    if seconds * 1_000_000 < 1000:
        return f"{_trim(f'{seconds * 1e6:.3f}')}µs"
    if seconds * 1_000 < 1000:
        return f"{_trim(f'{seconds * 1e3:.3f}')}ms"
    return f"{_trim(f'{seconds:.3f}')}s"


def build_agent():
    """构建 LLM + 工具 registry + dispatcher + LangGraph。"""
    registry = ToolRegistry()
    session = requests.Session()
    # 服务间鉴权：所有 HttpTool 请求自动携带 X-API-Key（RAG 侧校验）
    if settings.rag_api_key:
        session.headers["X-API-Key"] = settings.rag_api_key

    configs = json.loads(settings.tools_config.read_text(encoding="utf-8"))
    for name, cfg in configs.items():
        registry.register(HttpTool(session, settings.rag_base_url, cfg, name))

    dispatcher = Dispatcher(registry)
    llm = ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        timeout=settings.llm_timeout,      # 传输层超时：read 空闲上限，防"挂起"无限等待
        max_retries=settings.llm_max_retries,  # 0：重试收敛到业务层 call_llm，避免叠乘放大
    )
    # checkpointer 是 HITL 中断恢复 + 多轮会话的存档仓库；
    # 生产用 Postgres 持久化（独立库 paper_rag_agent），多副本共享会话状态，
    # 进程重启不丢对话。池内连接对齐 PostgresSaver 要求：autocommit + dict_row。
    pool = ConnectionPool(
        conninfo=settings.database_url,
        max_size=10,
        open=False,
        kwargs={"autocommit": True, "row_factory": dict_row},
    )
    pool.open()
    checkpointer = PostgresSaver(pool)
    checkpointer.setup()  # 自动建 checkpoints / checkpoint_blobs / checkpoint_writes
    # LLM 健康跟踪器：call_llm 记录真实调用成败 → /health 被动信号（零额外请求）
    llm_tracker = LLMHealthTracker()
    graph = build_graph(llm, dispatcher, checkpointer, llm_tracker=llm_tracker)
    # 文档管理转发器（BFF）：前端经 agent 中转访问 RAG，可信身份 + 服务端持有 API Key。
    # 复用同一个 session（已带 X-API-Key 默认头），rag_api_key 单独传入供显式引用。
    rag_proxy = RagProxy(session, settings.rag_base_url, settings.rag_api_key)
    return llm, dispatcher, graph, pool, rag_proxy, llm_tracker


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动校验：付费资源 / 生产关键配置缺失 → 拒绝启动（见 app/config.py）
    settings.ensure_runtime()
    llm, dispatcher, graph, pool, rag_proxy, llm_tracker = build_agent()
    app.state.llm = llm
    app.state.dispatcher = dispatcher
    app.state.graph = graph
    app.state.pool = pool
    app.state.rag_proxy = rag_proxy
    app.state.llm_tracker = llm_tracker
    app.state.settings = settings
    app.state.system_prompt = settings.system_prompt
    # 进程内"取消生成"注册表：thread_id → threading.Event（/chat/stream 创建，/chat/cancel 置位）
    app.state.cancels = {}
    try:
        yield
    finally:
        pool.close()


app = FastAPI(title="paper-rag-agent", lifespan=lifespan)
app.include_router(auth.router, prefix="/api", tags=["auth"])
app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(approve.router, prefix="/api", tags=["chat"])
app.include_router(threads.router, prefix="/api", tags=["threads"])
app.include_router(documents.router, tags=["documents"])
app.include_router(health.router)
app.include_router(llm_check.router)


@app.middleware("http")
async def access_log_middleware(request: Request, call_next):
    """统一访问日志（对齐 RAG/Gin 侧 [ACCESS] 格式），替代 uvicorn 默认 access log。"""
    start = time.perf_counter()
    response = await call_next(request)
    access_logger.info(
        '[ACCESS] %s | %d | %s | %s | %s "%s"',
        time.strftime("%Y/%m/%d - %H:%M:%S", time.localtime()),
        response.status_code,
        _format_duration(time.perf_counter() - start),
        request.client.host if request.client else "-",
        request.method,
        request.url.path,
    )
    return response


@app.exception_handler(Exception)
async def uncaught_exception_handler(request: Request, exc: Exception):
    """全局兜底：未捕获异常 → 分类响应（不裸 500）+ 结构化日志。

    - 工具调用失败不进这里（Dispatcher 已转 ToolResult → ToolMessage 正常回填）；
    - 这里接的是 LLM 调用失败（retry_call 重试穷尽 / is_transient 快速失败）
      及其余未预期异常；
    - thread_id 由 chat/approve 路由挂到 request.state 上捎带过来（仅本次请求内有效）。
    """
    status, code, message = classify_exception(exc)
    logger.error(
        "request_failed path=%s thread_id=%s user_id=%s exc_type=%s exc_msg=%s",
        request.url.path,
        getattr(request.state, "thread_id", None),
        getattr(request.state, "user_id", None),
        type(exc).__name__,
        str(exc)[:200],
    )
    logger.debug("request_failed_traceback", exc_info=True)  # 排查时开 DEBUG 看全栈
    return JSONResponse(
        status_code=status,
        content={"status": "error", "error_code": code, "message": message},
    )
