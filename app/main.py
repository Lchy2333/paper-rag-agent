"""FastAPI 应用入口。

用 lifespan 一次性构建 client / registry / dispatcher / graph，
挂到 app.state，路由通过依赖注入按需取用（进程生命周期内只建一次）。
全局 exception_handler 兜底未捕获异常：分类响应 + 结构化日志（工具调用失败
已被 Dispatcher 转成 ToolMessage，不进这里）。
"""
import json
import logging
import sys
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
from app.api import approve, auth, chat, threads
from app.config import settings
from app.errors import classify_exception
from core import Dispatcher, ToolRegistry
from tools.http_tool import HttpTool

sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("paper-rag-agent")


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
        base_url=settings.ark_base_url,
        api_key=settings.ark_api_key,
        model=settings.ark_model,
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
    graph = build_graph(llm, dispatcher, checkpointer)
    return llm, dispatcher, graph, pool


@asynccontextmanager
async def lifespan(app: FastAPI):
    llm, dispatcher, graph, pool = build_agent()
    app.state.llm = llm
    app.state.dispatcher = dispatcher
    app.state.graph = graph
    app.state.pool = pool
    app.state.system_prompt = settings.system_prompt
    try:
        yield
    finally:
        pool.close()


app = FastAPI(title="paper-rag-agent", lifespan=lifespan)
app.include_router(auth.router, prefix="/api", tags=["auth"])
app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(approve.router, prefix="/api", tags=["chat"])
app.include_router(threads.router, prefix="/api", tags=["threads"])


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


@app.get("/health")
def health():
    return {"status": "ok"}
