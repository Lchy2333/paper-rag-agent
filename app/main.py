"""FastAPI 应用入口。

用 lifespan 一次性构建 client / registry / dispatcher / graph，
挂到 app.state，路由通过依赖注入按需取用（进程生命周期内只建一次）。
"""
import json
import sys
from contextlib import asynccontextmanager

import requests
from fastapi import FastAPI
from openai import OpenAI

from agent.graph import build_graph
from app.api import chat
from app.config import settings
from core import Dispatcher, ToolRegistry
from tools.http_tool import HttpTool

sys.stdout.reconfigure(encoding="utf-8")


def build_agent():
    """构建 LLM client + 工具 registry + dispatcher + LangGraph。"""
    registry = ToolRegistry()
    session = requests.Session()

    configs = json.loads(settings.tools_config.read_text(encoding="utf-8"))
    for name, cfg in configs.items():
        registry.register(HttpTool(session, settings.rag_base_url, cfg, name))

    dispatcher = Dispatcher(registry)
    client = OpenAI(
        base_url=settings.ark_base_url,
        api_key=settings.ark_api_key,
    )
    graph = build_graph(client, dispatcher, settings.ark_model)
    return client, dispatcher, graph


@asynccontextmanager
async def lifespan(app: FastAPI):
    client, dispatcher, graph = build_agent()
    app.state.client = client
    app.state.dispatcher = dispatcher
    app.state.graph = graph
    app.state.system_prompt = settings.system_prompt
    yield


app = FastAPI(title="paper-rag-agent", lifespan=lifespan)
app.include_router(chat.router, prefix="/api", tags=["chat"])


@app.get("/health")
def health():
    return {"status": "ok"}
