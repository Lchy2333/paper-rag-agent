"""流式进度事件回归测试：验证 stream_mode="custom" 下，节点插桩的 stage 事件真实流出。

覆盖：call_llm 的"理解问题"、call_tool 的"调用工具/检索完成"、
以及支持 on_event 的流式工具自己转发的"检索中"进度。
"""
import asyncio

from langchain_core.messages import AIMessage, HumanMessage

from agent.graph import build_graph
from agent.nodes import call_llm
from core import Dispatcher, ToolRegistry
from core.tool import ToolBase, ToolResult

from tests.test_graph import FakeLLM, _tool_call_message


class _StreamingSearchTool(ToolBase):
    """流式假工具：execute 声明 on_event，模拟 RAG 的 SSE 进度转发。"""

    def name(self):
        return "search_rag"

    def build_schema(self):
        return {
            "type": "function",
            "function": {
                "name": "search_rag",
                "description": "检索论文库",
                "parameters": {"type": "object", "properties": {}},
            },
        }

    def execute(self, args, user_id=None, on_event=None):
        if on_event is not None:
            on_event({"label": "检索中 · 正在向量化查询…"})
        return ToolResult(success=True, data={"total_chunks": 2, "results": []})


def _build_streaming_dispatcher():
    registry = ToolRegistry()
    registry.register(_StreamingSearchTool())
    return Dispatcher(registry)


def test_astream_custom_emits_stage_events():
    dispatcher = _build_streaming_dispatcher()
    llm = FakeLLM([
        _tool_call_message("search_rag", {"query": "hi"}),
        AIMessage(content="回答：命中 2 条"),
    ])
    graph = build_graph(llm, dispatcher)

    events = []

    async def run():
        async for mode, chunk in graph.astream(
            {"messages": [HumanMessage("hi")]},
            stream_mode=["custom"],
        ):
            if mode == "custom" and isinstance(chunk, dict):
                events.append(chunk.get("label", ""))

    asyncio.run(run())

    labels = " | ".join(events)
    assert "正在理解问题、优化检索词" in labels, f"缺理解阶段: {labels}"
    assert "正在调用工具 search_rag" in labels, f"缺调用工具阶段: {labels}"
    assert "检索词：hi" in labels, f"调用工具阶段应带上检索词: {labels}"
    assert "检索中 · 正在向量化查询" in labels, f"缺工具内部进度: {labels}"
    assert "检索完成 · 命中 2 条相关片段" in labels, f"缺检索完成摘要: {labels}"
    # 工具调用轮不应有"正在综合检索结果"（那属于最终回答轮）
    assert "正在综合检索结果生成回答" in labels, f"缺最终回答阶段: {labels}"
