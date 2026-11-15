"""用官方 LangGraph 组装 ReAct 图（含人工审批护栏 + 上下文压缩）。

                          START
                            │ (compress=True：上次实测水位超阈值)
                            ▼
        ┌──(False)──────► call_llm ──(无 tool_calls)──► END
        │   ┌─ compress ──┘
        │   │ (旧历史 → 摘要 + 最近 N 轮，RemoveMessage 重建)
        │   ▼
        │  call_llm
        │   │
        │   │ (有 tool_calls)
        │   ▼
        │ guard ──(approve / noop)──► call_tool
        │   │                             │
        │   │ (reject)                    │
        └───┴─────────────────────────────┘
                回到 call_llm：reject 时跳过工具，下一轮仍可能再产 tool_calls

循环没有次数上限，唯一出口是 call_llm 的「无 tool_calls」分支；
兜底靠框架 recursion_limit（默认 25），超出抛 GraphRecursionError。

compress 只在请求起点跑一次（app 层按上次实测水位判定并经 config 传入）；
HITL 恢复段（/api/approve）不传 compress → 恢复时不会意外二次压缩。

guard 节点对 requires_approval 的工具调用 interrupt 挂起，等用户审批后恢复
（需 checkpointer 支持；生产换持久化实现，学习阶段用 MemorySaver）。
"""
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.nodes import (
    call_llm,
    call_tool,
    compress_node,
    guard_node,
    route_after_guard,
    route_after_llm,
    route_from_start,
)
from agent.state import AgentState


def build_graph(llm, dispatcher, checkpointer=None, llm_tracker=None,
                compress_keep_rounds: int | None = None) -> CompiledStateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("call_llm", call_llm(llm, dispatcher, tracker=llm_tracker))
    graph.add_node("guard", guard_node(dispatcher))
    graph.add_node("call_tool", call_tool(dispatcher))

    if compress_keep_rounds:
        # 压缩按请求粒度触发（configurable.compress），不触发时零开销直达 call_llm
        graph.add_node("compress", compress_node(llm, compress_keep_rounds))
        graph.add_conditional_edges(
            START, route_from_start, {"compress": "compress", "call_llm": "call_llm"},
        )
        graph.add_edge("compress", "call_llm")
    else:
        graph.add_edge(START, "call_llm")
    graph.add_edge("call_tool", "call_llm")
    graph.add_conditional_edges("call_llm", route_after_llm)
    graph.add_conditional_edges(
        "guard",
        route_after_guard,
        {"approve": "call_tool", "noop": "call_tool", "reject": "call_llm"},
    )

    return graph.compile(checkpointer=checkpointer)
