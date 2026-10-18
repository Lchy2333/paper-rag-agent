"""用官方 LangGraph 组装 ReAct 图（含人工审批护栏）。

    START
      │
      ▼
   call_llm ──(有 tool_calls)──► guard ──(approve/noop)──► call_tool ──► call_llm
      │                            │
      │                     (reject)──► call_llm（跳过工具）
      └─────(无 tool_calls)────────┘
                     │
                     ▼
                    END

guard 节点对 requires_approval 的工具调用 interrupt 挂起，等用户审批后恢复
（需 checkpointer 支持；生产换持久化实现，学习阶段用 MemorySaver）。
"""
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.nodes import (
    call_llm,
    call_tool,
    guard_node,
    route_after_guard,
    route_after_llm,
)
from agent.state import AgentState


def build_graph(llm, dispatcher, checkpointer=None, llm_tracker=None) -> CompiledStateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("call_llm", call_llm(llm, dispatcher, tracker=llm_tracker))
    graph.add_node("guard", guard_node(dispatcher))
    graph.add_node("call_tool", call_tool(dispatcher))

    graph.add_edge(START, "call_llm")
    graph.add_edge("call_tool", "call_llm")
    graph.add_conditional_edges("call_llm", route_after_llm)
    graph.add_conditional_edges(
        "guard",
        route_after_guard,
        {"approve": "call_tool", "noop": "call_tool", "reject": "call_llm"},
    )

    return graph.compile(checkpointer=checkpointer)
