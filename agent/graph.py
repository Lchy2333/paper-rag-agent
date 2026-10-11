"""用官方 LangGraph 组装 ReAct 图。

    START
      │
      ▼
   call_llm ──(有 tool_calls)──► call_tool
      │                            │
      └─────(无 tool_calls)────────┘
                     │
                     ▼
                    END
"""
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.nodes import call_llm, call_tool, route_after_llm
from agent.state import AgentState


def build_graph(llm, dispatcher) -> CompiledStateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("call_llm", call_llm(llm, dispatcher))
    graph.add_node("call_tool", call_tool(dispatcher))

    graph.add_edge(START, "call_llm")
    graph.add_edge("call_tool", "call_llm")
    graph.add_conditional_edges("call_llm", route_after_llm)

    return graph.compile()
