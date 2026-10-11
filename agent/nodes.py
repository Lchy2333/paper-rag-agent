"""图节点：call_llm / call_tool / route_after_llm。

节点收 state（AgentState），返回"增量"dict；官方引擎用 add_messages reducer
自动把返回的 messages 合并进全局 state。

节点内全程使用 LangChain 消息对象（SystemMessage/HumanMessage/AIMessage/ToolMessage）
——ChatOpenAI 原生收/发 LangChain 消息，无需手动转换 dict。
"""
import json
from typing import Any, Callable, Dict

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END

from core.dispatcher import Dispatcher


def call_llm(llm, dispatcher: Dispatcher) -> Callable:
    """节点：调 LLM，返回 assistant 消息（有 tool_calls 时带上）。

    llm 是 langchain_openai.ChatOpenAI 实例。
    bind_tools 把工具 schema 绑到模型；invoke 直接收 LangChain 消息、返回 AIMessage。
    """
    def node(state: Dict[str, Any]) -> Dict[str, Any]:
        llm_with_tools = llm.bind_tools(dispatcher.registry.list_schemas())
        response = llm_with_tools.invoke(state["messages"])
        print(f"llm_response: {response}")
        return {"messages": [response]}

    return node


def call_tool(dispatcher: Dispatcher) -> Callable:
    """节点：执行最后一条 assistant 消息里的所有工具调用，返回 tool 消息。"""
    def node(state: Dict[str, Any]) -> Dict[str, Any]:
        last = state["messages"][-1]
        tool_msgs = []
        for tc in last.tool_calls:
            name = tc["name"]
            args = json.dumps(tc["args"])
            print(f"Calling tool: {name}({args})")
            result = dispatcher.dispatch(name, args)
            content = result.data if result.success else f"工具执行失败: {result.error}"
            tool_msgs.append(ToolMessage(content=str(content), tool_call_id=tc["id"]))
        return {"messages": tool_msgs}

    return node


def route_after_llm(state: Dict[str, Any]) -> str:
    """条件边：LLM 决定调工具 → call_tool；否则结束。"""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "call_tool"
    return END
