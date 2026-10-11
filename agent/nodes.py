"""图节点：call_llm / call_tool / route_after_llm。

节点收 state（AgentState），返回"增量"dict；官方引擎用 add_messages reducer
自动把返回的 messages 合并进全局 state。

节点内使用 LangChain 消息对象（SystemMessage/HumanMessage/AIMessage/ToolMessage）
作为 state 的消息载体——这是 LangGraph 官方生态的标准做法，add_messages 原生支持；
调用 openai SDK 前统一转换为 OpenAI 兼容 dict。
"""
import json
from typing import Any, Callable, Dict

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END

from core.dispatcher import Dispatcher

# LangChain 消息 type -> OpenAI role
_ROLE_MAP = {
    "human": "user",
    "ai": "assistant",
    "system": "system",
    "tool": "tool",
}


def to_openai_dicts(messages: list) -> list[Dict[str, Any]]:
    """把 LangChain 消息列表转成 OpenAI 兼容 dict，供 SDK 调用。"""
    out = []
    for m in messages:
        if isinstance(m, ToolMessage):
            out.append({
                "role": "tool",
                "content": m.content or "",
                "tool_call_id": m.tool_call_id,
            })
        elif isinstance(m, AIMessage) and m.tool_calls:
            out.append({
                "role": "assistant",
                "content": m.content or "",
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["args"]),
                        },
                    }
                    for tc in m.tool_calls
                ],
            })
        else:
            out.append({"role": _ROLE_MAP.get(m.type, m.type), "content": m.content or ""})
    return out


def call_llm(client, dispatcher: Dispatcher, model: str) -> Callable:
    """节点：调 LLM，返回 assistant 消息（有 tool_calls 时带上）。"""
    def node(state: Dict[str, Any]) -> Dict[str, Any]:
        response = client.chat.completions.create(
            model=model,
            messages=to_openai_dicts(state["messages"]),
            tool_choice="auto",
            tools=dispatcher.registry.list_schemas(),
        )
        msg = response.choices[0].message

        if msg.tool_calls:
            print(f"tool_calls: {msg.tool_calls}")
            assistant_msg = AIMessage(
                content=msg.content or "",
                tool_calls=[
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "args": json.loads(tc.function.arguments or "{}"),
                    }
                    for tc in msg.tool_calls
                ],
            )
        else:
            assistant_msg = AIMessage(content=msg.content or "")

        return {"messages": [assistant_msg]}

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
