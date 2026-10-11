"""LangGraph 图状态定义。

官方 LangGraph 要求用 TypedDict 声明状态结构；
messages 字段使用 add_messages reducer，节点返回的增量消息会自动合并，
替代了 MVP 里手写的 `state["messages"] + [...]`。
"""
from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
