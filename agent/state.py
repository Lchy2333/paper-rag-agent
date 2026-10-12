"""LangGraph 图状态定义。

官方 LangGraph 要求用 TypedDict 声明状态结构；
messages 字段使用 add_messages reducer，节点返回的增量消息会自动合并，
替代了 MVP 里手写的 `state["messages"] + [...]`。
"""
from typing import Annotated, NotRequired, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    # guard 审批节点的决策：approve / reject / noop（仅在有 tool_calls 时写入）
    route: NotRequired[str]
