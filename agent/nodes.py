"""图节点：call_llm / call_tool / route_after_llm。

节点收 state（AgentState），返回"增量"dict；官方引擎用 add_messages reducer
自动把返回的 messages 合并进全局 state。

节点内全程使用 LangChain 消息对象（SystemMessage/HumanMessage/AIMessage/ToolMessage）
——ChatOpenAI 原生收/发 LangChain 消息，无需手动转换 dict。
"""
import json
import logging
from typing import Any, Callable, Dict

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from langgraph.types import interrupt

from core.dispatcher import Dispatcher
from core.retry import is_transient, retry_call

logger = logging.getLogger(__name__)


def call_llm(llm, dispatcher: Dispatcher, tracker=None) -> Callable:
    """节点：调 LLM，返回 assistant 消息（有 tool_calls 时带上）。

    llm 是 langchain_openai.ChatOpenAI 实例。
    bind_tools 把工具 schema 绑到模型；invoke 直接收 LangChain 消息、返回 AIMessage。
    LLM 调用包一层 retry_call：ARK 超时/限流/5xx 等瞬态失败指数退避重试（最多 3 次），
    复用 dispatcher.sleep（测试可注入 no-op）。
    tracker（core.llm_probe.LLMHealthTracker）：记录真实调用成败 → /health 被动信号。
    """
    def node(state: Dict[str, Any]) -> Dict[str, Any]:
        llm_with_tools = llm.bind_tools(dispatcher.registry.list_schemas())
        try:
            response = retry_call(
                lambda: llm_with_tools.invoke(state["messages"]),
                retryable=is_transient,
                max_attempts=3,
                jitter_ratio=0.1,
                sleep=dispatcher.sleep,
            )
        except Exception as e:
            if tracker is not None:
                tracker.record_failure(str(e))
            raise
        if tracker is not None:
            tracker.record_success()
        logger.info("llm_response: %s", response)
        return {"messages": [response]}

    return node


def call_tool(dispatcher: Dispatcher) -> Callable:
    """节点：执行最后一条 assistant 消息里的所有工具调用，返回 tool 消息。

    config 里可携带 user_id（由 /chat 路由从认证上下文注入，LLM 不可见），
    经 Dispatcher 透传给工具作为 X-User-ID 头（多用户文档隔离）。
    """
    def node(state: Dict[str, Any], config: RunnableConfig | None = None) -> Dict[str, Any]:
        # 兼容：graph.invoke 会传 config；测试直接调用 node(state) 时 config=None
        user_id = None
        if config:
            user_id = (config.get("configurable") or {}).get("user_id")
        last = state["messages"][-1]
        tool_msgs = []
        for tc in last.tool_calls:
            name = tc["name"]
            args = json.dumps(tc["args"])
            logger.info("Calling tool: %s(%s) user_id=%s", name, args, user_id)
            result = dispatcher.dispatch(name, args, user_id=user_id)
            content = result.data if result.success else f"工具执行失败: {result.error}"
            tool_msgs.append(ToolMessage(content=str(content), tool_call_id=tc["id"]))
        return {"messages": tool_msgs}

    return node


def route_after_llm(state: Dict[str, Any]) -> str:
    """条件边：LLM 决定调工具 → guard（先过审批）；否则结束。"""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "guard"
    return END


def guard_node(dispatcher: Dispatcher) -> Callable:
    """审批节点：发现 requires_approval 的工具调用 → interrupt 挂起等用户审批。

    interrupt() 第一次执行时挂起整张图（invoke 返回，状态存入 checkpointer）；
    用户通过 /api/approve 恢复后，interrupt() 返回 resume 值（"allow"/"reject"）。
    interrupt() 之前的代码必须是纯计算（恢复时节点会从头重跑，有副作用会重复执行）。
    """
    def node(state: Dict[str, Any]) -> Dict[str, Any]:
        last = state["messages"][-1]
        pending = [
            tc for tc in last.tool_calls
            if (tool := dispatcher.registry.get(tc["name"]))
            and getattr(tool, "requires_approval", False)
        ]
        if not pending:
            return {"route": "noop"}

        decision = interrupt({
            "tool": pending[0]["name"],
            "args": pending[0]["args"],
        })
        if decision == "allow":
            return {"route": "approve"}
        return {
            "route": "reject",
            "messages": [SystemMessage(
                content="用户拒绝了该工具调用，请勿执行，并向用户说明原因。")],
        }

    return node


def route_after_guard(state: Dict[str, Any]) -> str:
    """条件边：guard 决策 → approve/noop 走 call_tool，reject 跳过工具回 call_llm。"""
    return state.get("route", "approve")
