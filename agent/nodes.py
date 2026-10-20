"""图节点：call_llm / call_tool / route_after_llm。

节点收 state（AgentState），返回"增量"dict；官方引擎用 add_messages reducer
自动把返回的 messages 合并进全局 state。

节点内全程使用 LangChain 消息对象（SystemMessage/HumanMessage/AIMessage/ToolMessage）
——ChatOpenAI 原生收/发 LangChain 消息，无需手动转换 dict。
"""
import json
import logging
from typing import Any, Callable, Dict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.graph import END
from langgraph.types import interrupt

from core.dispatcher import Dispatcher
from core.retry import is_transient

logger = logging.getLogger(__name__)


class GenerationCancelled(Exception):
    """用户在前端主动中断生成：节点抛出，/chat/stream 捕获后以 cancel 事件收尾。

    必须单独捕获（不能走 classify_exception 兜底），否则会被当成 internal_error。
    """


def call_llm(llm, dispatcher: Dispatcher, tracker=None) -> Callable:
    """节点：调 LLM，返回 assistant 消息（有 tool_calls 时带上）。

    llm 是 langchain_openai.ChatOpenAI 实例。
    用 llm.stream() 逐字生成而非 invoke：既能逐 token 检查用户取消标记（随时可停），
    也顺带把 token 喂给 stream_mode="messages"（打字机效果）。
    取消/流式路径不包 retry_call：流式中途失败无法安全重放，且取消语义要干净。
    tracker（core.llm_probe.LLMHealthTracker）：记录真实调用成败 → /health 被动信号。
    """
    def node(state: Dict[str, Any], config: RunnableConfig | None = None) -> Dict[str, Any]:
        # 进度事件：stream_mode 含 custom 时有 writer；invoke/测试时返回 None（保持原行为）
        try:
            writer = get_stream_writer() or None
        except Exception:
            writer = None
        if writer is not None:
            last = state["messages"][-1] if state.get("messages") else None
            if isinstance(last, ToolMessage):
                writer({"label": "正在综合检索结果生成回答…"})
            elif isinstance(last, HumanMessage):
                writer({"label": "正在理解问题、优化检索词…"})
        # 取消标记：/chat/stream 创建的 threading.Event，/chat/cancel 置位
        cancel_event = None
        if config:
            cancel_event = (config.get("configurable") or {}).get("cancel_event")
        llm_with_tools = llm.bind_tools(dispatcher.registry.list_schemas())
        final = None
        attempt = 0
        while True:
            try:
                for chunk in llm_with_tools.stream(state["messages"]):
                    if cancel_event is not None and cancel_event.is_set():
                        raise GenerationCancelled()
                    final = chunk if final is None else final + chunk
                break  # 流正常走完
            except GenerationCancelled:
                raise  # 用户主动停止：不记为 LLM 失败
            except Exception as e:
                # 重试规则：仅当"一个 token 都没吐出来"时才对瞬态错误重试
                # （此时重试不会导致前端看到重复 token）；流中途失败不重试。
                if final is not None or not is_transient(e):
                    if tracker is not None:
                        tracker.record_failure(str(e))
                    raise
                attempt += 1
                if attempt >= 3:
                    if tracker is not None:
                        tracker.record_failure(str(e))
                    raise
                dispatcher.sleep(0.2 * attempt)
        if tracker is not None:
            tracker.record_success()
        if final is None:  # 防御：流式空响应
            final = AIMessage(content="")
        logger.info("llm_response: %s", final)
        return {"messages": [final]}

    return node


def call_tool(dispatcher: Dispatcher) -> Callable:
    """节点：执行最后一条 assistant 消息里的所有工具调用，返回 tool 消息。

    config 里可携带 user_id（由 /chat 路由从认证上下文注入，LLM 不可见），
    经 Dispatcher 透传给工具作为 X-User-ID 头（多用户文档隔离）。
    """
    def node(state: Dict[str, Any], config: RunnableConfig | None = None) -> Dict[str, Any]:
        # 兼容：graph.invoke 会传 config；测试直接调用 node(state) 时 config=None
        user_id = None
        cancel_event = None
        if config:
            configurable = config.get("configurable") or {}
            user_id = configurable.get("user_id")
            cancel_event = configurable.get("cancel_event")
        try:
            writer = get_stream_writer() or None
        except Exception:
            writer = None
        last = state["messages"][-1]
        tool_msgs = []
        for tc in last.tool_calls:
            # 用户中途停止：在下个工具调用前也检查一次（停在工具边界）
            if cancel_event is not None and cancel_event.is_set():
                raise GenerationCancelled()
            name = tc["name"]
            args = json.dumps(tc["args"])
            logger.info("Calling tool: %s(%s) user_id=%s", name, args, user_id)
            if writer is not None:
                # 带上检索词，用户能看出正在用什么词搜（如 search_rag 的 query）
                tool_args = tc["args"] or {}
                query = str(tool_args.get("query", ""))
                if query:
                    label = f"正在调用工具 {name}（检索词：{query}）…"
                    if len(label) > 100:  # 超长检索词截断，避免进度行过长
                        label = label[:97] + "…"
                else:
                    label = f"正在调用工具 {name}…"
                writer({"label": label})
            result = dispatcher.dispatch(name, args, user_id=user_id, on_event=writer)
            content = result.data if result.success else f"工具执行失败: {result.error}"
            if writer is not None and result.success and isinstance(result.data, dict):
                total = result.data.get("total_chunks")
                if total is not None:
                    writer({"label": f"检索完成 · 命中 {total} 条相关片段"})
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
