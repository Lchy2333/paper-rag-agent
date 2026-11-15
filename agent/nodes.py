"""图节点：call_llm / compress / call_tool / guard 与各条件边。

节点收 state（AgentState），返回"增量"dict；官方引擎用 add_messages reducer
自动把返回的 messages 合并进全局 state。

节点内全程使用 LangChain 消息对象（SystemMessage/HumanMessage/AIMessage/ToolMessage）
——ChatOpenAI 原生收/发 LangChain 消息，无需手动转换 dict。
"""
import json
import logging
from typing import Any, Callable, Dict

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.graph import END
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import interrupt

from app.span import span
from core.dispatcher import Dispatcher
from core.retry import is_transient
from core.stream_guard import GenerationCancelled, iter_stream_with_guard

logger = logging.getLogger(__name__)

# 首 token 守护（秒）：流开始后限期内无首 chunk → 判死。
# 上游挂起（对 stream 请求不吐数据）若不判死，会无限阻塞线程池线程；
# 判死抛 TimeoutError（is_transient 已覆盖）→ 由下方重试循环按瞬态错误重试。
FIRST_TOKEN_TIMEOUT = 30.0


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
        # span 包整个重试循环：用户感知的「等 LLM」= 所有 attempt + 退避等待之和。
        # 重试期间 rid 经 contextvar 随线程池复制可达（span 行自动带本次请求的 id）。
        with span("llm_call"):
            final = None
            attempt = 0
            while True:
                try:
                    # 用首 token 守护包住 stream：上游挂起（限期内无数据）→ TimeoutError
                    # （瞬态，进入下方重试）；逐 chunk 检查 cancel_event（挂起时也有效）
                    for chunk in iter_stream_with_guard(
                        llm_with_tools.stream(state["messages"]),
                        FIRST_TOKEN_TIMEOUT,
                        cancel_event=cancel_event,
                    ):
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
            # span 按工具名分组（tool.search_papers / tool.upload …）：阶段 P95 天然按工具统计。
            # 工具失败不上抛（Dispatcher 转 ToolResult 正常回填）——ok 必须手动标，
            # 否则失败请求的 span 恒为 ok=1，阶段错误率统计会漏。
            with span(f"tool.{name}") as sp:
                result = dispatcher.dispatch(name, args, user_id=user_id, on_event=writer)
                if not result.success:
                    sp.ok = 0
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


# ── 上下文压缩（摘要 + 前 N 轮）────────────────────────────────────────────
# 触发：app 层读 query_usage.last_input_tokens（上次实测水位），≥ 阈值 × input_budget
# 时经 config["configurable"]["compress"]=True 传入；START 条件边据此分流。
# 节点内一次 llm.invoke 生成摘要（走 callbacks 自动计费），失败 fail-open 放弃压缩。

SUMMARY_MARKER = "【历史摘要】"


def _is_summary(msg) -> bool:
    """上一轮压缩留下的摘要消息——再次压缩时并入旧历史（两级折叠），不进前导段。"""
    return isinstance(msg, SystemMessage) and str(msg.content).startswith(SUMMARY_MARKER)


def _render_for_summary(messages) -> str:
    """把旧消息渲染成纯文本喂给摘要模型（多模态 content 块只取 text）。"""
    lines = []
    for m in messages:
        role = {SystemMessage: "system", HumanMessage: "user",
                AIMessage: "assistant", ToolMessage: "tool"}.get(type(m), type(m).__name__)
        content = m.content
        if isinstance(content, list):
            content = " ".join(str(b.get("text", "")) for b in content if isinstance(b, dict))
        text = str(content).strip()
        if text:
            lines.append(f"{role}: {text[:2000]}")
    return "\n".join(lines)


def split_for_compression(messages, keep_rounds: int):
    """按轮切分历史，返回 (leading_system, old_msgs, kept_msgs)；无可压缩内容返回 None。

    轮 = 一条 HumanMessage 起头的完整段（Human…到下一条 Human 前），切点必落在
    HumanMessage 边界 → 永远不会拆散 AIMessage(tool_calls) / ToolMessage 配对
    （拆散会被 OpenAI 协议拒单）。leading_system = 前导 SystemMessage（system prompt）；
    旧摘要不在此列——进 old 被重新折叠进新摘要。
    """
    msgs = list(messages)
    head, i = [], 0
    while i < len(msgs) and isinstance(msgs[i], SystemMessage) and not _is_summary(msgs[i]):
        head.append(msgs[i])
        i += 1
    rest = msgs[i:]
    human_pos = [j for j, m in enumerate(rest) if isinstance(m, HumanMessage)]
    if not human_pos or len(human_pos) <= keep_rounds:
        return None
    cut = human_pos[-keep_rounds]  # 最近 keep_rounds 轮的起点
    old, kept = rest[:cut], rest[cut:]
    if not old:
        return None
    return head, old, kept


_SUMMARY_PROMPT = """你在压缩一个论文问答助手的对话历史。只陈述已发生的内容，严禁编造。
以下是按时间排列的旧消息（system=此前的历史摘要 / user=用户 / assistant=回答 / tool=工具结果）。
请用不超过 400 字输出（某部分没有就省略）：
1. 用户核心目标
2. 已确认的事实与约束
3. 已执行的工具调用及结果要点（论文名、检索结果的关键信息要保留）
4. 未决事项

对话历史：
{history}"""


def compress_node(llm, keep_rounds: int) -> Callable:
    """压缩节点：旧历史 → 一条摘要 + 保留最近 N 轮，RemoveMessage 重建 messages 通道。

    - 摘要调用走 config.callbacks → UsageCallbackHandler 自动计费（节点内新增
      LLM 调用点的零接入成本，见 §7.2 回调方案）；
    - 超时由 llm 客户端 read 超时兜底；失败 fail-open：放弃本次压缩返回空增量，
      绝不让压缩失败连累对话请求（与 rerank fail-open 同一哲学）；
    - 返回 [RemoveMessage(REMOVE_ALL), system, 摘要, *近N轮]：add_messages 在一次
      合并内完成"清空+重建"（REMOVE_ALL 必须在列表首位），本步落盘的 checkpoint
      即重建后的完整列表，无中间态。
    """
    def node(state: Dict[str, Any], config: RunnableConfig | None = None) -> Dict[str, Any]:
        cfg = (config or {}).get("configurable") or {}
        if not cfg.get("compress"):
            return {}
        split = split_for_compression(state["messages"], keep_rounds)
        if split is None:
            # 触发了但没得压（轮数 ≤ N）：静默跳过会让"水位超限却永不压缩"变得隐蔽
            # （§7.4 教训：不触发时不报错 ≠ 可以不打日志），记一条 info 便于排查
            logger.info("compress_skipped old_rounds_empty keep_rounds=%d", keep_rounds)
            return {}
        head, old, kept = split
        try:
            try:
                writer = get_stream_writer() or None
            except Exception:
                writer = None
            if writer is not None:
                writer({"label": "对话较长，正在压缩历史…"})
            summary = llm.invoke(
                _SUMMARY_PROMPT.format(history=_render_for_summary(old))
            ).content
            logger.info("history_compressed old_msgs=%d kept_msgs=%d summary_len=%d",
                        len(old), len(kept), len(str(summary)))
        except Exception as e:
            logger.warning("compress_failed_skip exc_type=%s exc_msg=%s",
                           type(e).__name__, str(e)[:200])
            return {}
        summary_msg = SystemMessage(content=f"{SUMMARY_MARKER}\n{summary}")
        return {"messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *head,
            summary_msg,
            *kept,
        ]}
    return node


def route_from_start(state: Dict[str, Any], config: RunnableConfig | None = None) -> str:
    """条件边：app 层按上次实测水位判定需要压缩 → 先进 compress，否则直达 call_llm。"""
    cfg = (config or {}).get("configurable") or {}
    return "compress" if cfg.get("compress") else "call_llm"
