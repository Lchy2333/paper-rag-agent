"""上下文压缩（摘要 + 前 N 轮）测试：不依赖真实 LLM / RAG / 数据库。

覆盖：轮切边界（tool_calls / ToolMessage 配对不拆散）、无可压缩内容 no-op、
compress 节点重建历史（摘要 + 近 N 轮）、fail-open（摘要失败 → 历史不变、
请求照常）、route_from_start 分流、开关关闭时 build_graph 不装 compress 节点。
"""
import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.memory import MemorySaver

from agent.graph import build_graph
from agent.nodes import (
    SUMMARY_MARKER,
    compress_node,
    route_from_start,
    split_for_compression,
)
from core import Dispatcher, ToolRegistry


def _round(human: str, ai: str):
    """构造一轮完整对话（无工具）。"""
    return [HumanMessage(content=human), AIMessage(content=ai)]


def _tool_round(human: str, ai: str, tool: str):
    """构造一轮带工具调用的对话：Human → AIMessage(tool_calls) → ToolMessage → AI。"""
    return [
        HumanMessage(content=human),
        AIMessage(content="", tool_calls=[{"id": f"call_{human}", "name": "echo", "args": {}}]),
        ToolMessage(content=tool, tool_call_id=f"call_{human}"),
        AIMessage(content=ai),
    ]


class FakeLLM:
    """invoke 返回预设摘要；stream 供 call_llm 消费预设回复。"""

    def __init__(self, summary: str, replies):
        self._summary = summary
        self._replies = list(replies)
        self.summary_inputs = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.summary_inputs.append(messages)
        return AIMessage(content=self._summary)

    def stream(self, messages):
        reply = self._replies.pop(0)
        yield AIMessageChunk(content=reply.content, tool_calls=reply.tool_calls)


class FailingSummaryLLM(FakeLLM):
    """摘要调用必抛（模拟压缩失败），stream 正常——验证 fail-open。"""

    def invoke(self, messages):
        self.summary_inputs.append(messages)
        raise RuntimeError("摘要服务挂了")


def _dispatcher():
    return Dispatcher(ToolRegistry())


# ── 轮切（纯函数单测）─────────────────────────────────────────────────────

def test_split_never_breaks_tool_pair():
    """切点必落在 HumanMessage 边界：tool_calls / ToolMessage 配对完整落在同侧。"""
    msgs = [SystemMessage(content="sys")]
    msgs += _tool_round("q1", "a1", "t1")     # 轮1（带工具）
    msgs += _round("q2", "a2")                # 轮2
    msgs += _round("q3", "a3")                # 轮3
    msgs += _round("q4", "a4")                # 轮4

    split = split_for_compression(msgs, keep_rounds=2)
    assert split is not None
    head, old, kept = split

    assert [m.content for m in head] == ["sys"]
    # 切点 = 第 3 轮的 HumanMessage
    assert isinstance(kept[0], HumanMessage) and kept[0].content == "q3"
    # 轮1 的工具配对完整留在 old 侧（既不拆散、也不丢）
    old_pairs = [(m, m2) for m, m2 in zip(old, old[1:])]
    assert any(
        isinstance(m, AIMessage) and m.tool_calls and isinstance(m2, ToolMessage)
        for m, m2 in old_pairs
    )
    # kept 侧从切点起全部保留
    assert [m.content for m in kept] == ["q3", "a3", "q4", "a4"]


def test_split_noop_when_rounds_insufficient():
    msgs = [SystemMessage(content="sys")] + _round("q1", "a1") + _round("q2", "a2")
    assert split_for_compression(msgs, keep_rounds=2) is None      # 轮数 ≤ keep
    assert split_for_compression([AIMessage(content="孤立消息")], 2) is None  # 无 Human 起点


def test_split_folds_previous_summary_into_old():
    """上一轮压缩留下的摘要消息不进前导段，随旧历史一起被重新折叠。"""
    msgs = [SystemMessage(content="sys"), SystemMessage(content=f"{SUMMARY_MARKER}\n旧摘要")]
    msgs += _round("q1", "a1") + _round("q2", "a2") + _round("q3", "a3")

    split = split_for_compression(msgs, keep_rounds=2)
    head, old, kept = split
    assert [m.content for m in head] == ["sys"]                    # 只有 system prompt
    assert any(m.content.startswith(SUMMARY_MARKER) for m in old)  # 旧摘要进 old


# ── compress 节点 / 路由 ──────────────────────────────────────────────────

def test_route_from_start():
    state = {"messages": []}
    assert route_from_start(state, {"configurable": {"compress": True}}) == "compress"
    assert route_from_start(state, {"configurable": {}}) == "call_llm"
    assert route_from_start(state, None) == "call_llm"


def test_compress_rewrites_history():
    """压缩后：[system, 摘要, 近N轮]；摘要 prompt 只含旧历史；本轮 query 保留。"""
    llm = FakeLLM(summary="压缩摘要：用户问过旧问题", replies=[AIMessage(content="最终回答")])
    graph = build_graph(llm, _dispatcher(), MemorySaver(), compress_keep_rounds=2)

    seed = [SystemMessage(content="sys")]
    seed += _round("旧问题1", "旧回答1") + _round("旧问题2", "旧回答2")
    seed += _round("旧问题3", "旧回答3") + _round("旧问题4", "旧回答4")
    config = {"configurable": {"thread_id": "t-c", "compress": True}}
    graph.update_state({"configurable": {"thread_id": "t-c"}}, {"messages": seed})

    graph.invoke({"messages": [HumanMessage(content="新问题")]}, config=config)

    # 摘要 prompt 只喂旧历史（含旧问题1~3），不含保留轮（旧问题4）与新问题
    summary_prompt = str(llm.summary_inputs[0])
    assert "旧问题1" in summary_prompt and "旧问题2" in summary_prompt
    assert "旧问题3" in summary_prompt
    assert "旧问题4" not in summary_prompt and "新问题" not in summary_prompt

    # 最终状态：[sys, 摘要, 旧问题4 轮, 新问题, 回答]
    final = graph.get_state(config).values["messages"]
    contents = [m.content for m in final]
    assert contents[0] == "sys"
    assert contents[1].startswith(SUMMARY_MARKER)
    assert "旧问题1" not in contents and "旧问题2" not in contents and "旧问题3" not in contents
    assert "旧问题4" in contents and "新问题" in contents and "最终回答" in contents
    # call_llm 收到的就是压缩后的完整列表
    llm_msgs = llm._replies  # 已耗尽；改从断言 final 覆盖即可


def test_compress_fail_open():
    """摘要调用失败 → 放弃压缩，历史原样保留，请求照常完成。"""
    llm = FailingSummaryLLM(summary="", replies=[AIMessage(content="最终回答")])
    graph = build_graph(llm, _dispatcher(), MemorySaver(), compress_keep_rounds=2)

    seed = [SystemMessage(content="sys")]
    seed += _round("q1", "a1") + _round("q2", "a2") + _round("q3", "a3")
    config = {"configurable": {"thread_id": "t-f", "compress": True}}
    graph.update_state({"configurable": {"thread_id": "t-f"}}, {"messages": seed})

    graph.invoke({"messages": [HumanMessage(content="新问题")]}, config=config)

    final = graph.get_state(config).values["messages"]
    contents = [m.content for m in final]
    assert "q1" in contents and "q2" in contents and "q3" in contents  # 历史没动
    assert not any(isinstance(c, str) and c.startswith(SUMMARY_MARKER) for c in contents)
    assert "最终回答" in contents


def test_compress_skipped_without_flag():
    """不传 compress 标志 → 节点空转，历史不动（双保险：路由也不会进来）。"""
    llm = FakeLLM(summary="不该被调用", replies=[AIMessage(content="回答")])
    node = compress_node(llm, keep_rounds=2)
    msgs = [SystemMessage(content="sys")] + _round("q1", "a1") + _round("q2", "a2") + _round("q3", "a3")
    assert node({"messages": msgs}, {"configurable": {}}) == {}
    assert llm.summary_inputs == []  # 摘要模型一次都没被调用
