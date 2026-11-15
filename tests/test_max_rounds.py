"""工具轮次防线测试（P0）：软上限 route → force_answer；动态收口提示；裸 LLM 收尾。

背景（loadtest/report-压测总报告.md §3.4）：232s 案例 = 模型 23 轮失控检索——
空结果是合法返回值不是错误，ReAct 唯一出口是"模型不再发 tool_calls"，
而代码与提示词都没有"搜 N 次就停"的约束。

防线四件套：prompt 检索纪律（settings.system_prompt，行为层难断言不做单测）
+ ToolMessage 动态收口提示 + route_after_llm 软上限 → force_answer（裸 LLM 物理禁工具）
+ recursion_limit 硬保险丝（chat.py/approve.py 接线，GraphRecursionError 转 SSE error）。
"""
import threading

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END

from agent.graph import build_graph
from agent.nodes import (
    GenerationCancelled,
    _tool_rounds_since_human,
    call_llm,
    call_tool,
    force_answer,
    route_after_llm,
)
from core import Dispatcher, ToolRegistry
from core.tool import ToolBase, ToolResult


class _EchoTool(ToolBase):
    """假工具：回显参数（无 requires_approval，guard 恒 noop）。"""

    def name(self) -> str:
        return "echo"

    def build_schema(self):
        return {"type": "function", "function": {"name": "echo", "description": "回显输入",
                                                 "parameters": {"type": "object", "properties": {}}}}

    def execute(self, args, user_id=None):
        return ToolResult(success=True, data={"echo": args.get("msg")})


class FakeLLM:
    """模拟 ChatOpenAI：按序返回预设回复，记录每次调用的消息与绑定工具。"""

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = []
        self.last_bound_tools = None

    def bind_tools(self, tools):
        self.last_bound_tools = tools
        return self

    def invoke(self, messages, config=None):
        self.calls.append({"messages": messages, "tools": self.last_bound_tools})
        return self._replies.pop(0)

    def stream(self, messages):
        self.calls.append({"messages": messages, "tools": self.last_bound_tools})
        reply = self._replies.pop(0)
        yield AIMessageChunk(content=reply.content, tool_calls=reply.tool_calls)


def _tool_msg(name="echo", args=None, call_id="call_1"):
    return AIMessage(content="", tool_calls=[{"id": call_id, "name": name, "args": args or {"msg": "q"}}])


def _dispatcher() -> Dispatcher:
    registry = ToolRegistry()
    registry.register(_EchoTool())
    return Dispatcher(registry)


# ── route_after_llm：软上限判定 ─────────────────────────────────────────────

def test_route_under_soft_cap_goes_to_guard():
    state = {"messages": [HumanMessage("问"), _tool_msg()]}
    assert route_after_llm(state) == "guard"


def test_route_at_soft_cap_goes_to_force_answer():
    # 默认 max_tool_rounds=4：第 4 条带 tool_calls 的 AI 消息触发软上限
    state = {"messages": [HumanMessage("问")] + [_tool_msg(call_id=f"c{i}") for i in range(4)]}
    assert route_after_llm(state) == "force_answer"


def test_route_counts_only_current_turn():
    # 多轮老会话：上一轮的历史 tool_calls 不参与计数——新问题第 1 轮不应误触软上限
    old_turn = [HumanMessage("旧问题"), _tool_msg(call_id="o1"),
                ToolMessage(content="旧结果", tool_call_id="o1"), AIMessage(content="旧回答")]
    new_turn = [HumanMessage("新问题"), _tool_msg(call_id="n1")]
    assert _tool_rounds_since_human(old_turn + new_turn) == 1
    assert route_after_llm({"messages": old_turn + new_turn}) == "guard"


# ── call_tool：动态收口提示 ─────────────────────────────────────────────────

def test_call_tool_appends_hint_from_second_round():
    dispatcher = _dispatcher()
    tool_node = call_tool(dispatcher)

    state = {"messages": [HumanMessage("问"), _tool_msg(call_id="c1")]}
    state["messages"] += tool_node(state)["messages"]
    first = state["messages"][-1]
    assert "系统提示" not in first.content          # 第 1 轮：无提示

    state["messages"] += [_tool_msg(call_id="c2")]
    state["messages"] += tool_node(state)["messages"]
    second = state["messages"][-1]
    assert "已检索 2 次未获得足够依据" in second.content   # 第 2 轮起：追加收口提示


# ── force_answer：裸 LLM 物理禁工具 + 协议安全 ──────────────────────────────

def test_force_answer_uses_unbound_llm_and_patches_tool_calls():
    llm = FakeLLM([AIMessage(content="基于已有信息的收尾回答")])
    node = force_answer(llm)
    state = {"messages": [HumanMessage("问"), _tool_msg(call_id="pending_1")]}

    result = node(state)

    call = llm.calls[0]
    assert call["tools"] is None                                   # 未 bind_tools：物理禁工具
    assert call["messages"][-1].content.startswith("已达到检索次数上限")  # 收尾指令在消息末尾
    patched = call["messages"][-2]
    assert isinstance(patched, ToolMessage) and patched.tool_call_id == "pending_1"  # 悬空补应答
    assert result["messages"][-1].content == "基于已有信息的收尾回答"
    assert result["messages"][0].content.startswith("已达检索次数上限")            # skip ToolMessage 也落 state


# ── force_answer：节点边界取消检查（停止生成最迟在收尾后立刻生效）────────────

def test_force_answer_cancelled_before_invoke():
    ev = threading.Event()
    ev.set()
    llm = FakeLLM([AIMessage(content="不应该生成")])
    node = force_answer(llm)
    state = {"messages": [HumanMessage("问"), _tool_msg()]}

    with pytest.raises(GenerationCancelled):
        node(state, {"configurable": {"cancel_event": ev}})
    assert llm.calls == []  # 进节点即拦截，LLM 没被调用（不白烧 token）


def test_force_answer_cancelled_after_invoke():
    """invoke 是阻塞调用、中途打断不了——但跑完后必须检查标记，丢弃结果按取消收尾。"""

    class _CancelDuringInvokeLLM(FakeLLM):
        def __init__(self, replies, ev):
            super().__init__(replies)
            self._ev = ev

        def invoke(self, messages, config=None):
            self._ev.set()  # 模拟 invoke 期间用户点了「停止生成」
            return super().invoke(messages, config)

    ev = threading.Event()
    llm = _CancelDuringInvokeLLM([AIMessage(content="跑完才被取消")], ev)
    node = force_answer(llm)
    state = {"messages": [HumanMessage("问"), _tool_msg()]}

    with pytest.raises(GenerationCancelled):
        node(state, {"configurable": {"cancel_event": ev}})


# ── 引擎端到端：永远发 tool_calls 的模型也会被软上限收住 ────────────────────

def test_engine_force_answer_end_to_end():
    # 4 条 tool_call 回复（模型永不放弃）+ 1 条给 force_answer 的裸 invoke 回复
    llm = FakeLLM([_tool_msg(call_id=f"c{i}") for i in range(4)]
                  + [AIMessage(content="基于已有信息的收尾回答")])
    graph = build_graph(llm, _dispatcher())

    result = graph.invoke({"messages": [HumanMessage("问")]})

    assert result["messages"][-1].content == "基于已有信息的收尾回答"   # 软上限接管，正常收尾
    assert len(llm.calls) == 5                                          # 4 轮 stream + 1 次裸 invoke
    last = llm.calls[-1]["messages"]
    assert isinstance(last[-1], SystemMessage) and "检索次数上限" in last[-1].content
    assert any(isinstance(m, ToolMessage) and "未执行" in m.content for m in last)  # 悬空已补
