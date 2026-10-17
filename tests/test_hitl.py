"""HITL（人工审批）回归测试。

覆盖：guard 挂起 / approve 放行 / reject 阻止 / 只读工具不挂起 /
get_state 检测挂起状态 / guard 路由映射。全部用假对象，不依赖真实网络与 LLM。
"""
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from agent.graph import build_graph
from agent.nodes import route_after_guard
from core import Dispatcher, ToolRegistry
from core.tool import ToolBase, ToolResult


class _FakeLLM:
    """模拟 ChatOpenAI：按预设回复序列逐个返回。"""

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = []

    def bind_tools(self, tools):
        self._tools = tools
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        return self._replies.pop(0)


def _tool_call(name, args):
    return AIMessage(content="", tool_calls=[
        {"id": "call_1", "name": name, "args": args}])


class _DeleteTool(ToolBase):
    """危险工具：requires_approval=True，记录是否真的被执行。"""

    requires_approval = True

    def __init__(self):
        self.called = 0

    def name(self):
        return "delete_doc"

    def build_schema(self):
        return {
            "type": "function",
            "function": {
                "name": "delete_doc",
                "description": "",
                "parameters": {"type": "object", "properties": {}},
            },
        }

    def execute(self, args, user_id=None):
        self.called += 1
        return ToolResult(success=True, data={"deleted": args["id"]})


class _EchoTool(ToolBase):
    """只读工具：requires_approval=False。"""

    def name(self):
        return "echo"

    def build_schema(self):
        return {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "",
                "parameters": {"type": "object", "properties": {}},
            },
        }

    def execute(self, args, user_id=None):
        return ToolResult(success=True, data={"echo": args.get("msg")})


def _build_graph(tool, llm):
    registry = ToolRegistry()
    registry.register(tool)
    dispatcher = Dispatcher(registry)
    return build_graph(llm, dispatcher, MemorySaver())


def test_hitl_approve_allows_execution():
    tool = _DeleteTool()
    llm = _FakeLLM([
        _tool_call("delete_doc", {"id": "A"}),
        AIMessage(content="已删除"),
    ])
    graph = _build_graph(tool, llm)
    config = {"configurable": {"thread_id": "t1"}}

    # 第一段：挂起，工具未执行
    r1 = graph.invoke({"messages": [HumanMessage("删A")]}, config=config)
    assert "__interrupt__" in r1
    assert r1["__interrupt__"][0].value == {"tool": "delete_doc", "args": {"id": "A"}}
    assert tool.called == 0

    # 第二段：审批通过 → 工具执行 → LLM 总结
    r2 = graph.invoke(Command(resume="allow"), config=config)
    assert tool.called == 1
    assert r2["messages"][-1].content == "已删除"


def test_hitl_reject_blocks_execution():
    tool = _DeleteTool()
    llm = _FakeLLM([
        _tool_call("delete_doc", {"id": "A"}),
        AIMessage(content="好的，不删"),
    ])
    graph = _build_graph(tool, llm)
    config = {"configurable": {"thread_id": "t2"}}

    graph.invoke({"messages": [HumanMessage("删A")]}, config=config)
    r2 = graph.invoke(Command(resume="reject"), config=config)

    assert tool.called == 0  # 拒绝后工具绝不执行
    assert r2["messages"][-1].content == "好的，不删"


def test_hitl_read_only_tool_no_interrupt():
    tool = _EchoTool()
    llm = _FakeLLM([
        _tool_call("echo", {"msg": "hi"}),
        AIMessage(content="echo: hi"),
    ])
    graph = _build_graph(tool, llm)
    config = {"configurable": {"thread_id": "t3"}}

    r = graph.invoke({"messages": [HumanMessage("hi")]}, config=config)
    assert "__interrupt__" not in r
    assert r["messages"][-1].content == "echo: hi"


def test_get_state_detects_pending_interrupt():
    tool = _DeleteTool()
    llm = _FakeLLM([_tool_call("delete_doc", {"id": "A"}), AIMessage(content="x")])
    graph = _build_graph(tool, llm)
    config = {"configurable": {"thread_id": "t4"}}

    graph.invoke({"messages": [HumanMessage("删A")]}, config=config)
    snap = graph.get_state(config)
    assert snap.interrupts          # 挂起中 → 非空
    assert snap.next                # 还有待执行任务

    graph.invoke(Command(resume="allow"), config=config)
    snap2 = graph.get_state(config)
    assert not snap2.interrupts     # 完成后 → 空
    assert not snap2.next


def test_route_after_guard_mapping():
    assert route_after_guard({"route": "approve"}) == "approve"
    assert route_after_guard({"route": "reject"}) == "reject"
    assert route_after_guard({"route": "noop"}) == "noop"
    assert route_after_guard({}) == "approve"  # 防御默认值
