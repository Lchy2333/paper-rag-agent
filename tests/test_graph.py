"""图行为测试（不依赖真实 LLM / RAG 服务）。

用 FakeClient 模拟 openai SDK 的 chat.completions.create 返回，
验证 call_llm → call_tool → 回填 → 结束的完整循环。
"""
import json

from langgraph.graph import END

from agent.graph import build_graph
from agent.nodes import call_llm, call_tool, route_after_llm
from agent.state import AgentState
from core import Dispatcher, ToolRegistry
from core.tool import ToolBase, ToolResult
from langchain_core.messages import HumanMessage, SystemMessage


class _EchoTool(ToolBase):
    """假工具：回显参数。"""

    def name(self) -> str:
        return "echo"

    def build_schema(self):
        return {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "回显输入",
                "parameters": {"type": "object", "properties": {}},
            },
        }

    def execute(self, args):
        return ToolResult(success=True, data={"echo": args.get("msg")})


class _FakeChoice:
    def __init__(self, msg):
        self.message = msg


class _FakeResponse:
    def __init__(self, msg):
        self.choices = [_FakeChoice(msg)]


class _FakeToolCall:
    def __init__(self, name, arguments, id="call_1"):
        self.id = id
        self.type = "function"
        self.function = type("F", (), {"name": name, "arguments": arguments})()


class _FakeMsg:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class FakeClient:
    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = []
        self.chat = _FakeChat(self)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse(self._replies.pop(0))


class _FakeChat:
    def __init__(self, owner):
        self.completions = _FakeCompletions(owner)


class _FakeCompletions:
    def __init__(self, owner):
        self.owner = owner

    def create(self, **kwargs):
        return self.owner.create(**kwargs)


def _build_dispatcher() -> Dispatcher:
    registry = ToolRegistry()
    registry.register(_EchoTool())
    return Dispatcher(registry)


def test_full_loop():
    dispatcher = _build_dispatcher()

    # 第一轮：要求调工具；第二轮：给最终答案
    client = FakeClient([
        _FakeMsg(
            content="",
            tool_calls=[_FakeToolCall("echo", json.dumps({"msg": "hi"}))],
        ),
        _FakeMsg(content="回答：hi"),
    ])

    llm_node = call_llm(client, dispatcher, model="fake")
    tool_node = call_tool(dispatcher)

    state: AgentState = {"messages": []}

    # call_llm 第一轮
    state["messages"] += llm_node(state)["messages"]
    assert route_after_llm(state) == "call_tool"

    # call_tool 执行并回填
    state["messages"] += tool_node(state)["messages"]
    assert state["messages"][-1].content == "{'echo': 'hi'}"

    # call_llm 第二轮（带 tool 结果）
    state["messages"] += llm_node(state)["messages"]
    assert route_after_llm(state) == END
    assert state["messages"][-1].content == "回答：hi"


def test_route_ends_without_tools():
    dispatcher = _build_dispatcher()
    client = FakeClient([_FakeMsg(content="直接回答")])

    llm_node = call_llm(client, dispatcher, model="fake")
    state: AgentState = {"messages": []}
    state["messages"] += llm_node(state)["messages"]
    assert route_after_llm(state) == END


def test_engine_full_loop():
    """用官方 LangGraph 引擎完整跑一遍图（FakeClient 模拟 LLM）。"""
    dispatcher = _build_dispatcher()
    client = FakeClient([
        _FakeMsg(content="", tool_calls=[_FakeToolCall("echo", json.dumps({"msg": "hi"}))]),
        _FakeMsg(content="回答：hi"),
    ])

    graph = build_graph(client, dispatcher, model="fake")
    final_state = graph.invoke({
        "messages": [SystemMessage(content="sys"), HumanMessage(content="hi")],
    })

    contents = [m.content for m in final_state["messages"]]
    assert contents == ["sys", "hi", "", "{'echo': 'hi'}", "回答：hi"]

    # 第二次调用 LLM 时，tools 已带上且 messages 是历史
    tools_arg = client.calls[1]["tools"]
    assert tools_arg[0]["function"]["name"] == "echo"
