"""图行为测试（不依赖真实 LLM / RAG 服务）。

用 FakeLLM 模拟 langchain_openai.ChatOpenAI 的 bind_tools().invoke() 返回，
验证 call_llm → call_tool → 回填 → 结束的完整循环。
"""
import json

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage
from langgraph.graph import END

from agent.graph import build_graph
from agent.nodes import call_llm, call_tool, route_after_llm
from agent.state import AgentState
from core import Dispatcher, ToolRegistry
from core.tool import ToolBase, ToolResult


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

    def execute(self, args, user_id=None):
        return ToolResult(success=True, data={"echo": args.get("msg")})


class FakeLLM:
    """模拟 ChatOpenAI：bind_tools 记录 schema，invoke/stream 返回预设的 AIMessage 序列。"""

    def __init__(self, replies):
        # replies: list of AIMessage，按调用顺序逐个返回
        self._replies = list(replies)
        self.calls = []
        self.last_bound_tools = None

    def bind_tools(self, tools):
        self.last_bound_tools = tools
        return self

    def invoke(self, messages):
        self.calls.append({"messages": messages, "tools": self.last_bound_tools})
        return self._replies.pop(0)

    def stream(self, messages):
        self.calls.append({"messages": messages, "tools": self.last_bound_tools})
        reply = self._replies.pop(0)
        # 模拟流式：整段回复作为一个 chunk 吐出来（保留 tool_calls，供路由判断）
        yield AIMessageChunk(content=reply.content, tool_calls=reply.tool_calls)


def _tool_call_message(name: str, args: dict) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"id": "call_1", "name": name, "args": args}])


def _build_dispatcher() -> Dispatcher:
    registry = ToolRegistry()
    registry.register(_EchoTool())
    return Dispatcher(registry)


def test_full_loop():
    dispatcher = _build_dispatcher()

    llm = FakeLLM([
        _tool_call_message("echo", {"msg": "hi"}),
        AIMessage(content="回答：hi"),
    ])

    llm_node = call_llm(llm, dispatcher)
    tool_node = call_tool(dispatcher)

    state: AgentState = {"messages": []}

    # call_llm 第一轮
    state["messages"] += llm_node(state)["messages"]
    assert route_after_llm(state) == "guard"

    # call_tool 执行并回填
    state["messages"] += tool_node(state)["messages"]
    assert state["messages"][-1].content == "{'echo': 'hi'}"

    # call_llm 第二轮（带 tool 结果）
    state["messages"] += llm_node(state)["messages"]
    assert route_after_llm(state) == END
    assert state["messages"][-1].content == "回答：hi"


def test_route_ends_without_tools():
    dispatcher = _build_dispatcher()
    llm = FakeLLM([AIMessage(content="直接回答")])

    llm_node = call_llm(llm, dispatcher)
    state: AgentState = {"messages": []}
    state["messages"] += llm_node(state)["messages"]
    assert route_after_llm(state) == END


def test_engine_full_loop():
    """用官方 LangGraph 引擎完整跑一遍图（FakeLLM 模拟 LLM）。"""
    dispatcher = _build_dispatcher()
    llm = FakeLLM([
        _tool_call_message("echo", {"msg": "hi"}),
        AIMessage(content="回答：hi"),
    ])

    graph = build_graph(llm, dispatcher)
    final_state = graph.invoke({
        "messages": [SystemMessage(content="sys"), HumanMessage(content="hi")],
    })

    contents = [m.content for m in final_state["messages"]]
    assert contents == ["sys", "hi", "", "{'echo': 'hi'}", "回答：hi"]

    # 第二次调用 LLM 时，tools 已带上且 messages 是历史
    tools_arg = llm.calls[1]["tools"]
    assert tools_arg[0]["function"]["name"] == "echo"
