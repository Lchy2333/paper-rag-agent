"""图行为测试（不依赖真实 LLM / RAG 服务）。

用 FakeLLM 模拟 langchain_openai.ChatOpenAI 的 bind_tools().invoke() 返回，
验证 call_llm → call_tool → 回填 → 结束的完整循环。
"""
import json

from langchain_core.language_models.chat_models import BaseChatModel, ChatGenerationChunk
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.graph import END

from agent.graph import build_graph
from agent.nodes import call_llm, call_tool, route_after_llm
from agent.state import AgentState
from core import Dispatcher, ToolRegistry
from core.tool import ToolBase, ToolResult
from core.usage import UsageCallbackHandler


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


class _UsageLLM(BaseChatModel):
    """真实走 LangChain 回调管线的假模型：流式 chunk 带 usage_metadata。

    FakeLLM 不触发 on_llm_end（不是 Runnable），测 callback 采集必须用这个。
    """

    def __init__(self, replies):
        super().__init__()
        self._replies = list(replies)

    @property
    def _llm_type(self) -> str:
        return "fake-usage"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        """BaseChatModel.bind_tools 默认 NotImplementedError，重写为返回自身。"""
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self._replies.pop(0))])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        yield ChatGenerationChunk(message=self._replies.pop(0))


def test_usage_callback_accumulates_multi_round():
    """方案 B：UsageCallbackHandler 累计一轮 query 里多次 LLM 调用（工具循环两轮）。"""
    dispatcher = _build_dispatcher()
    llm = _UsageLLM([
        AIMessageChunk(
            content="",
            tool_calls=[{"id": "call_1", "name": "echo", "args": {"msg": "hi"}}],
            usage_metadata={"input_tokens": 15, "output_tokens": 5, "total_tokens": 20},
        ),
        AIMessageChunk(
            content="回答：hi",
            usage_metadata={"input_tokens": 28, "output_tokens": 20, "total_tokens": 48},
        ),
    ])

    graph = build_graph(llm, dispatcher)
    handler = UsageCallbackHandler("t1", "u1", session_input=100, session_output=30)
    graph.invoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"callbacks": [handler]},
    )

    # 两轮 LLM 调用全部累计：input 15+28，output 5+20
    assert handler.usage.input_tokens == 43
    assert handler.usage.output_tokens == 25
    assert handler.usage.calls == 2
    snap = handler.snapshot()
    assert snap["query_total"] == 68
    # 会话累计 = 基线 + 本次
    assert snap["session_input"] == 143
    assert snap["session_output"] == 55
    assert snap["session_total"] == 198
