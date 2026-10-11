# Agent.md — paper-rag-agent 项目上下文

> 本文件为下一轮对话提供完整上下文。阅读后可理解项目全貌、架构决策、当前状态与后续方向。

## 1. 项目定位

学术论文检索 + RAG agent 服务：**FastAPI + LangGraph**。
用户对其**私有论文库**提问，agent 通过 ReAct 循环调用 RAG 服务的 HTTP 接口（配置驱动）检索、综合回答。

- 仓库路径：`D:\Project\paper-rag-agent`
- 由 `D:\Project\paper-rag-agent-MVP`（自研 StateGraph 的最小原型）升级而来
- 已提交 commit `ec1b372`，git 已初始化，**未推送远程**

## 2. 技术栈与版本（实测）

| 依赖 | 版本 | 说明 |
|---|---|---|
| langgraph | 1.2.14 | 图引擎（官方） |
| langchain-core | 1.6.9 | 消息对象（作为 langgraph 传递依赖） |
| fastapi | 0.143.0 | Web 服务 |
| uvicorn | 0.54.0 | ASGI 服务器 |
| pydantic-settings | 2.15.0 | .env 配置 |
| openai | 3.28.0 | LLM SDK（火山方舟 Agent Plan） |
| requests | 2.34.2 | RAG HTTP 工具 |
| Python | 3.14.4 | 系统 `C:\Python314`，venv 在 `.venv` |

**⚠️ 新版 API 差异（已适配，勿回退）**
- langgraph 1.2：`add_conditional_edge` → **`add_conditional_edges`**（复数）
- langgraph 1.2：`END` 常量值为 `"__end__"`（小写）；`START` 为 `"__start__"`
- openai 3.x：`api_key` 不能为空，无 key 直接抛 `OpenAIError`；测试需注入假 key

## 3. 目录结构

```
paper-rag-agent/
├── app/                  # FastAPI 服务层
│   ├── main.py             # 入口：lifespan 构建、挂路由
│   ├── config.py           # Settings（pydantic-settings，读 .env）
│   ├── schemas.py          # ChatRequest / ChatResponse
│   ├── deps.py             # get_graph 依赖注入
│   └── api/chat.py         # POST /api/chat
├── agent/                # LangGraph 层
│   ├── state.py            # AgentState（TypedDict + add_messages）
│   ├── nodes.py            # call_llm / call_tool / route_after_llm
│   └── graph.py            # 官方 StateGraph 组装
├── core/                 # 核心抽象层（与业务无关）
│   ├── tool.py             # ToolBase / ToolResult
│   ├── registry.py         # ToolRegistry（注册/查工具/列 schema）
│   ├── dispatcher.py       # Dispatcher（查工具→重试执行→ToolResult）
│   └── retry.py            # retry_call / is_transient
├── tools/
│   ├── http_tool.py        # HttpTool（配置驱动，一个配置=一个工具）
│   └── RAGtools.json       # 8 个 RAG 接口声明
├── tests/
│   └── test_graph.py       # FakeClient 模拟 LLM 的图集成测试（3 个）
├── .env.example
├── requirements.txt
└── README.md
```

## 4. 核心架构

### 4.1 图结构（ReAct 循环）

```
START
  │
  ▼
call_llm ──(有 tool_calls)──► call_tool
  │                              │
  └─────(无 tool_calls)──────────┘
                 │
                 ▼
                END
```

### 4.2 数据流

1. `POST /api/chat` → `app/api/chat.py` 构造初始 state：`[SystemMessage, HumanMessage]`
2. `graph.invoke(initial_state)` → 引擎从 START 执行
3. `call_llm` 节点：把 state 消息转 OpenAI dict → 调 LLM（`tool_choice=auto`，附上全部工具 schema）→ 返回 `AIMessage`（可能有 `tool_calls`）
4. `route_after_llm` 条件边：有 tool_calls → `call_tool`；否则 → END
5. `call_tool` 节点：遍历 `tool_calls` → `dispatcher.dispatch()` 执行 → 返回 `ToolMessage` 回填
6. 回到 `call_llm`（带工具结果继续思考），直到模型给出最终答案

### 4.3 关键机制

- **State**：`AgentState.messages` 用 `Annotated[list, add_messages]`，节点返回增量消息，引擎自动合并（`agent/state.py:12`）
- **消息载体**：节点内用 LangChain 消息对象（System/Human/AI/Tool），调 openai SDK 前经 `to_openai_dicts()` 转换（`agent/nodes.py:27`），转换做 OpenAI 兼容的 role 映射与 tool_calls 归一化
- **工具配置驱动**：`tools/RAGtools.json` 里 8 个接口声明（health_check / test_connection / upload_documents / documents_list / find_doc_by_id / rechunk_document / delete_document / ask_rag），加新接口=加一段 JSON，不改代码
- **HttpTool**（`tools/http_tool.py:9`）：一个通用类实例化出所有 RAG 工具，支持 `:param` 路径替换、GET query / 其他 body、30s 超时
- **重试**：Dispatcher 用 `retry_call` + `is_transient`（超时/连接/限流/5xx 才重试），`core/retry.py:26`

## 5. 配置（.env）

```ini
ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/plan/v3   # 火山方舟 Agent Plan 接入点
ARK_API_KEY=...                                              # plan API key
ARK_MODEL=deepseek-v4-flash                                  # plan 支持的模型
RAG_BASE_URL=http://localhost:8080                           # RAG 服务地址
```

- 读入 `app/config.py` 的 `Settings`（pydantic-settings，`extra="ignore"`）
- system_prompt 也在 config 中（论文检索助手工作流/回答原则/输出风格）

## 6. 运行与测试

```bash
# 环境
cd D:\Project\paper-rag-agent
.venv\Scripts\activate

# 首次配置
copy .env.example .env   # 填 ARK_API_KEY、RAG_BASE_URL

# 启动
uvicorn app.main:app --reload --port 8001

# 测试（不依赖网络，FakeClient 模拟 LLM）
pytest tests\ -v          # 3 passed

# 验证启动
curl http://localhost:8001/health
curl -X POST http://localhost:8001/api/chat -H "Content-Type: application/json" -d '{"query":"当前数据库中有哪些已向量化的文档？"}'
```

## 7. 测试策略

`tests/test_graph.py` 不依赖真实 LLM/RAG：
- `FakeClient` 模拟 `openai` SDK 的 `chat.completions.create` 返回值（可编排多轮回复序列）
- `_EchoTool` 假工具回显参数
- 3 个测试：节点级完整循环、无工具时路由、官方引擎完整 invoke（断言消息序列 `["sys","hi","","{'echo': 'hi'}","回答：hi"]`）

## 8. 已知边界 / 后续方向

- **多轮会话未做**：`thread_id` 字段已在 `app/schemas.py:9` 预留但未接 checkpointer。下一步：`agent/graph.py` 里 `graph.compile(checkpointer=MemorySaver())`，`chat.py` 用 `graph.invoke(..., config={"configurable": {"thread_id": ...}})` 即可获得多轮记忆
- **流式未做**：可加 `POST /chat/stream` 用 `astream_events` 做 SSE
- **无鉴权**：`/api/chat` 目前无 auth
- **langchain 工具层未引入**：工具仍走自研 `Dispatcher`，节点内调用 openai SDK（不依赖 langchain 的 Tool/ToolNode）
- **未推送远程**：git 已 init、已 commit，无 remote

## 9. 与本仓库无关的上下文提醒

- 项目演进史：`paper-rag-agent-MVP`（天气 demo → 配置驱动 HttpTool → 自研 StateGraph）→ 本仓库（官方 LangGraph + FastAPI）
- 火山方舟 Agent Plan：LLM 走 `/api/plan/v3`，用**模型名**（如 deepseek-v4-flash）而非 ep-xxx endpoint
