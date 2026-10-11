# paper-rag-agent

学术论文检索 + RAG agent：**FastAPI + LangGraph** 服务。

由 paper-rag-agent-MVP 升级而来：自研 StateGraph 引擎替换为官方 `langgraph`，新增 FastAPI 服务层。

## 项目结构

```
app/                  # FastAPI 服务层
  main.py               # 应用入口：lifespan 构建 graph、挂路由
  config.py             # pydantic-settings 读 .env
  schemas.py            # Pydantic 请求/响应模型
  api/chat.py           # POST /chat
agent/                # LangGraph 层
  state.py              # AgentState（add_messages reducer）
  nodes.py              # call_llm / call_tool / route_after_llm
  graph.py              # 官方 StateGraph 组装 ReAct 图
core/                 # 核心抽象层（与业务无关）
  tool.py / registry.py / dispatcher.py / retry.py
tools/                # 具体工具
  RAGtools.json         # RAG 接口声明（一个接口 = 一段配置）
  http_tool.py          # 通用 HTTP 工具（配置驱动，无需子类）
tests/                # 测试
```

## 图结构

```
START → call_llm ──(有 tool_calls)──► call_tool
              │                         │
              └─────(无 tool_calls)─────┘
                         ▼
                        END
```

## 快速开始

```bash
# 1. 创建虚拟环境
uv venv .venv
.venv\Scripts\activate

# 2. 安装依赖
uv pip install -r requirements.txt

# 3. 配置 .env
copy .env.example .env
# 编辑 .env：ARK_API_KEY、RAG_BASE_URL 等

# 4. 启动服务
uvicorn app.main:app --reload --port 8001
```

## 测试

```bash
.venv\Scripts\activate
pytest tests\ -v
```

测试用 FakeClient 模拟 LLM，不依赖真实网络。

## API

| 端点 | 方法 | 说明 |
|---|---|---|
| `/health` | GET | 健康检查 |
| `/api/chat` | POST | 对话 |

```bash
curl -X POST http://localhost:8001/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "当前数据库中有哪些已向量化的文档？"}'
```

## RAG 接口接入

在 `tools/RAGtools.json` 中新增一段配置即可，无需改代码。配置项：

- `description`：给 LLM 看的函数描述
- `path`：接口路径，`/api/v1/documents/:id` 形式的 `:param` 会替换为参数值
- `method`：HTTP 方法（GET / POST / DELETE / ...）
- `schema`：OpenAI 工具调用参数 schema（`type` / `properties` / `required`）
