# paper-rag-agent

学术论文检索 + RAG agent：**FastAPI + LangGraph** 服务。

由 paper-rag-agent-MVP 升级而来：自研 StateGraph 引擎替换为官方 `langgraph`，新增 FastAPI 服务层。
已落地**多用户隔离 + 对话持久化**（PostgreSQL checkpointer + JWT 认证 + 线程归属绑定）+ **BFF 文档管理转发**（前端经 agent 中转访问 RAG）。

> **生产部署**：已容器化（`Dockerfile`，单 worker）。完整编排（postgres + rag + agent + fronted）
> 在 [paper-rag-deploy](https://github.com/Lchy2333/paper-rag-deploy) 仓，`docker compose up -d --build` 一键拉起。

## 项目结构

```
app/                  # FastAPI 服务层
  main.py               # 应用入口：lifespan 构建 graph / PostgresSaver / 挂路由
  config.py             # pydantic-settings 读 .env（含 LLM_* / DATABASE_URL / JWT_SECRET，带启动校验）
  schemas.py            # Pydantic 请求/响应模型
  auth.py               # bcrypt 密码哈希 + JWT 签发/验签 + get_current_user 依赖
  db.py                 # users / threads / audit_logs 仓储
  errors.py             # 异常分类（LLM 错误 → 502/429/500 语义化响应）
  api/chat.py           # POST /chat（认证 + 线程归属校验）
  api/approve.py        # POST /approve（HITL 审批，归属校验 + 审计）
  api/auth.py           # POST /auth/register / /auth/login
  api/threads.py        # GET/PATCH/DELETE /threads（列表 / 改名 / 置顶 / 删除）
  api/documents.py      # 文档管理转发（BFF：前端经 agent 中转 RAG，可信身份注入 + 删除审计）
  api/health.py         # GET /health（分层：status 存活 / checks 就绪）
  api/llm_check.py      # POST /api/llm/check（LLM 就绪验证，BYOK 复用）
agent/                # LangGraph 层
  state.py              # AgentState（messages / route 通道）
  nodes.py              # call_llm / call_tool / guard_node（审批；call_llm 记录 LLM 健康被动信号）
  graph.py              # 官方 StateGraph 组装 ReAct 图（含 HITL 护栏）
core/                 # 核心抽象层（与业务无关）
  tool.py / registry.py / dispatcher.py / retry.py
  rag_proxy.py          # BFF 转发器：注入可信 X-User-ID + X-API-Key，上传大超时
  llm_probe.py          # LLM 就绪探针（max_tokens=1）+ 被动信号 tracker
tools/                # 具体工具
  RAGtools.json         # RAG 接口声明（一个接口 = 一段配置）
  http_tool.py          # 通用 HTTP 工具（自动注入 X-User-ID 头）
migrations/           # 数据库迁移脚本（001 建表 / 002 置顶列）
tests/                # 测试
```

## 图结构

```
START → call_llm ──(有 tool_calls)──► guard ──(approve/noop)──► call_tool
              │                           └────(reject)──► call_llm（跳过工具）
              └─────(无 tool_calls)────────────────────► END
```

## 架构总览

```
用户 ──► Agent 服务（多副本，无状态编排）
          ├─ /auth/register · /auth/login   → 签发 JWT
          ├─ get_current_user               → 验签得 user_id（可信）
          ├─ /chat · /approve · /threads    → threads 归属校验（越权 403）
          ├─ /api/documents/*               → BFF 转发 RAG（可信身份 + 服务端持有 API Key）
          ├─ /health                        → 分层健康检查（status 存活 / checks 就绪）
          ├─ /api/llm/check                 → LLM 就绪验证（BYOK 复用）
          ├─ PostgresSaver                  → 对话落盘（库 paper_rag_agent）
          └─ HttpTool → X-API-Key + X-User-ID
                │
                ▼
        RAG 服务（Go）：UserContext 读 X-User-ID → 文档级隔离
```

> **BFF 原则**：前端只连 Agent（唯一入口）。文档管理经 `/api/documents/*` 由 Agent 转发到 RAG，
> `X-User-ID` 从 JWT 验签得到（前端无法伪造）、`X-API-Key` 只在 Agent 进程持有（浏览器不可见）。
> 前端健康状态也读 Agent `/health` 的 `checks`，不直连 RAG（见 `app/api/health.py`）。

## 快速开始

### 1. 准备 PostgreSQL

- 建库：`paper_rag_agent`（与 RAG 的 `paper_rag` 分库，共用同一 PG 实例）
- 执行迁移（幂等，可重复跑）：
  - `migrations/001_init.sql`：users / threads / audit_logs
  - `migrations/002_threads_pinned.sql`：threads 加 pinned 列
- `checkpoints` 系列表由 `PostgresSaver.setup()` 启动时自动创建

### 2. 配置 .env

```bash
copy .env.example .env
# 必填：LLM_BASE_URL / LLM_API_KEY / LLM_MODEL、DATABASE_URL、JWT_SECRET
```

关键项：

```
# 环境标识：dev（仅告警缺失）/ prod（关键配置缺失拒绝启动）
APP_ENV=dev
# LLM 接入（OpenAI 兼容协议，可指向任意兼容服务：火山方舟 / DeepSeek / OpenAI…）
LLM_BASE_URL=https://ark.cn-beijing.volces.com/api/plan/v3/
LLM_API_KEY=your-llm-api-key
LLM_MODEL=deepseek-v4-flash
RAG_BASE_URL=http://localhost:8080
RAG_API_KEY=your-rag-api-key
DATABASE_URL=postgresql://postgres:密码@localhost:5432/paper_rag_agent
JWT_SECRET=<随机串，python -c "import secrets; print(secrets.token_urlsafe(48))">
JWT_EXPIRE_MINUTES=10080
```

> ⚠️ **启动校验**（`app/config.py` `ensure_runtime()`）：LLM 三要素（base_url/api_key/model）缺失 → 拒绝启动；
> `APP_ENV=prod` 时 JWT_SECRET / RAG_API_KEY / 生产库缺失也拒绝启动。密钥一律零默认值，不提供兜底 key。

### 3. 创建用户并启动

```bash
# 建库建表后，用注册接口或直接插库创建用户（如 admin）
uvicorn app.main:app --reload --port 8001
```

## 测试

```bash
.venv\Scripts\activate
pytest tests\ -v
```

测试用 FakeClient 模拟 LLM，不依赖真实网络与数据库。

## API

| 端点 | 方法 | 认证 | 说明 |
|---|---|---|---|
| `/health` | GET | 无 | 分层健康检查：`status`（存活=agent+db）供编排 / `checks`（db/rag/llm 就绪）供前端 |
| `/api/auth/register` | POST | 无 | 注册（201 / 403 / 409 / 422；`REGISTRATION_ENABLED=false` 时 403 关闭） |
| `/api/auth/login` | POST | 无 | 登录 → JWT（200 / 401 / 403） |
| `/api/chat` | POST | Bearer | 对话（新会话 / 续聊 / 审批挂起） |
| `/api/approve` | POST | Bearer | HITL 审批（越权 403 / 不存在 404） |
| `GET /api/threads` | GET | Bearer | 列自己的会话（置顶优先） |
| `PATCH /api/threads/{id}` | PATCH | Bearer | 改名 / 置顶 |
| `DELETE /api/threads/{id}` | DELETE | Bearer | 删除会话（连带对话快照） |
| `GET /api/documents` | GET | Bearer | 文档列表（BFF 转发 RAG） |
| `POST /api/documents` | POST | Bearer | 上传 PDF（BFF 转发，大超时；RAG 异步摄入） |
| `GET /api/documents/{id}` | GET | Bearer | 文档详情（BFF 转发） |
| `POST /api/documents/{id}/rechunk` | POST | Bearer | 重新分块（BFF 转发） |
| `DELETE /api/documents/{id}` | DELETE | Bearer | 删除文档（BFF 转发 + 审计留痕） |
| `POST /api/llm/check` | POST | Bearer | LLM 就绪验证（BYOK：用户自测 base_url/api_key/model） |

**认证方式**：所有业务接口需 `Authorization: Bearer <JWT>` 头；JWT 由 `/api/auth/login` 签发。

```bash
# 1. 登录拿 token
curl -s -X POST http://localhost:8001/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"..."}'

# 2. 带 token 对话
curl -s -X POST http://localhost:8001/api/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"query":"当前数据库中有哪些已向量化的文档？"}'
```

**多用户隔离**：会话按 `threads` 表绑定 `thread_id ↔ user_id`，跨用户访问返回 403；
`user_id` 经 LangGraph config → Dispatcher → HttpTool 注入 `X-User-ID` 头传给 RAG，
实现文档级隔离（admin 为全库超级用户，由 RAG 侧 `server.admin_users` 配置）。

## RAG 接口接入

在 `tools/RAGtools.json` 中新增一段配置即可，无需改代码。配置项：

- `description`：给 LLM 看的函数描述
- `path`：接口路径，`/api/v1/documents/:id` 形式的 `:param` 会替换为参数值
- `method`：HTTP 方法（GET / POST / DELETE / ...）
- `schema`：OpenAI 工具调用参数 schema（`type` / `properties` / `required`）

> 注意：`user_id` 不要写进 schema——它由 `HttpTool` 从认证上下文自动注入 `X-User-ID` 头，
> LLM 不应看到、也无法伪造。`ask_rag` 工具已移除（ask 是 RAG 侧全局检索接口，仅管理员直接调用）。
