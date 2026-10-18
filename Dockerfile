# paper-rag-agent（FastAPI + LangGraph）
# 多阶段：pip 安装依赖层（缓存友好）→ 运行时
FROM python:3.12-slim AS deps
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.12-slim AS runtime
WORKDIR /app
COPY --from=deps /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin
COPY . .

EXPOSE 8001
# 多 worker：PostgresSaver 共享 DB（架构支持多副本），LLMHealthTracker 进程内单例（可接受）
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "2"]
