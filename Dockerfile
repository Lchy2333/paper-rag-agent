# paper-rag-agent（FastAPI + LangGraph）
# 多阶段：pip 安装依赖层（缓存友好）→ 运行时
FROM python:3.12-slim AS deps
WORKDIR /app
COPY requirements.txt .
# 国内 PyPI 镜像（pypi.org 在国内慢/不稳定）
RUN pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt

FROM python:3.12-slim AS runtime
WORKDIR /app
COPY --from=deps /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin
COPY . .

EXPOSE 8001
# 单 worker：多 worker 并发 PostgresSaver.setup() 会撞表冲突；
# agent 是 I/O 等待型（等 LLM），单 worker + FastAPI 异步已够用
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
