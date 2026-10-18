# paper-rag-agent（FastAPI + LangGraph）
# 多阶段：pip 安装依赖层（缓存友好）→ 运行时
FROM python:3.12-slim AS deps
WORKDIR /app
COPY requirements.txt .
# 国内 PyPI 镜像（pypi.org 在国内慢/不稳定）
RUN pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt

FROM python:3.12-slim AS runtime
WORKDIR /app
# apt 换国内源（deb.debian.org 在国内慢/超时）+ 装 curl（compose healthcheck 探活用）
RUN sed -i 's|deb.debian.org|mirrors.aliyun.com|g; s|security.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || \
    sed -i 's|deb.debian.org|mirrors.aliyun.com|g; s|security.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=deps /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin
COPY . .

EXPOSE 8001
# 单 worker：多 worker 并发 PostgresSaver.setup() 会撞表冲突；
# agent 是 I/O 等待型（等 LLM），单 worker + FastAPI 异步已够用
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
