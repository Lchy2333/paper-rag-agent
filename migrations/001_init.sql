-- ============================================================
-- 001_init.sql  ·  Agent 独立库初始化
-- 库：paper_rag_agent（与 RAG 的 paper_rag 分库，共用同一 PG 实例）
--
-- 说明：
--   - checkpoints / checkpoint_blobs / checkpoint_writes 三张表
--     由 PostgresSaver.setup() 在应用启动时自动创建，无需在此建。
--   - 本脚本只建 Agent 自管理的三张表：users / threads / audit_logs。
--   - 全部使用 IF NOT EXISTS，幂等，可重复执行。
--
-- 执行方式：Navicat 连上 paper_rag_agent 库直接运行本脚本；
--           部署时作为迁移脚本随代码发版执行（后续可迁到 Alembic）。
-- ============================================================

-- ---------- 用户表：认证 ----------
-- id 为 user_id（UUID，跨库软约束：RAG 侧只存该字符串，不建用户表）
CREATE TABLE IF NOT EXISTS users (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username      TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,            -- bcrypt 哈希，不存明文
    disabled      BOOLEAN NOT NULL DEFAULT FALSE,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------- 会话归属绑定：thread_id ↔ user_id ----------
-- 多用户安全核心：/chat 与 /approve 先查本表校验归属，跨用户访问 403
CREATE TABLE IF NOT EXISTS threads (
    id         TEXT PRIMARY KEY,            -- thread_id（uuid4().hex）
    user_id    UUID NOT NULL REFERENCES users(id),
    title      TEXT NOT NULL DEFAULT '',
    pinned     BOOLEAN NOT NULL DEFAULT FALSE,   -- 置顶会话
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_threads_user
    ON threads(user_id, pinned DESC, updated_at DESC);

-- ---------- 审计日志：审批 + 危险操作留痕 ----------
CREATE TABLE IF NOT EXISTS audit_logs (
    id         BIGSERIAL PRIMARY KEY,
    user_id    UUID NOT NULL,
    thread_id  TEXT NOT NULL,
    action     TEXT NOT NULL,               -- chat / approve / delete_document ...
    payload    JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_audit_user
    ON audit_logs(user_id, created_at DESC);
