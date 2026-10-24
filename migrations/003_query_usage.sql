-- ============================================================
-- 003_query_usage.sql  ·  增量迁移：token 用量明细表
-- 适用：已执行过 001/002 的环境；新环境跑 001 后同样执行本脚本。
-- 幂等：IF NOT EXISTS，可重复执行。
--
-- 作用：每次 query 结束时落一行（/chat 与 /chat/stream 的 finally），
--       session（会话）累计 = 按 thread_id SUM 本表 —— 这是会话用量的
--       唯一事实源：即使 state 被压缩/裁剪，账也不丢。
-- ============================================================

CREATE TABLE IF NOT EXISTS query_usage (
    id            BIGSERIAL PRIMARY KEY,
    thread_id     TEXT NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
    user_id       UUID NOT NULL,
    input_tokens  INT NOT NULL DEFAULT 0,
    output_tokens INT NOT NULL DEFAULT 0,
    calls         INT NOT NULL DEFAULT 0,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_query_usage_thread
    ON query_usage(thread_id, created_at);
