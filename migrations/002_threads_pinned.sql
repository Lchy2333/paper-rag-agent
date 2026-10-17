-- ============================================================
-- 002_threads_pinned.sql  ·  增量迁移：threads 表新增 pinned（置顶）列
-- 适用：已执行过 001_init.sql 的环境；新环境直接跑 001 即可（已含本列）。
-- 幂等：IF NOT EXISTS，可重复执行。
-- ============================================================

ALTER TABLE threads
    ADD COLUMN IF NOT EXISTS pinned BOOLEAN NOT NULL DEFAULT FALSE;

-- 排序索引带上置顶：置顶优先，其次按更新时间倒序
DROP INDEX IF EXISTS idx_threads_user;
CREATE INDEX idx_threads_user
    ON threads(user_id, pinned DESC, updated_at DESC);
