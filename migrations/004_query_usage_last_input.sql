-- ============================================================
-- 004_query_usage_last_input.sql  ·  增量迁移：持久化上下文水位
-- 适用：已执行过 001/002/003 的环境。
-- 幂等：ADD COLUMN IF NOT EXISTS，可重复执行。
--
-- 作用：query_usage 记录"本次 query 最后一轮 LLM 的 input_tokens"。
--       上下文占用率 = last_input_tokens / 输入预算，分子不累加
--       （每轮 input 都是完整重发，最后一轮即当前真实水位）。
--       落库后，前端切换会话 / 刷新页面可直接恢复环形指示器，
--       不必等下一次 LLM 调用返回 usage。
-- 历史行默认 0 → 接口返回 ratio=null → 前端隐藏，不展示假数据。
-- ============================================================

ALTER TABLE query_usage
    ADD COLUMN IF NOT EXISTS last_input_tokens INT NOT NULL DEFAULT 0;
