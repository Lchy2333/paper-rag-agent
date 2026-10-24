"""数据库访问层：users / threads / audit_logs 仓储。

表结构见 migrations/001_init.sql、002_threads_pinned.sql。所有函数都接收
连接池参数，不直接持有全局连接——池由 app.state.pool 提供（main.py lifespan
创建）。测试时可注入假 pool 或 monkeypatch 本模块函数，无需真实数据库。
"""
import json
from typing import Any, Dict, List, Optional, Tuple

# 对外暴露的字段（绝不含 password_hash）
PUBLIC_COLUMNS = "id, username, disabled"


def get_user_by_username(pool, username: str) -> Optional[Dict[str, Any]]:
    """按用户名查用户（含密码哈希，仅登录时用）。"""
    with pool.connection() as conn:
        return conn.execute(
            "SELECT id, username, password_hash, disabled FROM users WHERE username = %s",
            (username,),
        ).fetchone()


def get_user_by_id(pool, user_id: str) -> Optional[Dict[str, Any]]:
    """按 user_id 查用户（含密码哈希；认证中间件校验账号状态用）。"""
    with pool.connection() as conn:
        return conn.execute(
            "SELECT id, username, password_hash, disabled FROM users WHERE id = %s",
            (user_id,),
        ).fetchone()


def create_user(pool, username: str, password_hash: str) -> Dict[str, Any]:
    """创建用户，返回公开字段。"""
    with pool.connection() as conn:
        return conn.execute(
            f"""
            INSERT INTO users (username, password_hash)
            VALUES (%s, %s)
            RETURNING {PUBLIC_COLUMNS}
            """,
            (username, password_hash),
        ).fetchone()


# ---------- threads 仓储（会话归属绑定 + 改名 + 置顶） ----------

def get_thread(pool, thread_id: str) -> Optional[Dict[str, Any]]:
    """按 thread_id 查会话（归属校验用）。"""
    with pool.connection() as conn:
        return conn.execute(
            "SELECT id, user_id, title, pinned, created_at, updated_at"
            " FROM threads WHERE id = %s",
            (thread_id,),
        ).fetchone()


def create_thread(pool, thread_id: str, user_id: str) -> None:
    """新建会话（title 默认为空，用户可后改名）。"""
    with pool.connection() as conn:
        conn.execute(
            "INSERT INTO threads (id, user_id) VALUES (%s, %s)",
            (thread_id, user_id),
        )


def list_threads(pool, user_id: str) -> List[Dict[str, Any]]:
    """列出某用户的会话：置顶优先，其次按更新时间倒序。"""
    with pool.connection() as conn:
        return conn.execute(
            "SELECT id, title, pinned, created_at, updated_at"
            " FROM threads WHERE user_id = %s"
            " ORDER BY pinned DESC, updated_at DESC",
            (user_id,),
        ).fetchall()


def update_thread(
    pool,
    thread_id: str,
    title: Optional[str] = None,
    pinned: Optional[bool] = None,
) -> Dict[str, Any]:
    """更新会话的 title / pinned（只更新传入的字段），并刷新 updated_at。"""
    sets, params = ["updated_at = now()"], []
    if title is not None:
        sets.append("title = %s")
        params.append(title)
    if pinned is not None:
        sets.append("pinned = %s")
        params.append(pinned)
    params.append(thread_id)
    with pool.connection() as conn:
        return conn.execute(
            f"UPDATE threads SET {', '.join(sets)} WHERE id = %s"
            " RETURNING id, title, pinned, created_at, updated_at",
            params,
        ).fetchone()


def delete_thread_row(pool, thread_id: str) -> None:
    """删除 threads 表中的会话记录（checkpoints 由调用方另行清理）。"""
    with pool.connection() as conn:
        conn.execute("DELETE FROM threads WHERE id = %s", (thread_id,))


# ---------- audit_logs 审计 ----------

def add_audit_log(
    pool,
    user_id: str,
    thread_id: str,
    action: str,
    payload: Dict[str, Any],
) -> None:
    """写入审计日志：谁在何时对哪个会话做了什么。"""
    with pool.connection() as conn:
        conn.execute(
            "INSERT INTO audit_logs (user_id, thread_id, action, payload)"
            " VALUES (%s, %s, %s, %s)",
            (user_id, thread_id, action, json.dumps(payload, ensure_ascii=False, default=str)),
        )


# ---------- query_usage 用量仓储（会话累计的事实源，表见 003_query_usage.sql） ----------

def get_session_usage(pool, thread_id: str) -> Tuple[int, int]:
    """会话累计用量（SUM）。请求开始读一次作为本次 query 的基线。"""
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT COALESCE(SUM(input_tokens), 0) AS input_tokens,"
            " COALESCE(SUM(output_tokens), 0) AS output_tokens"
            " FROM query_usage WHERE thread_id = %s",
            (thread_id,),
        ).fetchone()
    if row is None:
        return 0, 0
    # pool 的 row_factory 是 dict_row（main.py），fetchone 返回字典，按列名取值
    return row["input_tokens"], row["output_tokens"]


def record_usage(
    pool,
    thread_id: str,
    user_id: str,
    input_tokens: int,
    output_tokens: int,
    calls: int,
) -> None:
    """每次 query 结束落一行（调用方在 finally 里调，cancel / error 也落）。"""
    with pool.connection() as conn:
        conn.execute(
            "INSERT INTO query_usage (thread_id, user_id, input_tokens, output_tokens, calls)"
            " VALUES (%s, %s, %s, %s, %s)",
            (thread_id, user_id, input_tokens, output_tokens, calls),
        )
