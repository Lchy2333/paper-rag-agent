"""线程管理路由：列会话 / 改名 / 置顶 / 删除（全部校验归属）。

多用户隔离关键：所有操作先查 threads 表校验归属——
不存在 → 404，非本人 → 403。
"""
import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request

from app import db
from app.auth import CurrentUser
from app.schemas import (
    DeleteThreadResponse,
    ThreadOut,
    ThreadUsageOut,
    UpdateThreadRequest,
)
from core.usage import context_ratio

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/threads", tags=["threads"])


def _get_owned_thread(pool, thread_id: str, user_id: str) -> dict:
    """取会话并校验归属；不存在/非本人分别抛 404/403。"""
    thread = db.get_thread(pool, thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    if thread["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="无权访问该会话")
    return thread


@router.get("", response_model=List[ThreadOut])
def list_threads(request: Request, user=CurrentUser) -> List[ThreadOut]:
    """列出当前用户的会话：置顶优先，其次按最后活跃倒序。"""
    rows = db.list_threads(request.app.state.pool, user["id"])
    return [
        ThreadOut(
            thread_id=r["id"],
            title=r["title"],
            pinned=r["pinned"],
            created_at=r["created_at"],
            updated_at=r["updated_at"],
        )
        for r in rows
    ]


@router.get("/{thread_id}/usage", response_model=ThreadUsageOut)
def get_thread_usage(
    thread_id: str,
    request: Request,
    user=CurrentUser,
) -> ThreadUsageOut:
    """会话用量 + 上下文占用：切换会话/刷新页面时前端据此恢复展示。

    context_used 取 query_usage 最近一行的 last_input_tokens（当前水位），
    与 SSE usage 事件同一口径，因此恢复出来的环与实时值一致。
    """
    _get_owned_thread(request.app.state.pool, thread_id, user["id"])
    row = db.get_thread_usage(request.app.state.pool, thread_id)
    limits = getattr(request.app.state, "llm_limits", None)
    budget = limits.input_budget if limits else None
    used = row["last_input_tokens"]
    return ThreadUsageOut(
        thread_id=thread_id,
        session_input=row["input_tokens"],
        session_output=row["output_tokens"],
        session_total=row["input_tokens"] + row["output_tokens"],
        context_window=limits.context_window if limits else None,
        context_budget=budget,
        context_used=used,
        context_ratio=context_ratio(used, budget),
    )


@router.patch("/{thread_id}", response_model=ThreadOut)
def update_thread(
    thread_id: str,
    req: UpdateThreadRequest,
    request: Request,
    user=CurrentUser,
) -> ThreadOut:
    """改名 / 置顶。至少传一个字段（title 或 pinned）。"""
    if req.title is None and req.pinned is None:
        raise HTTPException(status_code=422, detail="至少提供一个字段：title 或 pinned")

    _get_owned_thread(request.app.state.pool, thread_id, user["id"])
    updated = db.update_thread(
        request.app.state.pool,
        thread_id,
        title=req.title,
        pinned=req.pinned,
    )
    if updated is None:  # 归属校验后并发删除等极端情况
        raise HTTPException(status_code=404, detail="会话不存在")
    logger.info("thread_updated thread_id=%s user_id=%s", thread_id, user["id"])
    return ThreadOut(
        thread_id=updated["id"],
        title=updated["title"],
        pinned=updated["pinned"],
        created_at=updated["created_at"],
        updated_at=updated["updated_at"],
    )


@router.delete("/{thread_id}", response_model=DeleteThreadResponse)
def delete_thread(
    thread_id: str,
    request: Request,
    user=CurrentUser,
) -> DeleteThreadResponse:
    """删除会话：清理 threads 记录 + 对应 checkpoints（对话内容一并删除）。"""
    _get_owned_thread(request.app.state.pool, thread_id, user["id"])

    db.delete_thread_row(request.app.state.pool, thread_id)
    graph = request.app.state.graph
    if graph is not None and hasattr(graph, "checkpointer"):
        try:
            graph.checkpointer.delete_thread(thread_id)  # 清理对话快照
        except Exception as e:  # 快照清理失败不阻塞删除（表记录已删）
            logger.warning("checkpoint_delete_failed thread_id=%s err=%s", thread_id, e)

    db.add_audit_log(
        request.app.state.pool,
        str(user["id"]),
        thread_id,
        "delete_thread",
        {},
    )
    logger.info("thread_deleted thread_id=%s user_id=%s", thread_id, user["id"])
    return DeleteThreadResponse(thread_id=thread_id)
