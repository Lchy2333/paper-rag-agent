"""审批路由：用 Command(resume) 恢复被 interrupt 挂起的会话。

HITL 第二段：/chat 返回 awaiting_approval 后，客户端收集用户选择，
再调本接口传入 thread_id + approved，图从上次中断点恢复执行。
多用户安全：先查 threads 表校验归属，非本人 → 403。
"""
import logging

from fastapi import APIRouter, HTTPException, Request
from langgraph.types import Command

from app import db
from app.auth import CurrentUser
from app.schemas import ApproveRequest

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/approve")
def approve(req: ApproveRequest, request: Request, user=CurrentUser):
    graph = request.app.state.graph

    # 归属校验：不能审批别人的挂起操作
    thread = db.get_thread(request.app.state.pool, req.thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    if thread["user_id"] != user["id"]:
        raise HTTPException(status_code=403, detail="无权操作该会话")

    # user_id 来自 JWT 验签（user["id"]），不可取自请求体（前端可伪造，见避坑 #18）
    config = {"configurable": {"thread_id": req.thread_id, "user_id": str(user["id"])}}
    request.state.thread_id = req.thread_id  # 捎带给 exception_handler 做日志关联
    request.state.user_id = str(user["id"])
    resume_value = "allow" if req.approved else "reject"

    logger.info("approve thread_id=%s user_id=%s approved=%s", req.thread_id, user["id"], req.approved)
    result = graph.invoke(Command(resume=resume_value), config=config)

    # 审计：审批动作留痕（谁、何时、批/拒）
    db.add_audit_log(
        request.app.state.pool,
        str(user["id"]),
        req.thread_id,
        "approve",
        {"approved": req.approved},
    )

    # 若审批后还有下一个待审批操作，继续返回等待态
    if "__interrupt__" in result:
        intr = result["__interrupt__"][0]
        return {
            "status": "awaiting_approval",
            "thread_id": req.thread_id,
            "approval_payload": intr.value,
        }
    return {
        "status": "done",
        "reply": result["messages"][-1].content,
        "thread_id": req.thread_id,
    }
