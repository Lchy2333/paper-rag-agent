"""审批路由：用 Command(resume) 恢复被 interrupt 挂起的会话。

HITL 第二段：/chat 返回 awaiting_approval 后，客户端收集用户选择，
再调本接口传入 thread_id + approved，图从上次中断点恢复执行。
"""
from fastapi import APIRouter, Request
from langgraph.types import Command

from app.schemas import ApproveRequest

router = APIRouter()


@router.post("/approve")
def approve(req: ApproveRequest, request: Request):
    graph = request.app.state.graph
    config = {"configurable": {"thread_id": req.thread_id}}
    request.state.thread_id = req.thread_id  # 捎带给 exception_handler 做日志关联
    resume_value = "allow" if req.approved else "reject"

    result = graph.invoke(Command(resume=resume_value), config=config)

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
