"""对话路由。"""
import uuid

from fastapi import APIRouter, HTTPException, Request
from langchain_core.messages import HumanMessage, SystemMessage

from app.schemas import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request) -> ChatResponse:
    graph = request.app.state.graph
    system_prompt = request.app.state.system_prompt

    thread_id = req.thread_id or uuid.uuid4().hex
    config = {"configurable": {"thread_id": thread_id}}

    # 已有会话 → 读取快照判断状态（挂起中 / 已完成 / 不存在）
    snapshot = graph.get_state(config) if req.thread_id else None

    # 防御：该会话正挂起等待审批 → 拒绝新命令，避免覆盖存档
    if snapshot is not None and snapshot.interrupts:
        raise HTTPException(
            status_code=409,
            detail="该会话正在等待审批，请先调用 /api/approve 完成审批，或开启新会话",
        )

    # 新会话带 system prompt；续聊只追加用户消息（历史里已有 system）
    msgs = [HumanMessage(content=req.query)]
    is_new = snapshot is None or not snapshot.values.get("messages")
    if is_new:
        msgs.insert(0, SystemMessage(content=system_prompt))

    result = graph.invoke({"messages": msgs}, config=config)

    if "__interrupt__" in result:
        intr = result["__interrupt__"][0]
        return ChatResponse(
            status="awaiting_approval",
            thread_id=thread_id,
            approval_payload=intr.value,
        )
    return ChatResponse(reply=result["messages"][-1].content, thread_id=thread_id)
