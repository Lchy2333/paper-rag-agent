"""请求 / 响应模型。"""
from typing import Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., description="用户问题")
    thread_id: Optional[str] = Field(None, description="会话 ID；不传则新建会话")


class ChatResponse(BaseModel):
    status: str = "ok"                      # ok / awaiting_approval
    reply: Optional[str] = None
    thread_id: Optional[str] = None
    approval_payload: Optional[dict] = None


class ApproveRequest(BaseModel):
    thread_id: str = Field(..., description="待审批会话的 thread_id")
    approved: bool = Field(..., description="是否允许该操作")
