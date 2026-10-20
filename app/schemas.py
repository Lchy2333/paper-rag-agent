"""请求 / 响应模型。"""
from datetime import datetime
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


class CancelRequest(BaseModel):
    thread_id: str = Field(..., description="要中断生成会话的 thread_id")


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64, description="用户名")
    password: str = Field(..., min_length=8, max_length=64, description="密码")


class LoginRequest(BaseModel):
    username: str = Field(..., description="用户名")
    password: str = Field(..., description="密码")


class TokenResponse(BaseModel):
    access_token: str = Field(..., description="JWT 访问凭证")
    token_type: str = "bearer"
    user_id: str = Field(..., description="用户 UUID")


class UserOut(BaseModel):
    user_id: str = Field(..., description="用户 UUID")
    username: str = Field(..., description="用户名")


class ThreadOut(BaseModel):
    thread_id: str = Field(..., description="会话 ID")
    title: str = Field(..., description="会话标题")
    pinned: bool = Field(..., description="是否置顶")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="最后活跃时间")


class UpdateThreadRequest(BaseModel):
    title: Optional[str] = Field(None, max_length=100, description="新标题")
    pinned: Optional[bool] = Field(None, description="是否置顶（true/false）")


class DeleteThreadResponse(BaseModel):
    status: str = "deleted"
    thread_id: str
