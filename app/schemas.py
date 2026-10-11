"""请求 / 响应模型。"""
from typing import Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., description="用户问题")
    thread_id: Optional[str] = Field(None, description="会话 ID，暂不接 checkpointer，预留")


class ChatResponse(BaseModel):
    reply: str
