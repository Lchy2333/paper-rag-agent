"""LLM 就绪验证接口：POST /api/llm/check。

用途（BYOK 复用点，见 architecture.md §13 / §11）：
- 用户在"配置自己的 LLM"时，用本接口验证 base_url / api_key / model 是否可用；
- 实现复用 core.llm_probe.check_llm（与 /health 主动探针同源，一份逻辑）。
- 注意：body 里的 api_key 是用户自己的，仅本请求内存态使用，不落库。
"""
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.auth import CurrentUser
from core.llm_probe import check_llm

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/llm", tags=["llm"])


class LLMCheckRequest(BaseModel):
    base_url: str = Field(..., description="LLM 服务地址（OpenAI 兼容）")
    api_key: str = Field(..., description="用户自己的 API Key")
    model: str = Field(..., description="模型名")


@router.post("/check")
def llm_check(req: LLMCheckRequest, user=CurrentUser):
    """验证一组 LLM 配置是否可用（发一次 max_tokens=1 探针调用）。需要登录。"""
    result = check_llm(req.base_url, req.api_key, req.model)
    if not result.get("ok"):
        logger.info("llm_check_failed model=%s error=%s", req.model, result.get("error"))
        raise HTTPException(status_code=400, detail=result.get("error", "LLM 配置不可用"))
    return {"ok": True, "latency_ms": result.get("latency_ms")}
