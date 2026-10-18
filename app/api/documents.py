"""文档管理转发路由：前端经 agent 中转访问 RAG 文档接口（BFF 模式）。

信任链：前端带 JWT → 本路由 CurrentUser 验签 → 取可信 user_id →
RagProxy 注入 X-User-ID + X-API-Key → 转发 RAG。
前端不再持有 RAG 的 API Key、也无法伪造 X-User-ID。
"""
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.auth import CurrentUser
from app import db
from core.rag_proxy import RagProxyError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])


class RechunkRequest(BaseModel):
    chunk_size: Optional[int] = Field(None, ge=100, le=8000, description="新分块大小（字符）")
    chunk_overlap: Optional[int] = Field(None, ge=0, le=2000, description="新重叠字符数")


def _get_proxy(request: Request) -> "RagProxy":
    """从 app.state 取 RagProxy（build_agent 时构造，进程内单例）。"""
    return request.app.state.rag_proxy


def _handle_rag_error(e: RagProxyError) -> HTTPException:
    """把 RagProxyError 转成与 RAG 一致语义的 HTTP 响应。"""
    logger.warning("rag_proxy_error status=%s body=%s", e.status_code, e.body)
    if e.status_code in (400, 404, 409, 413):
        detail = e.body if isinstance(e.body, str) else None
        if isinstance(e.body, dict):
            detail = e.body.get("error") or e.body.get("message")
        raise HTTPException(status_code=e.status_code, detail=detail or "RAG 请求失败")
    if e.status_code == 403:
        raise HTTPException(status_code=403, detail="无权访问该文档")
    if e.status_code in (502, 503, 504):
        raise HTTPException(status_code=502, detail=str(e) or "RAG 服务暂不可用")
    # 其余 5xx 一律 502，避免透出内部细节
    raise HTTPException(status_code=502, detail="RAG 服务内部错误")


@router.get("")
def list_documents(request: Request, user=CurrentUser):
    """列出当前用户可见的文档（admin 全库）。"""
    try:
        data = _get_proxy(request).list_documents(str(user["id"]))
    except RagProxyError as e:
        _handle_rag_error(e)
    return JSONResponse(content=data)


@router.post("")
async def upload_documents(request: Request, files: list[UploadFile], user=CurrentUser):
    """上传 PDF（一个或多个，multipart 字段名 file）。可选分块参数经 form 字段传入。

    RAG 为异步摄入：本接口只做接收段（秒回），处理进度由前端轮询文档详情。
    """
    # 构造 requests 的 files 结构：(field_name, (filename, fileobj, content_type))
    rag_files = []
    for f in files:
        content = await f.read()
        rag_files.append(("file", (f.filename or "upload.pdf", content, f.content_type or "application/pdf")))

    # 透传可选分块参数（form 字段）
    form = await request.form()
    data = {}
    for key in ("chunk_size", "chunk_overlap"):
        if form.get(key) is not None:
            data[key] = str(form[key])

    try:
        result = _get_proxy(request).upload_documents(str(user["id"]), rag_files, data)
    except RagProxyError as e:
        _handle_rag_error(e)
    return JSONResponse(content=result)


@router.get("/{document_id}")
def get_document(document_id: str, request: Request, user=CurrentUser):
    """查询文档详情（上传后轮询进度用）。"""
    try:
        data = _get_proxy(request).get_document(str(user["id"]), document_id)
    except RagProxyError as e:
        _handle_rag_error(e)
    return JSONResponse(content=data)


@router.post("/{document_id}/rechunk")
def rechunk_document(
    document_id: str, body: RechunkRequest, request: Request, user=CurrentUser
):
    """按新分块参数重新分块入库（慢，期间旧 chunk 仍可用）。"""
    opts = body.model_dump(exclude_none=True)
    try:
        data = _get_proxy(request).rechunk_document(str(user["id"]), document_id, opts)
    except RagProxyError as e:
        _handle_rag_error(e)
    return JSONResponse(content=data)


@router.delete("/{document_id}")
def delete_document(document_id: str, request: Request, user=CurrentUser):
    """删除文档及其向量。

    场景说明（见 architecture.md §12）：
    - 前端文档库页主动删除 = 用户本人明确操作，无需二次审批，直接转发 + 审计留痕；
    - Chat 场景 LLM 调 delete_document 工具仍走 RAGtools.json 的 requires_approval
      + guard_node HITL（本端点不参与）。
    """
    try:
        data = _get_proxy(request).delete_document(str(user["id"]), document_id)
    except RagProxyError as e:
        _handle_rag_error(e)

    # 审计留痕：谁在何时删除了哪个文档（thread_id 不适用，传空字符串）
    db.add_audit_log(
        request.app.state.pool,
        str(user["id"]),
        "",
        "document_delete",
        {"document_id": document_id},
    )
    logger.info("document_deleted user_id=%s document_id=%s", user["id"], document_id)

    if data is None:
        return {"status": "deleted", "document_id": document_id}
    return JSONResponse(content=data)
