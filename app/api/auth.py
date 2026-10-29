"""认证路由：注册 / 登录 / 服务间身份校验。

- 注册：username 唯一性校验 + bcrypt 哈希入库。
- 登录：校验密码 → 签发 JWT（无状态，见 app/auth.py）。
- introspect：Bearer JWT → 可信用户信息，供 MCP 等后端服务换取 user_id。
前两个接口是"发凭证的入口"，本身不带认证头；introspect 需要 Bearer。
"""
import logging
from typing import Any, Dict

import bcrypt
import jwt
from fastapi import APIRouter, HTTPException, Request

from app import db
from app.auth import CurrentUser, create_access_token, decode_token, hash_password, verify_password
from app.config import settings
from app.schemas import LoginRequest, RegisterRequest, TokenResponse, UserOut

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserOut, status_code=201)
def register(req: RegisterRequest, request: Request) -> UserOut:
    if not settings.registration_enabled:
        # 开关关闭：公开注册暂停，账号只能由管理员直接插库创建
        raise HTTPException(status_code=403, detail="注册已关闭，请联系管理员开通账号")

    if db.get_user_by_username(request.app.state.pool, req.username):
        raise HTTPException(status_code=409, detail="用户名已存在")

    try:
        password_hash = hash_password(req.password)
    except ValueError as e:
        # bcrypt 对过长的密码抛错（如超 72 字节），转为 400
        raise HTTPException(status_code=400, detail="密码格式不支持") from e

    user = db.create_user(request.app.state.pool, req.username, password_hash)
    logger.info("user_registered username=%s user_id=%s", req.username, user["id"])
    return UserOut(user_id=str(user["id"]), username=user["username"])


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest, request: Request) -> TokenResponse:
    user = db.get_user_by_username(request.app.state.pool, req.username)
    if user is None or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    if user["disabled"]:
        raise HTTPException(status_code=403, detail="账号已被禁用")

    token = create_access_token(user["id"], user["username"])
    return TokenResponse(access_token=token, user_id=str(user["id"]))


@router.post("/introspect")
def introspect(request: Request, user: Dict[str, Any] = CurrentUser) -> Dict[str, Any]:
    """服务间身份校验：Bearer JWT → 可信用户信息（不含 token / 密钥）。

    复用 get_current_user，因此天然继承「验签 + 查库 + 禁用校验」。
    返回 exp 供调用方计算缓存 TTL：缓存不得超过 token 剩余有效期。
    该接口只返回身份，不返回业务数据；部署时应限制为内网/服务网段可访问。
    """
    exp = None
    authorization = request.headers.get("authorization") or ""
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() == "bearer" and token.strip():
        try:
            exp = decode_token(token.strip()).get("exp")
        except jwt.InvalidTokenError:
            # get_current_user 已验签通过，此处失败属极端竞态；不影响身份返回
            exp = None

    return {
        "user_id": str(user["id"]),
        "username": user["username"],
        "disabled": bool(user["disabled"]),
        "exp": exp,
    }
