"""认证：bcrypt 密码哈希 + JWT 签发/验签 + get_current_user 依赖。

信任链：客户端登录 → 服务器签发 JWT（载荷含 user_id）→ 客户端带
`Authorization: Bearer <token>` → get_current_user 验签后查 users 表确认账号有效。

设计要点：
- JWT 无状态：服务器不存 token，验签即可得 user_id（payload.sub）。
- JWT_SECRET 是签名密钥：只有它才能伪造/篡改通过验签的 token，务必保密（.env）。
- 所有 /chat、/approve 等需登录的接口，用 `Depends(get_current_user)` 收口。
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
from uuid import UUID

import bcrypt
import jwt
from fastapi import Depends, Header, HTTPException, Request

from app.config import settings
from app import db

logger = logging.getLogger(__name__)

ALGORITHM = "HS256"


def hash_password(password: str) -> str:
    """bcrypt 哈希（加盐），数据库只存哈希。"""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """校验明文密码与哈希是否匹配；哈希损坏时视为不匹配（不抛异常）。"""
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


def create_access_token(user_id: UUID, username: str) -> str:
    """签发 JWT：payload 里写 user_id（sub）与过期时间，用 JWT_SECRET 签名。"""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "username": username,
        "iat": now,
        "exp": now + timedelta(minutes=settings.jwt_expire_minutes),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=ALGORITHM)


def decode_token(token: str) -> Dict[str, Any]:
    """验签并解析 JWT；签名错误/过期由 jwt 库抛异常，由调用方转 HTTP 401。"""
    return jwt.decode(token, settings.jwt_secret, algorithms=[ALGORITHM])


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(status_code=401, detail=detail)


def get_current_user(
    request: Request,
    authorization: Optional[str] = Header(None, description="Bearer <JWT>"),
) -> Dict[str, Any]:
    """FastAPI 依赖：验 JWT → 查 users 表 → 返回当前用户（含 id/username/disabled）。

    任意一步失败均返回 401，不向客户端暴露内部细节。
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise _unauthorized("未提供认证凭证，请先登录")
    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise _unauthorized("认证凭证为空")

    try:
        payload = decode_token(token)
    except jwt.ExpiredSignatureError:
        raise _unauthorized("登录已过期，请重新登录")
    except jwt.InvalidTokenError:
        raise _unauthorized("无效的登录凭证")

    user_id = payload.get("sub")
    if not user_id:
        raise _unauthorized("登录凭证缺少用户信息")

    user = db.get_user_by_id(request.app.state.pool, user_id)
    if user is None:
        logger.warning("auth_user_not_found user_id=%s", user_id)
        raise _unauthorized("用户不存在或已注销")
    if user["disabled"]:
        raise _unauthorized("账号已被禁用")

    return user


CurrentUser = Depends(get_current_user)
