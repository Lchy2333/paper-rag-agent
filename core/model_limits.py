"""模型上下文窗口能力：models.dev 目录读取 + 本地覆盖 + 输入预算推导。

数据源：https://models.dev/api.json（顶层即 provider 字典）
    catalog[provider_id]["models"][model_id]["limit"] = {"context": int, "input"?: int, "output"?: int}

为什么要"输入预算"而不是直接用上下文窗口当分母：
- 上下文窗口是 input + output 共享池，必须给本轮生成预留位置；
- 但进度条量的是 input 的实际占用（output 会滚入下一轮 input，用 total 会重复计数）；
- 所以分母 = min(input_limit, window - output_reserve) - safety_margin。

解析优先级（高 → 低）：
1. `.env` 的 LLM_CONTEXT_WINDOW（当前火山 deepseek-v4-flash 先写死 1048576）；
2. models.dev 目录（24h 缓存，拉取失败降级旧缓存）；
3. None → 前端隐藏上下文指示器（宁可不显示，也不显示错的）。
"""
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

MODELS_DEV_URL = "https://models.dev/api.json"
CATALOG_TTL_SECONDS = 24 * 3600  # 目录 24h 刷新一次即可（模型上下限极少变动）
FETCH_TIMEOUT_SECONDS = 5.0      # 启动期拉取，不能拖慢服务启动
OUTPUT_RESERVE_TOKENS = 8192     # 给本轮生成预留的输出空间
SAFETY_MARGIN_TOKENS = 4096      # 再留一点余量，抵消 tokenizer / 工具 schema 估算偏差

# 进程内缓存：目录约数 MB，启动解析一次后复用
_catalog: Optional[dict] = None
_catalog_fetched_at: float = 0.0


@dataclass
class ModelLimits:
    """一个模型的能力上限 + 由它推导出的输入预算。"""

    context_window: Optional[int] = None  # 上下文窗口（input + output 共享）
    input_limit: Optional[int] = None     # 服务商声明的独立输入上限（可能没有）
    output_limit: Optional[int] = None    # 最大输出
    input_budget: Optional[int] = None    # 进度条分母（见 calculate_input_budget）


def load_models_catalog(force_refresh: bool = False) -> dict:
    """拉取 models.dev 目录（带 TTL 缓存）。失败时返回旧缓存，没有旧缓存则返回空字典。"""
    global _catalog, _catalog_fetched_at
    now = time.time()
    if (
        not force_refresh
        and _catalog is not None
        and now - _catalog_fetched_at < CATALOG_TTL_SECONDS
    ):
        return _catalog

    try:
        resp = requests.get(MODELS_DEV_URL, timeout=FETCH_TIMEOUT_SECONDS)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            raise ValueError(f"目录格式异常，顶层应为 dict，实际为 {type(data).__name__}")
        _catalog = data
        _catalog_fetched_at = now
        return data
    except Exception as e:  # 网络 / 解析失败都不应影响服务启动
        logger.warning("models.dev 目录拉取失败（沿用旧缓存或跳过）：%s", e)
        return _catalog if _catalog is not None else {}


def get_model_limits(
    provider_id: str,
    model_id: str,
    catalog: Optional[dict] = None,
) -> Optional[ModelLimits]:
    """按 provider + model 查目录。先精确命中，再大小写不敏感兜底。"""
    if not provider_id or not model_id:
        return None
    data = catalog if catalog is not None else load_models_catalog()
    models = (data.get(provider_id) or {}).get("models") or {}

    entry = models.get(model_id)
    if entry is None:
        lowered = model_id.lower()
        entry = next(
            (m for mid, m in models.items() if mid.lower() == lowered),
            None,
        )
    if not isinstance(entry, dict):
        return None

    limit = entry.get("limit") or {}
    return ModelLimits(
        context_window=_positive_int(limit.get("context")),
        input_limit=_positive_int(limit.get("input")),
        output_limit=_positive_int(limit.get("output")),
    )


def resolve_llm_limits(settings: Any, catalog: Optional[dict] = None) -> ModelLimits:
    """按优先级解析当前 LLM 的限额，并算出输入预算（启动期调用一次）。"""
    window = _positive_int(getattr(settings, "llm_context_window", 0))
    entry: Optional[ModelLimits] = None
    if window is None:
        # 仅当 .env 没写死时才查目录（避免无谓的网络请求）
        entry = get_model_limits(
            getattr(settings, "llm_provider_id", "") or "",
            getattr(settings, "llm_model", "") or "",
            catalog,
        )
        window = entry.context_window if entry else None

    return ModelLimits(
        context_window=window,
        input_limit=entry.input_limit if entry else None,
        output_limit=entry.output_limit if entry else None,
        input_budget=calculate_input_budget(
            window,
            entry.input_limit if entry else None,
            entry.output_limit if entry else None,
        ),
    )


def calculate_input_budget(
    context_window: Optional[int],
    input_limit: Optional[int] = None,
    output_limit: Optional[int] = None,
    *,
    output_reserve: int = OUTPUT_RESERVE_TOKENS,
    safety_margin: int = SAFETY_MARGIN_TOKENS,
) -> Optional[int]:
    """输入预算 = min(输入上限, 窗口 - 输出预留) - 安全余量。

    - 窗口未知 → None（前端隐藏指示器）；
    - 输出预留不超过服务商声明的 max_output（小模型别预留过头）；
    - 结果夹到 0，绝不返回负数。
    """
    if not context_window or context_window <= 0:
        return None
    reserve = min(output_reserve, output_limit) if output_limit else output_reserve
    context_based_cap = context_window - reserve
    if context_based_cap <= 0:
        return 0
    known_input_cap = min(input_limit, context_window) if input_limit else context_window
    return max(0, min(known_input_cap, context_based_cap) - safety_margin)


def _positive_int(value: Any) -> Optional[int]:
    """只接受正整数（目录里偶尔有 0 / 缺失 / 字符串）。"""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value > 0 else None
