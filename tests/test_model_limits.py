"""模型上下文限额单测：目录解析 / 覆盖优先级 / 输入预算边界 / usage 快照比例。"""
from types import SimpleNamespace

import pytest

from core import model_limits
from core.model_limits import (
    calculate_input_budget,
    get_model_limits,
    resolve_llm_limits,
)
from core.usage import UsageCallbackHandler

CATALOG = {
    "volcengine-coding-plan": {
        "models": {
            "deepseek-v4-flash": {
                "limit": {"context": 1_000_000, "output": 384_000},
            },
            "has-input-limit": {
                "limit": {"context": 1_000_000, "input": 200_000, "output": 8_000},
            },
            "tiny": {
                # 输出上限比预留还小：预算应夹到 0，不能出现负数
                "limit": {"context": 4_000, "output": 8_000},
            },
            "invalid": {"limit": {"context": 0, "output": "384000"}},
        }
    }
}


def test_get_model_limits_exact_and_case_insensitive():
    """目录结构 catalog[provider].models[model].limit；裸名精确命中，大小写兜底。"""
    exact = get_model_limits("volcengine-coding-plan", "deepseek-v4-flash", CATALOG)
    assert exact is not None
    assert exact.context_window == 1_000_000
    assert exact.output_limit == 384_000
    assert exact.input_limit is None  # 该模型没有独立 input 上限

    upper = get_model_limits("volcengine-coding-plan", "DEEPSEEK-V4-FLASH", CATALOG)
    assert upper is not None and upper.context_window == 1_000_000

    assert get_model_limits("volcengine-coding-plan", "not-exist", CATALOG) is None
    assert get_model_limits("no-such-provider", "deepseek-v4-flash", CATALOG) is None


def test_get_model_limits_rejects_invalid_numbers():
    """目录里的 0 / 字符串 / 缺失都按"未知"处理，不能当成真实上限。"""
    limits = get_model_limits("volcengine-coding-plan", "invalid", CATALOG)
    assert limits is not None
    assert limits.context_window is None
    assert limits.output_limit is None


def test_calculate_input_budget_main_case():
    """写死窗口 1,048,576 → 1,048,576 - 8,192 - 4,096 = 1,036,288。"""
    assert calculate_input_budget(1_048_576) == 1_036_288


def test_calculate_input_budget_respects_smaller_input_limit():
    """有独立 input 上限时取更严的那个：min(200000, 991808) - 4096。"""
    assert calculate_input_budget(1_000_000, input_limit=200_000) == 195_904


def test_calculate_input_budget_does_not_over_reserve():
    """输出上限 8000 < 预留 8192 → 按 8000 预留；窗口装不下时夹到 0。"""
    assert calculate_input_budget(32_768, output_limit=8_000) == 32_768 - 8_000 - 4_096
    assert calculate_input_budget(4_000, output_limit=8_000) == 0


def test_calculate_input_budget_unknown_window():
    """窗口未知 → None（前端隐藏指示器，不猜）。"""
    assert calculate_input_budget(None) is None
    assert calculate_input_budget(0) is None


def test_resolve_uses_env_override_without_touching_catalog(monkeypatch):
    """.env 写了窗口 → 直接采用，且不发起 models.dev 请求（启动期无网络也能起）。"""

    def boom(*args, **kwargs):
        raise AssertionError("LLM_CONTEXT_WINDOW 非 0 时不应访问 models.dev")

    monkeypatch.setattr(model_limits, "load_models_catalog", boom)
    settings = SimpleNamespace(
        llm_context_window=1_048_576,
        llm_provider_id="volcengine-coding-plan",
        llm_model="deepseek-v4-flash",
    )
    limits = resolve_llm_limits(settings)
    assert limits.context_window == 1_048_576
    assert limits.input_budget == 1_036_288


def test_resolve_falls_back_to_catalog_when_window_unset():
    """LLM_CONTEXT_WINDOW=0 → 查目录；窗口/输出上限来自目录。"""
    settings = SimpleNamespace(
        llm_context_window=0,
        llm_provider_id="volcengine-coding-plan",
        llm_model="deepseek-v4-flash",
    )
    limits = resolve_llm_limits(settings, CATALOG)
    assert limits.context_window == 1_000_000
    assert limits.output_limit == 384_000
    assert limits.input_budget == 1_000_000 - 8_192 - 4_096


def test_resolve_returns_none_budget_when_catalog_missing():
    """目录查不到且没有覆盖值 → 预算 None（前端隐藏，而不是显示满格）。"""
    settings = SimpleNamespace(
        llm_context_window=0,
        llm_provider_id="volcengine-coding-plan",
        llm_model="not-exist",
    )
    limits = resolve_llm_limits(settings, CATALOG)
    assert limits.context_window is None
    assert limits.input_budget is None


class _Gen:
    def __init__(self, input_tokens, output_tokens):
        self.message = SimpleNamespace(
            usage_metadata={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }
        )


class _Result:
    def __init__(self, input_tokens, output_tokens):
        self.generations = [[_Gen(input_tokens, output_tokens)]]


def test_snapshot_context_ratio_uses_last_input():
    """分子取最后一轮 input（不累加，避免把滚入的历史重复计数）。"""
    handler = UsageCallbackHandler("t1", "u1", input_budget=1_000, context_window=1_048_576)
    assert handler.snapshot()["context_ratio"] is None  # 还没调用 → 未知

    handler.on_llm_end(_Result(100, 10))
    assert handler.snapshot()["context_ratio"] == pytest.approx(0.1)
    assert handler.snapshot()["context_used"] == 100

    handler.on_llm_end(_Result(200, 20))  # 第二轮 input 已含上一轮历史
    snap = handler.snapshot()
    assert snap["query_input"] == 300     # 累计账本仍在累加（计费用）
    assert snap["context_used"] == 200    # 水位只取最新一轮
    assert snap["context_ratio"] == pytest.approx(0.2)
    assert snap["context_budget"] == 1_000
    assert snap["context_window"] == 1_048_576


def test_snapshot_context_ratio_hidden_without_budget():
    """没有预算（模型未知）→ ratio 恒为 None，前端隐藏。"""
    handler = UsageCallbackHandler("t1", "u1")
    handler.on_llm_end(_Result(100, 10))
    snap = handler.snapshot()
    assert snap["context_used"] == 100
    assert snap["context_ratio"] is None
