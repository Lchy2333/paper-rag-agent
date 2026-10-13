"""异常分类：把未捕获异常翻译成对外 HTTP 响应（状态码 + error_code + 用户提示）。

设计原则：
- 对外脱敏：用户只需知道"发生了什么类别的事、该不该重试"；
- 对内详尽：完整异常类型/消息进日志（见 main.py 的 exception_handler）。
- 判定用 openai 基类：langchain-openai 的自定义异常保留了 openai 继承链
 （见 tests/test_retry.py 的继承验证），无论 LLM 库怎么包装都能命中。
"""
import openai


def classify_exception(exc: Exception) -> tuple:
    """异常 → (HTTP 状态码, error_code, 用户可见提示)。"""
    # LLM 上游不可达 / 超时（通常是 retry_call 重试穷尽后冒出来的）
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
        return 502, "llm_unreachable", "LLM 服务暂时不可达，请稍后重试"
    # LLM 限流（重试穷尽）
    if isinstance(exc, openai.RateLimitError):
        return 429, "llm_rate_limited", "请求过于频繁，请稍后再试"
    # LLM 配置错误（key/模型名写错）——is_transient 快速失败的那类，不重试
    if isinstance(exc, (openai.AuthenticationError, openai.NotFoundError)):
        return 500, "llm_config_error", "服务配置异常，请联系管理员"
    # 兜底：未知异常
    return 500, "internal_error", "服务内部错误，请稍后重试"
