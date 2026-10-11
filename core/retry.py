import random
import time
from typing import Callable

import openai
import requests


def retry_call(fn: Callable, *, retryable: Callable[[Exception], bool],
               max_attempts=3, base_delay=1.0, max_delay=8.0,
               jitter_ratio=0.0, sleep=time.sleep):
    """带指数退避 + 抖动的高阶调用器。只在 retryable(e) 为真的失败上重试。"""
    attempt = 0
    while True:
        try:
            return fn()
        except BaseException as e:
            if not retryable(e) or attempt >= max_attempts - 1:
                raise
            delay = min(base_delay * 2 ** attempt, max_delay)
            delay *= 1 + random.uniform(-jitter_ratio, jitter_ratio)
            sleep(delay)
            attempt += 1


def is_transient(e: Exception) -> bool:
    """判定异常是否为「暂时失败」：超时/连接/限流/5xx → 可重试。"""
    if isinstance(e, requests.Timeout) or isinstance(e, requests.ConnectionError):
        return True
    if isinstance(e, openai.RateLimitError):
        return True
    if isinstance(e, (openai.APITimeoutError, openai.APIConnectionError)):
        return True
    if isinstance(e, openai.APIStatusError):
        return 500 <= e.status_code < 600
    return False
