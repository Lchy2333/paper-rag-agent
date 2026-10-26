"""Token 用量统计（callback 方案，节点零改动）。

UsageCallbackHandler 挂在 graph 的 config["callbacks"] 上，每次 LLM 调用结束
（含子图内，已实测）触发 on_llm_end，累加 usage_metadata，并用
get_stream_writer() 把每轮快照推进 custom 流（chat.py 转成 SSE event: usage）。

每次请求新建一个实例 = 天然按单次 query 隔离；session 累计由
query_usage 表 SUM 提供基线（db.get_session_usage），实例只算本次 query 的增量，
请求结束由调用方落库（db.record_usage）。
"""
import threading
from dataclasses import dataclass
from typing import Any, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langgraph.config import get_stream_writer


@dataclass
class Usage:
    """本次 query 的累计账本。"""

    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def context_ratio(used: int, budget: Optional[int]) -> Optional[float]:
    """上下文占用率 = 最后一轮 input / 输入预算；未知时返回 None（前端隐藏指示器）。"""
    if budget and used > 0:
        return round(used / budget, 4)
    return None


class UsageCallbackHandler(BaseCallbackHandler):
    """按"一次图运行"累计所有 LLM 调用的 token 用量。

    - 节点零侵入：只把实例塞进 config["callbacks"]；
    - 子图 / 并行分支内的 LLM 调用也会被捕获（callback 随 config 传播）；
    - 流式调用同样触发 on_llm_end（usage 落在最终消息上）。
    """

    def __init__(
        self,
        thread_id: str,
        user_id: str,
        session_input: int = 0,
        session_output: int = 0,
        input_budget: Optional[int] = None,
        context_window: Optional[int] = None,
    ):
        self.thread_id = thread_id
        self.user_id = user_id
        self.usage = Usage()
        self.session_input = session_input   # 会话基线（请求开始时从 DB SUM 读出）
        self.session_output = session_output
        # 上下文占用：分母是输入预算（窗口 − 输出预留 − 安全余量，见 core/model_limits.py），
        # 分子只取"最后一轮 input"——每轮 input 都是完整重发，最后一轮即当前真实占用。
        self.input_budget = input_budget
        self.context_window = context_window
        self.last_input_tokens = 0
        self._lock = threading.Lock()        # 防御性：FastAPI 线程池下并发安全

    def on_llm_end(self, response: Any, **kwargs) -> None:
        u = self._extract(response)
        if u is None:
            return  # provider 没返回 usage → 跳过（不崩，不污染统计）
        with self._lock:
            self.usage.input_tokens += u["input_tokens"]
            self.usage.output_tokens += u["output_tokens"]
            self.usage.calls += 1
            self.last_input_tokens = u["input_tokens"]  # 覆盖而非累加：取最新一轮的水位
        # 每轮结束推一次 custom 事件；非流式（graph.invoke）时 writer 为 None 自动跳过
        try:
            writer = get_stream_writer()
        except Exception:
            writer = None
        if writer is not None:
            writer({"usage": self.snapshot()})

    def snapshot(self) -> dict:
        """SSE 事件载荷：本次 query 增量 + 会话累计（含基线）+ 上下文占用。"""
        # 还没发生 LLM 调用 → 占用未知（ratio=None，前端隐藏指示器，不展示假的 0%）
        ratio = context_ratio(self.last_input_tokens, self.input_budget)
        return {
            "thread_id": self.thread_id,
            "calls": self.usage.calls,
            "query_input": self.usage.input_tokens,
            "query_output": self.usage.output_tokens,
            "query_total": self.usage.total_tokens,
            "session_input": self.session_input + self.usage.input_tokens,
            "session_output": self.session_output + self.usage.output_tokens,
            "session_total": self.session_input + self.session_output + self.usage.total_tokens,
            "context_window": self.context_window,
            "context_budget": self.input_budget,
            "context_used": self.last_input_tokens,
            "context_ratio": ratio,
        }

    @staticmethod
    def _extract(response: Any) -> Optional[dict]:
        """取一次调用的用量。首选消息上的 usage_metadata，兜底 OpenAI 的 token_usage。"""
        gens = getattr(response, "generations", None)
        if gens:
            meta = getattr(gens[0][0].message, "usage_metadata", None)
            if meta:
                return {
                    "input_tokens": meta.get("input_tokens", 0),
                    "output_tokens": meta.get("output_tokens", 0),
                }
        llm_output = getattr(response, "llm_output", None) or {}
        tu = llm_output.get("token_usage") or {}
        if tu:
            return {
                "input_tokens": tu.get("prompt_tokens", 0),
                "output_tokens": tu.get("completion_tokens", 0),
            }
        return None
