"""LLM 就绪探针：验证一个 LLM 配置（base_url / api_key / model）是否可用。

设计动机（见 architecture.md §13）：
- LLM 是外部上游依赖，健康检查语义为**就绪**（readiness），不是存活（liveness）——
  重启 agent 救不了 LLM，就绪失败只应标记"不可服务对话"，不应触发编排重启；
- 就绪验证 = 一次 max_tokens=1 的真实调用（端到端证明 key 有效 + 模型可调 + 网络通），
  比 TCP 连通（无法证明 key 有效）准确；
- 本模块同时服务两个场景：
  ① /health 的 LLM 就绪探针（用全局 LLM 配置 + 节流缓存）；
  ② POST /api/llm/check（BYOK：用户验证自己上传的模型配置，body 传入三要素）。
"""
import logging
import threading
import time
from typing import Optional

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# 主动探针节流间隔（秒）：即使 /health 被 5s 轮询，也最多每 60s 真调一次 LLM
PROBE_THROTTLE_SECONDS = 60
# 单次探针超时（秒）：不能拖慢 /health 响应
PROBE_TIMEOUT_SECONDS = 10
# 被动信号：连续失败 N 次才判定 error（抖动抑制）
FAILURE_THRESHOLD = 2
# 被动信号：距上次成功超过此时间且无新调用 → 回到 unknown（不主动探）
STALE_SECONDS = 600


def check_llm(base_url: str, api_key: str, model: str, timeout: int = PROBE_TIMEOUT_SECONDS) -> dict:
    """就绪验证：用给定配置发一次 max_tokens=1 调用，返回 {ok, latency_ms, error?}。

    可复用场景：/health 主动探针、/api/llm/check（BYOK 用户验证自有配置）。
    不抛异常，任何失败都归一化为 ok=False + error。
    """
    if not (base_url and api_key and model):
        return {"ok": False, "error": "LLM 配置不完整（base_url / api_key / model 均需提供）"}

    llm = ChatOpenAI(
        base_url=base_url,
        api_key=api_key,
        model=model,
        max_tokens=1,
        timeout=timeout,
        max_retries=0,  # 探针不做重试，一次定生死，避免放大
    )
    start = time.time()
    try:
        llm.invoke([HumanMessage(content="ping")])
        latency_ms = round((time.time() - start) * 1000)
        logger.info("llm_probe_ok latency_ms=%s", latency_ms)
        return {"ok": True, "latency_ms": latency_ms}
    except Exception as e:  # noqa: BLE001 —— 探针必须吞掉一切异常
        latency_ms = round((time.time() - start) * 1000)
        logger.warning("llm_probe_failed latency_ms=%s error=%s", latency_ms, str(e)[:200])
        return {"ok": False, "latency_ms": latency_ms, "error": str(e)[:200]}


class LLMHealthTracker:
    """被动信号记录器：从真实 chat 调用中更新 LLM 健康状态（零额外请求）。

    call_llm 节点在每次真实调用后 record_success / record_failure，
    /health 据此 + 节流主动探针输出 llm 就绪状态。
    进程内单例（app.state.llm_tracker），多线程安全。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._last_success_at: Optional[float] = None
        self._consecutive_failures = 0
        self._last_error: Optional[str] = None
        self._last_probe_at: Optional[float] = None
        self._last_probe_result: Optional[dict] = None

    # ---------- 被动信号写入（call_llm 调用） ----------

    def record_success(self) -> None:
        with self._lock:
            self._last_success_at = time.time()
            self._consecutive_failures = 0
            self._last_error = None

    def record_failure(self, error: str) -> None:
        with self._lock:
            self._consecutive_failures += 1
            self._last_error = error[:200]

    # ---------- 就绪状态判定 ----------

    def status(self, global_llm: tuple, now: Optional[float] = None) -> dict:
        """输出 LLM 就绪状态：{status, error?}。status ∈ ok / error / unknown。

        - 被动信号优先：最近有成功 → ok；连续失败超阈值 → error；
        - 长时间无真实调用（冷启动 / 空闲）→ unknown，此时若上次主动探针过期，
          触发一次节流主动探针（max_tokens=1）作为兜底信号。
        """
        now = now or time.time()
        with self._lock:
            last_success = self._last_success_at
            failures = self._consecutive_failures
            last_error = self._last_error
            last_probe_at = self._last_probe_at
            last_probe = self._last_probe_result

        # 1) 真实调用有失败记录 → 抖动抑制后判 error
        if failures >= FAILURE_THRESHOLD:
            return {"status": "error", "error": last_error or "LLM 连续调用失败"}
        # 2) 最近成功过且未过期 → ok
        if last_success is not None and (now - last_success) < STALE_SECONDS:
            return {"status": "ok"}
        # 3) 有历史成功但已闲置 → unknown（不主动探，避免空闲烧钱）
        if last_success is not None:
            return {"status": "unknown"}

        # 4) 从未成功过（冷启动）→ 用节流主动探针兜底
        if last_probe_at is None or (now - last_probe_at) >= PROBE_THROTTLE_SECONDS:
            base_url, api_key, model = global_llm
            result = check_llm(base_url, api_key, model)
            with self._lock:
                self._last_probe_at = now
                self._last_probe_result = result
            if result["ok"]:
                return {"status": "ok"}
            return {"status": "error", "error": result.get("error")}
        # 5) 探针结果在缓存窗口内
        if last_probe is not None:
            if last_probe.get("ok"):
                return {"status": "ok"}
            return {"status": "error", "error": last_probe.get("error")}
        return {"status": "unknown"}
