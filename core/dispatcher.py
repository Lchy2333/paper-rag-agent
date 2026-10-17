import json
import time
from typing import TYPE_CHECKING

from .retry import is_transient, retry_call
from .tool import ToolResult

if TYPE_CHECKING:
    from .registry import ToolRegistry


class Dispatcher:
    """调度器：查工具 → (可重试地) 执行工具 → 包成 ToolResult。"""

    def __init__(self, registry: "ToolRegistry", sleep=time.sleep):
        self.registry = registry
        self.sleep = sleep

    def dispatch(self, tool_name: str, args_json: str, user_id: str | None = None) -> ToolResult:
        try:
            args = json.loads(args_json) if args_json else {}
        except json.JSONDecodeError as e:
            return ToolResult(success=False, error=f"invalid args json: {e}")

        tool = self.registry.get(tool_name)
        if not tool:
            return ToolResult(success=False, error=f"found no tool: {tool_name}")

        start = time.time()
        try:
            tool_result = retry_call(
                lambda: tool.execute(args, user_id=user_id),
                retryable=is_transient,
                max_attempts=3,
                jitter_ratio=0.1,
                sleep=self.sleep,
            )
        except Exception as e:
            return ToolResult(success=False, error=f"calling tool failed: {e}")

        tool_result.latency_ms = (time.time() - start) * 1000
        return tool_result
