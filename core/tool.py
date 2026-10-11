from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ToolResult:
    success: bool
    data: Optional[Any] = None
    error: Optional[Any] = None
    latency_ms: float = 0.0


class ToolBase:
    """工具基类，初始化时为每个工具创建实例"""
    def name(self) -> str:
        raise NotImplementedError

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        raise NotImplementedError

    def build_schema(self) -> Dict[str, Any]:
        raise NotImplementedError
