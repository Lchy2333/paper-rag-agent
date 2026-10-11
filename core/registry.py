from typing import Dict, List, Optional

from .tool import ToolBase


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, ToolBase] = {}

    def register(self, tool: ToolBase):
        self._tools[tool.name()] = tool

    def get(self, tool_name: str) -> Optional[ToolBase]:
        return self._tools.get(tool_name)

    def list_schemas(self) -> List[Dict[str, object]]:
        return [tool.build_schema() for tool in self._tools.values()]
