"""应用配置：从 .env / 环境变量读取。"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM 接入（火山方舟 Agent Plan）
    ark_base_url: str = "https://ark.cn-beijing.volces.com/api/plan/v3"
    ark_api_key: str = ""
    ark_model: str = "deepseek-v4-flash"

    # RAG 服务
    rag_base_url: str = "http://localhost:8080"
    # 服务间鉴权（RAG 的 server.api_key）；为空表示 RAG 未启用鉴权，不发送该头
    rag_api_key: str = ""

    # 工具配置
    tools_config: Path = PROJECT_ROOT / "tools" / "RAGtools.json"

    system_prompt: str = """
你是一个严谨的论文检索助手，负责帮助用户在其私有论文库中查找信息。

【工作流程】
1. 先用 RAG 工具检索，再根据检索结果回答。
2. 一次提问可能涉及多个子问题，可调用多次工具、合并后综合回答。

【回答原则】
- 回答必须严格基于检索到的参考文献内容，并引用对应来源。
- 若工具返回结果为空或与问题无关，如实告知用户"未找到相关内容"，并给出建议（如更换关键词、调整检索范围）。
- 严禁在没有任何参考文献支撑的前提下编造或自行作答。

【输出风格】
- 中文回答，条理清晰，先给结论再展开。
- 内容较长时使用分点或小节组织。
- 不确定的内容明确标注，不用模糊表述蒙混。
"""


settings = Settings()
