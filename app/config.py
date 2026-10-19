"""应用配置：从 .env / 环境变量读取。

环境约定（12-factor）：
- `APP_ENV` 显式标识运行环境（dev / prod），生产必须由部署平台注入，
  本地开发在 `.env` 里写 `APP_ENV=dev`。
- 密钥一律**零默认值**：不内置任何"兜底 key"。LLM 是付费资源，
  谁跑谁配置，避免他人 clone 后无意调用项目 owner 的付费 key。
- 启动校验见 `Settings.ensure_runtime()`（在 app/main.py 的 lifespan 中调用，
  import 时不校验，避免测试 / CI 无 .env 时崩溃）。
"""
import logging
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)

# 开发数据库默认值（仅本地开发兜底；生产必须显式配置）
_DEFAULT_DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/paper_rag_agent"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── 环境标识：dev / prod（生产由部署平台注入 APP_ENV=prod）──
    app_env: str = "dev"

    # LLM 接入（OpenAI 兼容协议，可指向任意兼容服务：火山方舟 / DeepSeek / OpenAI / Ollama…）
    # ⚠️ LLM_API_KEY 为付费资源，必须显式配置（不提供任何兜底 key）
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""

    # RAG 服务
    rag_base_url: str = "http://localhost:8080"
    # 服务间鉴权（RAG 的 server.api_key）；为空表示 RAG 未启用鉴权，不发送该头
    rag_api_key: str = ""

    # Agent 数据库（Postgres 独立库 paper_rag_agent；会话持久化 + 多用户）
    database_url: str = _DEFAULT_DATABASE_URL

    # JWT 认证（阶段 B1：注册/登录）
    jwt_secret: str = ""  # 空密钥 = 任何人可伪造 token，必配
    jwt_expire_minutes: int = 10080

    # 新用户注册开关：false = 暂时关闭公开注册（登录不受影响，已有用户可正常使用）
    registration_enabled: bool = True

    # 工具配置
    tools_config: Path = PROJECT_ROOT / "tools" / "RAGtools.json"

    system_prompt: str = """
你是一个严谨的论文检索助手，负责帮助用户在其私有论文库中查找信息。

【工作流程】
1. 默认先调 search_rag 检索，再用结果组织回答。
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

    def ensure_runtime(self) -> None:
        """启动校验：配置缺失 → 拒绝启动（付费资源）或按环境分级处理。

        - 任何环境：LLM 关键配置缺失 → 直接抛错（防止误用他人付费 key）；
        - 生产（APP_ENV=prod）：JWT_SECRET / RAG_API_KEY / 生产库缺失 → 抛错；
        - 开发：JWT_SECRET / RAG_API_KEY 缺失 → 仅告警，不阻塞。
        """
        missing_llm = [
            name
            for name, val in [
                ("LLM_BASE_URL", self.llm_base_url),
                ("LLM_API_KEY", self.llm_api_key),
                ("LLM_MODEL", self.llm_model),
            ]
            if not val
        ]
        if missing_llm:
            # 演进说明（BYOK，见 architecture.md §11）：
            # 现状要求全局 LLM 非空（内测期部署者 key 兜底）；未来做用户自带 LLM 配置
            # 后，这里应放宽为"全局配置 或 BYOK 已启用"二选一，允许 LLM_* 为空启动，
            # 按请求从 user_settings 表读用户配置构建客户端。
            raise RuntimeError(
                "启动失败：未配置 LLM（OpenAI 兼容服务）。LLM 为付费资源，请配置你自己的"
                f" {', '.join(missing_llm)}（不要复用他人 key）；"
                "本地开发可在 .env 中填写"
            )

        if self.app_env == "prod":
            missing = []
            if not self.jwt_secret:
                missing.append("JWT_SECRET")
            if not self.rag_api_key:
                missing.append("RAG_API_KEY")
            if self.database_url == _DEFAULT_DATABASE_URL:
                missing.append("DATABASE_URL（当前仍是本地开发默认值）")
            if missing:
                raise RuntimeError(
                    f"启动失败（APP_ENV=prod）：缺少必需配置 {missing}，"
                    "请通过环境变量 / 部署平台 secret 注入"
                )
            return

        # 开发环境：非付费关键项缺失仅告警
        warn = []
        if not self.jwt_secret:
            warn.append("JWT_SECRET（为空 = 任何人可伪造登录 token，建议生成随机串填入 .env）")
        if not self.rag_api_key:
            warn.append("RAG_API_KEY（为空 = RAG 服务间鉴权未启用）")
        if warn:
            logger.warning("开发环境配置缺失（仅告警）：%s", "；".join(warn))


settings = Settings()
