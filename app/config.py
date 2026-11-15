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
    # LLM 调用超时约束（2026-10 修复"上游挂起导致无响应"，见 architecture.md §15）：
    llm_timeout: float = 120.0  # 阶段超时（秒）：read 空闲上限——首 token / 相邻 chunk 的等待上限，
    #                            防"请求发出但数据流停滞"无限阻塞线程池
    llm_max_retries: int = 0    # 客户端自动重试：0 = 收敛到业务层 call_llm 统一重试
    #                            （只重试"首 token 前瞬态失败"），避免客户端层叠乘放大
    # 上下文窗口（tokens）：前端"上下文占用率"指示器的分母来源（见 core/model_limits.py）
    # - >0：直接用该值（火山 deepseek-v4-flash 控制台标称 1049K，先写死 1048576）
    # - 0 ：回退到 models.dev 目录，按 LLM_PROVIDER_ID + LLM_MODEL 查询
    llm_context_window: int = 1048576
    # models.dev 目录里的 provider 条目名（火山 Agent Plan 为 volcengine-coding-plan）
    llm_provider_id: str = "volcengine-coding-plan"

    # 上下文压缩（摘要 + 前 N 轮）：上次实测水位（query_usage.last_input_tokens）
    # ≥ 阈值 × 输入预算时，本次请求进图先压缩历史再对话（agent/nodes.py compress_node）
    context_compression_enabled: bool = True
    compression_threshold: float = 0.8
    compression_keep_rounds: int = 4

    # 工具轮次防线（P0；背景：压测 232s 案例 = 模型 23 轮失控检索，
    # 见 loadtest/report-压测总报告.md §3.4——空结果是合法返回值不是错误，
    # ReAct 唯一出口是"模型不再发 tool_calls"，必须用结构兜底）
    max_tool_rounds: int = 4    # 软上限：单次提问内带 tool_calls 的 AI 消息数（只数本轮，
                                # 从最后一条 HumanMessage 起算，多轮老会话的历史不计）。
                                # 达到后 route_after_llm 分流到 force_answer：裸 LLM
                                # （不绑工具，物理无法再发 tool_calls）强制收尾，
                                # 用户仍能拿到基于已有检索结果的回答（体面降级）
    recursion_limit: int = 20   # 硬上限：随 invoke config 传给 LangGraph 的保险丝，
                                # 超出抛 GraphRecursionError（gen() 捕获转 SSE error，
                                # 正常流量到不了这里——软上限才是主力）。
                                # ⚠️ langgraph 1.x 默认 10007（_internal/_config.py 的
                                # DEFAULT_RECURSION_LIMIT，可环境变量覆盖）等于无保险丝，
                                # 必须显式设置；取值需 ≥ max_tool_rounds × 3（每轮
                                # call_llm+guard+call_tool 三节点）+ compress / HITL 余量

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
2. 每个问题最多检索 2 次；两次用不同的关键词角度，不要重复相近的检索词。
3. 检索结果为空或明显无关 → 直接告知用户"库中没有相关内容"，不要继续尝试。
4. 找到主题相关但缺少问题细节的资料 → 基于已有资料回答，并明确说明
   "你问的 XX 细节在库中未找到"——这是合法的回答，不要为它继续检索。
5. 一次提问涉及多个子问题时可合并检索，但整体仍受 2 次检索上限约束。

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
