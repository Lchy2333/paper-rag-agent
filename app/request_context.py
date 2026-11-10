"""请求级上下文：request_id 的唯一权威存放处。

两处消费、一处生产：
- 生产：main.py 的 access_log_middleware 进门时 set（透传上游 X-Request-ID 或本地生成）；
- 消费 1：main.py 的 RequestIdFilter 每条日志输出前「现场读取」，自动贴进日志行；
- 消费 2：需要跨层捎带时（如 SSE 线程池 worker），contextvar 随上下文复制机制传播。

注意：contextvar 是「本次执行流」的随身变量——并发下每个请求各存各的，
互不覆盖；这也是不能用全局变量存 rid 的原因。
"""
import contextvars
import uuid

# default="-"：请求之外（启动日志/后台线程）读到 "-" 而不是报错
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)


def new_request_id() -> str:
    """生成 12 位 rid：hex 无分隔符不与其他日志字段冲突，12 位足够 grep 且省体积。"""
    return uuid.uuid4().hex[:12]
