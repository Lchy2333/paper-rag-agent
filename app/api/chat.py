"""对话路由。"""
import json
import logging
import threading
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, SystemMessage

from agent.nodes import GenerationCancelled
from app import db
from app.auth import CurrentUser
from app.errors import classify_exception
from app.schemas import CancelRequest, ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

router = APIRouter()


def _sse(event: str, data: dict) -> str:
    """把事件打包成一条 SSE 文本。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _resolve_thread(req: ChatRequest, request: Request, user) -> str:
    """线程归属校验（chat 与 chat/stream 共用）：已有会话校验归属；新会话由服务端生成。"""
    if req.thread_id:
        thread = db.get_thread(request.app.state.pool, req.thread_id)
        if thread is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        if thread["user_id"] != user["id"]:
            raise HTTPException(status_code=403, detail="无权访问该会话")
        return req.thread_id
    thread_id = uuid.uuid4().hex
    db.create_thread(request.app.state.pool, thread_id, user["id"])
    return thread_id


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request, user=CurrentUser) -> ChatResponse:
    graph = request.app.state.graph
    system_prompt = request.app.state.system_prompt

    # 线程归属：已有会话校验归属（404/403）；新会话由服务端生成 thread_id 并绑定用户
    # （与 /chat/stream 共用 _resolve_thread，逻辑一致）
    thread_id = _resolve_thread(req, request, user)

    config = {"configurable": {"thread_id": thread_id, "user_id": str(user["id"])}}
    request.state.thread_id = thread_id  # 捎带给 exception_handler 做日志关联
    request.state.user_id = str(user["id"])

    # 已有会话 → 读取快照判断状态（挂起中 / 已完成 / 不存在）
    snapshot = graph.get_state(config) if req.thread_id else None

    # 防御：该会话正挂起等待审批 → 拒绝新命令，避免覆盖存档
    if snapshot is not None and snapshot.interrupts:
        raise HTTPException(
            status_code=409,
            detail="该会话正在等待审批，请先调用 /api/approve 完成审批，或开启新会话",
        )

    # 新会话带 system prompt；续聊只追加用户消息（历史里已有 system）
    msgs = [HumanMessage(content=req.query)]
    is_new = snapshot is None or not snapshot.values.get("messages")
    if is_new:
        msgs.insert(0, SystemMessage(content=system_prompt))

    result = graph.invoke({"messages": msgs}, config=config)

    if "__interrupt__" in result:
        intr = result["__interrupt__"][0]
        return ChatResponse(
            status="awaiting_approval",
            thread_id=thread_id,
            approval_payload=intr.value,
        )
    return ChatResponse(reply=result["messages"][-1].content, thread_id=thread_id)


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request, user=CurrentUser):
    """SSE 流式对话：把图执行过程实时推给前端。

    事件协议：
      event: session  data: {"thread_id"}                会话 ID（最先发，前端据此可随时取消）
      event: stage    data: {"label": "..."}             阶段进度（LLM 理解 / 检索中 / 命中N条…）
      event: token    data: {"delta": "..."}             最终回答逐 token
      event: approval data: {"thread_id","approval_payload"}   HITL 挂起，等待审批
      event: cancel   data: {"thread_id"}                用户主动停止（前端 POST /chat/cancel 触发）
      event: done     data: {"thread_id"}                正常跑完
      event: error    data: {"message"}                  执行失败
    """
    graph = request.app.state.graph
    system_prompt = request.app.state.system_prompt

    thread_id = _resolve_thread(req, request, user)
    request.state.thread_id = thread_id
    request.state.user_id = str(user["id"])

    # 取消标记：独立请求 /chat/cancel 置位，节点逐 token 检查。
    # 存进 app.state.cancels（进程内 dict），生成结束在 finally 清理。
    cancel_event = threading.Event()
    request.app.state.cancels[thread_id] = cancel_event

    config = {"configurable": {
        "thread_id": thread_id,
        "user_id": str(user["id"]),
        "cancel_event": cancel_event,
    }}

    # 已有会话 → 读取快照判断状态（挂起中 / 已完成 / 不存在）
    snapshot = graph.get_state(config) if req.thread_id else None
    if snapshot is not None and snapshot.interrupts:
        raise HTTPException(
            status_code=409,
            detail="该会话正在等待审批，请先调用 /api/approve 完成审批，或开启新会话",
        )

    msgs = [HumanMessage(content=req.query)]
    is_new = snapshot is None or not snapshot.values.get("messages")
    if is_new:
        msgs.insert(0, SystemMessage(content=system_prompt))

    def gen():
        try:
            # 先把会话 ID 发给前端：新会话时前端据此才拿到 thread_id，才能调用 /chat/cancel
            yield _sse("session", {"thread_id": thread_id})
            # 注意：用同步 graph.stream，而不是 astream——
            # PostgresSaver 只实现了同步 checkpoint 方法，astream 会调它没实现的
            # aget_tuple 抛 NotImplementedError（即之前看到的"服务内部错误"）。
            # 同步 stream 由 FastAPI 在线程池里迭代，不会阻塞事件循环。
            for mode, chunk in graph.stream(
                {"messages": msgs},
                config=config,
                stream_mode=["values", "custom", "messages"],
            ):
                if mode == "custom":
                    # call_llm / call_tool / HttpTool 转发的阶段进度
                    if isinstance(chunk, dict):
                        yield _sse("stage", chunk)
                elif mode == "messages":
                    # LLM token 增量：列表模式下标出的是 (message, metadata) 元组。
                    # 只转发 call_llm 轮的模型输出（call_tool 的 ToolMessage 会混进来，必须过滤掉）。
                    msg, meta = chunk if isinstance(chunk, tuple) else (chunk, {})
                    if meta.get("langgraph_node") != "call_llm":
                        continue
                    text = msg.content if getattr(msg, "content", None) else ""
                    if isinstance(text, list):  # 多模态 content 可能是内容块列表
                        text = "".join(
                            str(b.get("text", "")) for b in text if isinstance(b, dict)
                        )
                    if text:
                        yield _sse("token", {"delta": text})
                elif mode == "values":
                    # 图挂起等待审批 → 推 approval 事件后结束
                    if "__interrupt__" in chunk:
                        intr = chunk["__interrupt__"][0]
                        yield _sse("approval", {
                            "thread_id": thread_id,
                            "approval_payload": intr.value,
                        })
                        return
            yield _sse("done", {"thread_id": thread_id})
        except GenerationCancelled:
            # 用户主动停止 → 推 cancel 事件，前端显示"已停止生成"
            yield _sse("cancel", {"thread_id": thread_id})
        except Exception as e:
            # 流中途失败（LLM 报错等）→ 用 SSE error 事件收尾，避免前端白屏
            logger.error("chat_stream_failed thread_id=%s exc_type=%s exc_msg=%s",
                         thread_id, type(e).__name__, str(e)[:500])
            logger.debug("chat_stream_failed_traceback", exc_info=True)
            _, code, message = classify_exception(e)
            yield _sse("error", {"error_code": code, "message": message})
        finally:
            # 清理取消标记，防止内存泄漏 / 下次误置位
            request.app.state.cancels.pop(thread_id, None)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    })


@router.post("/chat/cancel")
def chat_cancel(req: CancelRequest, request: Request, user=CurrentUser):
    """中断指定会话正在进行的生成（幂等）。

    只置位取消标记（threading.Event）；正在跑的图会在下一个 token / 工具边界
    检查到并抛出 GenerationCancelled，/chat/stream 随即以 cancel 事件收尾。
    归属校验与 /chat 一致：不是本人会话 → 403。
    """
    thread = db.get_thread(request.app.state.pool, req.thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    if thread["user_id"] != user["id"]:
        raise HTTPException(status_code=403, detail="无权访问该会话")

    cancel_event = request.app.state.cancels.get(req.thread_id)
    if cancel_event is not None:
        cancel_event.set()
        return {"status": "cancelled"}
    # 幂等：该会话当前没有正在进行的生成也算成功
    return {"status": "no_active_generation"}
