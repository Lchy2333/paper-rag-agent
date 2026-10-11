"""对话路由。"""
from fastapi import APIRouter, Depends, Request
from langchain_core.messages import HumanMessage, SystemMessage

from app.schemas import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request) -> ChatResponse:
    graph = request.app.state.graph
    system_prompt = request.app.state.system_prompt

    initial_state = {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=req.query),
        ]
    }

    final_state = graph.invoke(initial_state)
    reply = final_state["messages"][-1].content
    return ChatResponse(reply=reply)
