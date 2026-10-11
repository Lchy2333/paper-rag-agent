"""依赖注入：让路由拿到 app 级组件。"""
from fastapi import Request


def get_graph(request: Request):
    return request.app.state.graph
