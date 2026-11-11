"""纯协议，规定loop怎么跟模型对话"""


from __future__ import annotations

from typing import Protocol
from collections.abc import AsyncIterator

from agent.message import AgentMessage
from agent.tools import AgentTool

class CancellationToken(Protocol):
    def is_canclled(self)->bool:
        """返回当前stream是否被取消"""
        ...

class ModelProvider(Protocol):

    def stream_response(
            self,
            *,
            model:str,
            system:str,
            message: list[AgentMessage],
            tools: list[AgentTool],
            signal : CancellationToken |None = None,
            session_id : str| None = None,
    ):
        """厂商无关的流式输出封装"""
        ...