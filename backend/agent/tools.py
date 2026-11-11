"""规定工具执行契约"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import Field, model_validator

from agent.message import ImageContent,TextContent,ToolCall,WireModel

#Protocol:鸭子类型，长得像鸭子那就是鸭子
class ToolCancellationToken(Protocol):
    def is_cancelled(self) -> bool:
        """返回：工具执行是否应该停止"""
        ...

class AgentToolResult(WireModel):
    """工具执行的部分/全部结果"""
    ...

class ToolCallRenderer(Protocol):
    def __call__(self, arguments:Mapping[str, JSONValue])->str | None:
        """供前端渲染使用"""
        ...

class ToolResultRenderer(Protocol):
    def __call__():
        ...

ToolUpdateCallback = Callable[[AgentToolResult], None]

class ToolExecutor(Protocol):
    def __call__(
            self, 
            tool_call_id: str,
            arguments:Mapping[str, JSONValue],  
            signal:ToolCancellationToken | None = None, #是否中断
            on_update :  ToolUpdateCallback | None = None #回调函数，可用于中途向外部传递进度
        )-> Awaitable[AgentToolResult]:
        ...
#Mapping包含:
# 1.普通dict
# 2.只读字典MappingProxyType
# 3.第三方只读映射对象


ToolExecutionMode = Literal['sequential','parrallel']

#把原始输入对象转换为工具执行器需要的arguments参数的转换器函数类型
# 输入 object
# 返回 Mapping[str,JSONValude
# 作用：给此类函数起了类型别名
ToolArgumentPreparer = Callable[[object],Mapping[str,JSONValude]]

@dataclass(forzen=True, slots=True)
class AgentTool:
    name:str
    label:str
    description:str
    parameters: Mapping[str, JSONValue]
    execute_fn : ToolExecutor
    prompt_snippet :str |None = None #可选提示词片段
    prompt_guidelines:tuple[str,...] =() #可选，多条提示指引，多条独立规则，会注入提示词用于约束行为
    prepare_arguments: ToolArgumentPreparer|None = None #参数预处理器
    execution_mode: ToolExecutionMode='parrallel' #默认并行
    render_call : ToolCallRenderer |None = None #渲染工具调用请求的回调
    render_result: ToolResultRenderer | None = None

    @property
    def input_schema(self)->Mapping[str,JSONValue]:
        """兼容Anthropic的命名方法"""
        return self.parameters

    async def execute(
            self,
            tool_call_id:str,
            arguments: Mapping[str,JSONValue],
            signal : ToolCancellationToken | None = None,
            on_update : ToolUpdateCallback | None = None,
    )-> AgentToolResult:
        """执行一次工具调用"""
        return await self.execute_fn(tool_call_id,arguments,signal,on_update)

__all__ = [
    "AgentTool",
    "AgentToolResult",
    "ToolCall",
    "ToolCallRenderer",
    "ToolCancellationToken",
    "ToolExecutionMode",
    "ToolResultRenderer",
    "ToolExecutor",
    "ToolUpdateCallback",
]