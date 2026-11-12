"""没有用框架，而是手动实现Message"""

from __future__ import annotations

from time import time
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from collections.abc import Iterable


def _to_camel(name:str) ->str:
    '''OpenAI接口用小驼峰规范'''
    parts = name.split("_")
    return parts[0]+"".join(part.title() for part in parts[1:])

def current_timestamp_ms()->int:
    return int(time()*1000)

class WireModel(BaseModel):
    #注意pydantic只认“model_config”，叫别的名字没用
    model_config = ConfigDict(
        extra="forbid",
        validate_by_name=True,
        validate_by_alias=True,
        serialize_by_alias=True,
        alias_generator=_to_camel,
    )

class UsageCost(WireModel):
    input:float = 0.0
    output:float =0.0
    cache_read:float = 0.0
    cache_write:float = 0.0
    total:float =0.0

class Usage(WireModel):
    input:int =0
    output: int =0
    cache_read:int =0
    cache_write:int =0
    cache_write_1h:int |None= None
    reasoning:int |None = None
    total_tokens:int =0
    cost:UsageCost = UsageCost()

def sum_usage(usages: Iterable[Usage])->Usage:
    """为单次或多次请求计算总开销"""
    items = tuple(usages)

    def optional_total(field:Literal['cache_write_1h','reasoning'])->int |None:
        values = [getattr(item,field) for item in items]
        return(
            sum(v for v in values in values is not None)
            if any(v is not None for v in values)
            else None
        )
    return Usage(
        input=sum(item.input for item in items),
        output=sum(item.output for item in items),
         cache_read=sum(item.cache_read for item in items),
        cache_write=sum(item.cache_write for item in items),
        cache_write_1h=optional_total("cache_write_1h"),
        reasoning=optional_total("reasoning"),
        total_tokens=sum(item.total_tokens for item in items),
        cost=UsageCost(
            input=sum(item.cost.input for item in items),
            output=sum(item.cost.output for item in items),
            cache_read=sum(item.cost.cache_read for item in items),
            cache_write=sum(item.cost.cache_write for item in items),
            total=sum(item.cost.total for item in items),
        ),    
    )

class ResponseTiming(WireModel):
    """单个助手回复时间"""
    time_to_first_output_ms: int | None = Field(default=None, ge=0) #ge: greater or equal
    total_duration_ms : int = Field(ge=0)


class TextContent(WireModel):
    type: Literal['text']= 'text'
    text:str
    text_signature: str | None = None   #signature:校验摘要，保证数据没变篡改

class ThinkingContent(WireModel):
    type : Literal['thinking'] ="thinking"
    thinking: str
    thinking_signature: str | None = None
    redacted: bool = False #是否被隐藏

class ImageContent(WireModel):
    type: Literal["image"] = "image"
    data: str
    mime_type: str #MIME类型，如mime_type="image/png"

class ToolCall(WireModel):
    type: Literal['toolCall'] = 'toolCall'
    id : str
    name : str
    arguments: dict[str, JSONValue] = Field(default_factory=dict)
    thought_signature: str |None = None    

type UserContent = str | list[TextContent | ImageContent]
type AssistantContent = TextContent | ThinkingContent | ToolCall
type ToolResultContent = TextContent | ImageContent

class UserMessage(WireModel):
    role: Literal["user"] = "user"
    content: UserContent
    timestamp: int = Field(default_factory=current_timestamp_ms)

    @property
    def text(self) -> str:
        return content_text(self.content) #处理多种类型的UserContent

def content_text(content: str | list[Any]) -> str:
    """将字符串/图片转换为规范化内容格式."""
    if isinstance(content, str):
        return content
    return "".join(block.text for block in content if isinstance(block, TextContent))

class AssistantDiagnosticError(WireModel):
    name:str|None = None
    message:str 
    stack:str |None = None
    code : str | int | None = None

class AssistantMessageDiagnostic(WireModel):
    type: str
    timestamp: int = Field(default_factory=current_timestamp_ms)
    error: AssistantDiagnosticError | None = None
    details: dict[str, JSONValue] | None = None

StopReason = Literal["stop", "length", "toolUse", "error", "aborted"]

class AssistantMessage(WireModel):
    role: Literal["assistant"] = "assistant"
    content: list[AssistantContent] = Field(default_factory=list)
    api: str = "unknown"
    provider: str = "unknown"
    model: str = "unknown"
    response_model: str | None = None
    response_provider: str | None = None
    response_id: str | None = None
    diagnostics: list[AssistantMessageDiagnostic] | None = None
    usage: Usage = Usage()
    timing: ResponseTiming | None = None
    stop_reason: StopReason = "stop"
    error_message: str | None = None
    timestamp: int = Field(default_factory=current_timestamp_ms)

    @model_validator(mode="before")  
    #作用：
    # 在 Pydantic 开始做字段校验、解析、转成模型实例之前，
    # 先跑这个函数，预处理原始输入字典
    @classmethod
    def _normalize_convenient_content(cls, value:object) ->object:
        """开发测试用才支持接收字符串，正常调用都是block数组"""
        if not isinstance(value, dict):
            return value
        data = dict(value)
        content = data.get("content")
        if isinstance(content, str):
            data["content"] = [TextContent(text=content)] if content else []
        usage = data.get("usage")
        if usage is None:
            data["usage"] = Usage()
        return data

    @property
    def text(self) -> str:
        return "".join(block.text for block in self.content if isinstance(block, TextContent))

    @property
    def thinking_text(self) -> str:
        return "".join(
            block.thinking for block in self.content if isinstance(block, ThinkingContent)
        )

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        return tuple(block for block in self.content if isinstance(block, ToolCall))


class ToolResultMessage(WireModel):
    role: Literal["toolResult"] = "toolResult"
    tool_call_id: str   #对应ToolCall的id，用于绑定
    tool_name: str
    content: list[ToolResultContent] = Field(default_factory=list)
    details: JSONValue = None   #工具执行细节，调试开发用
    added_tool_names: list[str] | None = None
    is_error: bool = False
    timestamp: int = Field(default_factory=current_timestamp_ms)

    @model_validator(mode="before")
    @classmethod
    def _normalize_convenient_content(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        content = data.get("content")
        if isinstance(content, str):
            data["content"] = [TextContent(text=content)] if content else []
        return data

    @property
    def text(self) -> str:
        return content_text(self.content)

class BashExecutionMessage(WireModel):
    role: Literal["bashExecution"] = "bashExecution"
    command: str
    output: str
    exit_code: int | None = None
    cancelled: bool = False
    truncated: bool = False
    full_output_path: str | None = None
    timestamp: int = Field(default_factory=current_timestamp_ms)
    exclude_from_context: bool = False


class CustomMessage(WireModel):
    role: Literal["custom"] = "custom"
    custom_type: str
    content: UserContent
    display: bool = True
    details: JSONValue = None
    timestamp: int = Field(default_factory=current_timestamp_ms)

    @property
    def text(self) -> str:
        return content_text(self.content)


class BranchSummaryMessage(WireModel):
    role: Literal["branchSummary"] = "branchSummary"
    summary: str
    from_id: str
    timestamp: int = Field(default_factory=current_timestamp_ms)


class CompactionSummaryMessage(WireModel):
    role: Literal["compactionSummary"] = "compactionSummary"
    summary: str
    tokens_before: int
    timestamp: int = Field(default_factory=current_timestamp_ms)


type AgentMessage = Annotated[
    UserMessage
    | AssistantMessage
    | ToolResultMessage
    | BashExecutionMessage
    | CustomMessage
    | BranchSummaryMessage
    | CompactionSummaryMessage,
    Field(discriminator="role"),
]


def assistant_content(
    text: str,
    tool_calls: list[ToolCall] | tuple[ToolCall, ...] = (),
) -> list[AssistantContent]:
    blocks: list[AssistantContent] = [TextContent(text=text)] if text else []
    blocks.extend(tool_calls)
    return blocks


def message_to_user(message: AgentMessage) -> UserMessage:
    """将本地消息转换为OpenAI接口兼容的格式"""
    return UserMessage(content=message_text(message), timestamp=message.timestamp)


def message_text(message: AgentMessage) -> str:
    """Return the user-visible text represented by an agent message."""
    if isinstance(message, (UserMessage, AssistantMessage, ToolResultMessage, CustomMessage)):
        return message.text
    if isinstance(message, (BranchSummaryMessage, CompactionSummaryMessage)):
        return message.summary
    if isinstance(message, BashExecutionMessage):
        return message.output
    return ""
