"""手搓ReAct循环"""

from __future__ import annotations
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from time import monotonic_ns
import asyncio

from agent.events import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from agent.messages import (
    AgentMessage,
    AssistantMessage,
    ResponseTiming,
    TextContent,
    ToolCall,
    ToolResultMessage,
)
from agent.provider import CancellationToken, ModelProvider
from agent.provider_events import (
    AssistantDoneEvent,
    AssistantErrorEvent,
    AssistantMessageEvent,
    AssistantStartEvent,
    TextDeltaEvent,
    ThinkingDeltaEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from agent.tool_history import repair_tool_history
from agent.tools import AgentTool, AgentToolResult

def _error_message(model: str, message: str) -> AssistantMessage:
    return AssistantMessage(
        model=model,
        content=[],
        stop_reason="error",
        error_message=message,
    )


def _aborted_message(model: str, reason: str | None) -> AssistantMessage:
    return AssistantMessage(
        model=model,
        content=[],
        stop_reason="aborted",
        error_message=reason or "Operation aborted",
    )

async def _assistant_events(
    *,
    provider: ModelProvider,
    model: str,
    system: str,
    messages: list[AgentMessage],
    tools: list[AgentTool],
    signal: CancellationToken | None,
    session_id: str | None,
) -> AsyncIterator[AgentEvent]:
    source: AsyncIterator[AssistantMessageEvent] = provider.stream_response(
        model=model,
        system=system,
        messages=messages,
        tools=tools,
        signal=signal,
        session_id=session_id,
    )
    started = False
    provider_elapsed_ns = 0
    first_output_elapsed_ns: int | None = None
    source_iterator = source.__aiter__()
    while True:
        wait_started_ns = monotonic_ns()
        try:
            event = await anext(source_iterator)
        except StopAsyncIteration:
            break
        provider_elapsed_ns += max(0, monotonic_ns() - wait_started_ns)
        if first_output_elapsed_ns is None and isinstance(
            event,
            (
                TextDeltaEvent,
                ThinkingDeltaEvent,
                ToolCallStartEvent,
                ToolCallDeltaEvent,
                ToolCallEndEvent,
            ),
        ):
            first_output_elapsed_ns = provider_elapsed_ns
        if isinstance(event, AssistantStartEvent):
            started = True
            yield MessageStartEvent(message=event.partial)
        elif isinstance(event, AssistantDoneEvent):
            event.message.timing = _response_timing(
                first_output_elapsed_ns,
                provider_elapsed_ns,
            )
            if not started:
                yield MessageStartEvent(message=event.message)
            yield MessageEndEvent(message=event.message)
        elif isinstance(event, AssistantErrorEvent):
            event.error.timing = _response_timing(
                first_output_elapsed_ns,
                provider_elapsed_ns,
            )
            if not started:
                yield MessageStartEvent(message=event.error)
            yield MessageEndEvent(message=event.error)
        else:
            yield MessageUpdateEvent(
                message=event.partial,
                assistant_message_event=event,
            )

def _response_timing(
    first_output_elapsed_ns: int | None,
    total_elapsed_ns: int,
) -> ResponseTiming:
    """Build persistable durations from time spent awaiting provider events."""
    return ResponseTiming(
        time_to_first_output_ms=(
            first_output_elapsed_ns // 1_000_000 if first_output_elapsed_ns is not None else None
        ),
        total_duration_ms=total_elapsed_ns // 1_000_000,
    )

def _error_result():
    ...

BeforeToolCall = Callable[[ToolCall],Awaitable[tuple[bool,str | None]]]
AfterToolCall = Callable[
    [ToolCall,AgentToolResult,bool],
    Awaitable[tuple[AgentToolResult, bool]],
]

async def _run_tool(
        tool:AgentTool,
        call: ToolCall,
        signal: CancellationToken | None
)-> tuple[AgentToolResult,bool,list[AgentToolResult]]:
    updates : list[AgentToolResult]=[]
    accepting = True

    def on_update(partial: AgentToolResult)->None:
        if accepting:
            updates.append(partial.model_copy(deep=True))

    try:
        result = await tool.execute(call.id,call.arguments,signal, on_update)
        return result, False, updates
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return _error_result(str(exc)), True, updates
    finally:
        accepting = False

async def _execute_tool_call(
        call:ToolCall,
        tools: Mapping[str, AgentTool],
        signal: CancellationToken | None,
        before_tool_call : BeforeToolCall | None,
        after_tool_call : AfterToolCall | None,
)-> AsyncIterator[AgentEvent]:
    yield ToolExecutionStartEvent(
        tool_call_id=call.id,
        tool_name=call.name,
        args=call.arguments
    )

    blocked = False
    block_reason : str | None = None
    if before_tool_call is not None:
        blocked, block_reason = await before_tool_call(call)

    if blocked:
        result = _error_result(block_reason or "Tool execution was blocked")
        is_error = True
    elif signal is not None and signal.is_canclled():
        result = _error_result("Operation aborted")
        is_error = True
    else:
        tool = tools.get(call.name)
        if tool is None:
            result = _error_result(f"Tool {call.name} not found")
            is_error = True
        else:
            result, is_error, updates = await _run_tool(tool,call,signal)   #signal是对象引用，执行过程中可变
            for update in updates:
                yield ToolExecutionUpdateEvent(
                    tool_call_id=call.id,
                    tool_name=call.name,
                    args = call.arguments,
                    partial_result = update
                )

        if after_tool_call is not None:
            result, is_error = await after_tool_call(call,result, is_error)

        yield ToolExecutionEndEvent(
            tool_call_id=call.id,
            tool_name=call.name,
            result=result,
            is_error=is_error
        )
        message = ToolResultMessage(
            tool_call_id=call.id,
            tool_name=call.name,
            result=result,
            is_error=is_error
        )
        yield MessageStartEvent(message=message)
        yield MessageEndEvent(message=message)

async def _provider_context(messages:list[AgentMessage])->list[AgentMessage]:
    """过滤失败信息，给LLM清洁后的上下文"""
    replayable = tuple(
        message for message in messages
        if not(
            isinstance(message,AssistantMessage)
            and message.stop_reason in {'error','aborted'}
            and not message.content
        )
    )
    return list(repair_tool_history(replayable).messages)


async def run_agent_loop(
    *,
    provider: ModelProvider,
    model : str,
    system: str,
    messages: list[AgentMessage],
    tools: list[AgentTool],
    prompts: Sequence[AgentMessage] =(),
    prelude_message: Sequence[AgentMessage]=(),
    max_turns: int | None= None,
    signal: CancellationToken|None = None,
    session_id:str | None = None,
    get_steering_messages: Callable[[],Sequence[AgentMessage]] | None = None,
    get_follow_up_messages: Callable[[],Sequence[AgentMessage]] |None = None,
    before_tool_call : BeforeToolCall | None = None,
    after_tool_call: AfterToolCall | None = None,
):
    """agent核心循环"""
    #1.初始化new_message，注入prompt
    new_messages =list(prompts)
    for prompt in prompts:
        messages.extend(prompt)

    #2.检查max_turns，异常直接结束
    if max_turns is not None and max_turns<1:
        error = _error_message(model,"max_turns must be at least 1")

    #3.初始化工具索引、turn计数器、pending（上轮遗留消息）
    turn = 1
    tool_by_name = {tool.name : tool for tool in tools}
    first_turn = True
    pending = tuple(get_steering_messages() if get_steering_messages else ())
    stop_note :str |None = None

    #4.开启Agent对话循环
    while True:
        has_more_tools = True
        while has_more_tools or pending:
            if not first_turn:
                first_turn=True

            # 4.1 内层注入pending引导信息并清空pending
            for msg in pending:
                messages.append(msg)
                new_messages.append(msg)
            pending =()

            # 4.2 检查max_turn
            if max_turns is not None and turn >max_turns:
                error = _error_message(model, f"Agent stopped after max_turns={max_turns}")
                messages.append(error)
                new_messages.append(error)

            if signal is not None and signal.is_canclled():
                aborted = _aborted_message(model, stop_note)
                messages.append(aborted)
                new_messages.append(aborted)

             # 4.3调用LLM拿到AssistantMessage
            assistant = None
            async for event in _assistant_events(
                provider=provider,
                model=model,
                system=system,
                messages=_provider_context(messages),
                tools=tools,
                signal=signal,
                session_id=session_id
            ):
                yield event 
                if isinstance(event,MessageEndEvent) and isinstance(event.message, AssistantMessage):
                    assistant = event.message

            if assistant is None:  # defensive: _assistant_events always terminates
                assistant = _error_message(model, "Provider produced no assistant message")
                yield MessageStartEvent(message=assistant)
                yield MessageEndEvent(message=assistant)

            messages.append(assistant)
            new_messages.append(assistant)

            # 4.4 检查stop_reason，异常则退出
            if assistant.stop_reason in {"error", "aborted"}:
                yield TurnEndEvent(message=assistant)
                yield AgentEndEvent(messages=new_messages)
                return

            # 4.5 遍历tool_calls调用工具，并收集结果加入上下文
            tool_results: list[ToolResultMessage] = []
            calls = list(assistant.tool_calls)
            has_more_tools = bool(calls)
            for call in calls:
                cancelled_before = signal is not None and signal.is_cancelled()
                async for event in _execute_tool_call(
                    call,
                    tool_by_name,
                    signal,
                    before_tool_call,
                    after_tool_call,
                ):
                    yield event
                    if isinstance(event, MessageEndEvent) and isinstance(
                        event.message, ToolResultMessage
                    ):
                        tool_results.append(event.message)
                        messages.append(event.message)
                        new_messages.append(event.message)
                        if (
                            stop_note is None
                            and not cancelled_before
                            and signal is not None
                            and signal.is_cancelled()
                            and event.message.is_error
                        ):
                            stop_note = event.message.text or None
                yield TurnEndEvent(message=assistant, tool_results=tool_results)
                # 4.6 turn+=1并将新steering加入pending
                turn += 1
                pending = tuple(get_steering_messages() if get_steering_messages else ())

                follow_ups = tuple(get_follow_up_messages() if get_follow_up_messages else ())
                if follow_ups:
                    pending = follow_ups
                    continue
                break
            # 5. 结束
            yield AgentEndEvent(messages=new_messages)
    
    

    
