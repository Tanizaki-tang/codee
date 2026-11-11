"""手搓ReAct循环"""

async def run_agent(
        
):
    """agent核心循环"""
    ...
    #1.初始化new_message，注入prompt
    #2.检查max_turns，异常直接结束
    #3.初始化工具索引、turn计数器、pending（上轮遗留消息）
    #4.开启Agent对话循环
    # 4.1 内层注入pending引导信息并清空pending
    # 4.2 检查max_turn
    # 4.3调用LLM拿到AssistantMessage
    # 4.4 检查stop_reason，异常则退出
    # 4.5 遍历tool_calls调用工具，并收集结果加入上下文
    # 4.6 turn+=1并将新steering加入pending
    # 5. 结束