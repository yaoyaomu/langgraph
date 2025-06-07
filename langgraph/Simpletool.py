from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage #所有信息的父类
from langchain_core.messages import ToolMessage #在调用tool后将信息传回给LLM
from langchain_core.messages import SystemMessage #为LLM提供指导的信息（prompting）
from langchain_community.llms import Tongyi
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a: int, b: int):
    """Add two numbers. 这个tool这里的注释是必要的不然会报错"""
    return a + b

def ssubtract(a: int, b: int):
    """Subtract two numbers."""
    return a - b

def multiply(a: int, b: int):
    """Multiply two numbers."""
    return a * b

tools = [add, ssubtract, multiply]  # 定义工具列表
# 注意：如果使用的是ChatTongyi，确保安装了langchain-community库，并且版本支持工具调用。

model = ChatTongyi(model="qwen-plus").bind_tools(tools)

def model_call(state:AgentState) -> AgentState:
    system_prompt = SystemMessage(content="You are a helpful assistant.")

    response = model.invoke([system_prompt] + state["messages"])
    print(state["messages"])
    return {"messages":[response]}

def should_continue(state:AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    
graph = StateGraph(AgentState) 
graph.add_node("our_agent",model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools",tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    }
)
graph.add_edge("tools", "our_agent")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

# def print_stream(stream):
#     for s in stream:
#         message = s["messages"][-1]
        
#         if isinstance(message, BaseMessage.HumanMessage):
#             print(f"Human: {message.content}")
#         elif isinstance(message, BaseMessage.AIMessage):
#             # 如果有tool_calls，则打印工具调用信息
#             if message.tool_calls:
#                 # 这里我们只简单打印第一个工具调用（如果有多个）
#                 tool_call = message.tool_calls[0]
#                 print(f"AI wants to call tool {tool_call['name']} with arguments {tool_call['args']}")
#             else:
#                 print(f"AI: {message.content}")
#         elif isinstance(message, ToolMessage):
#             print(f"Tool ({message.name}): {message.content}")
#         else:
#             print(message)

inputs = {"messages":[("user","Add 40 + 12 and then multiply the result by 6,并且给我讲一个笑话")]}  # 输入消息，注意这里的格式是一个元组列表，元组的第一个元素是角色（user或assistant），第二个元素是消息内容
print("Streaming output:")
print_stream(app.stream(inputs,stream_mode="values"))  # or "all" for all messages
