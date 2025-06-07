from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage,AIMessage
# from langchian_openai import ChatOpenAI
from langchain_community.llms import Tongyi
from langchain_community.chat_models.tongyi import ChatTongyi
from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv # 存储密钥等配置文件。Load environment variables from .env file
import os

load_dotenv()  # Load environment variables from .env file

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

llm = ChatTongyi(model="qwen-plus")

def process(state: AgentState)->AgentState:
    """
    Process the current state and generate a response using the LLM.
    """
    # Generate a response using the LLM
    response = llm.invoke(state['messages'])
    
    print(f"\nAI Response: {response.content}\n")
    # Append the AI response to the messages
    state['messages'].append(AIMessage(content=response.content))
    print(f"Current State: {state['messages']}\n")
    
    return state

graph = StateGraph(AgentState)
graph.add_node("process",process)
graph.add_edge(START,"process")
graph.add_edge("process",END)
agent = graph.compile()

conversation_history = []

user_input = input("Enter: ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))

    result = agent.invoke({"messages": conversation_history})

    # print(result["messages"])
    conversation_history = result["messages"]
    user_input = input("Enter: ")

with open("logger.txt","w") as f:
    f.write("对话记录：\n")

    for message in conversation_history:
        if isinstance(message, HumanMessage):
            f.write(f"User: {message.content}\n")
        elif isinstance(message, AIMessage):
            f.write(f"AI: {message.content}\n")
    f.write("\n对话结束。\n")

print("对话结束，记录已保存到 logger.txt")


