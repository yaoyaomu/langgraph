from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage #所有信息的父类
from langchain_core.messages import HumanMessage #人类用户发送的信息
from langchain_core.messages import ToolMessage #在调用tool后将信息传回给LLM
from langchain_core.messages import SystemMessage #为LLM提供指导的信息（prompting）
# from langchain_community.llms import Tongyi
from langchain_community.embeddings import TongyiEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi
# from langchain_community.embeddings import TongyiEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import os

load_dotenv()

llm = ChatTongyi(model="qwen-plus", temperature=0)

embeddings = Tongyi(model="text-embedding-v2")
# embedding 要适配与 LLM模型


pdf_path = "Stock_Market_Performance_2024.pdf"

# 检查PDF文件是否存在
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF文件没有找到: {pdf_path}")

pdf_loader = PyPDFLoader(pdf_path)

# 加载PDF文档
try:
    pages = pdf_loader.load()
    print(f"PDF文件加载成功，共有 {len(pages)} 页。")
except Exception as e:
    print(f"加载PDF文件时出错: {str(e)}")
    raise 

# 分割文档
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, # 每个chunk的大小
    chunk_overlap=200 # 每个chunk之间的重叠部分
    )


pages_split = text_splitter.split_documents(pages) #应用分割器

persist_dictionary = r"/root/autodl-tmp/Rag_Agent" #持久化目录
collection_name = "stock_market_performance" #集合名称

if not os.path.exists(persist_dictionary):
    os.makedirs(persist_dictionary)

# 创建或加载Chroma集合
try:
    # Here, we actually create the chroma database using our embeddigns model
    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_dictionary,
        collection_name=collection_name
    )
    print(f"Created ChromaDB vector store!")
    
except Exception as e:
    print(f"Error setting up ChromaDB: {str(e)}")
    raise

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5} # K is the amount of chunks to return
)

@tool
def retrieve(query: str) -> str:
    """Retrieves relevant documents based on the query."""
   
    docs = retriever.invoke(query)
    if not docs:
        return "没有找到相关文档。"
    
    result=[]
    for i,doc in enumerate(docs):
        result.append(f"文档 {i+1}:\n{doc.page_content}\n")

    return "\n\n".join(result)

tools = [retrieve]  # 定义工具列表

llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def should_continue(state: AgentState) -> AgentState:
    """Check if the user wants to continue or end the conversation."""
    result = state["messages"][-1]
    return hasattr(result, "tool_calls") and len(result.tool_calls) > 0 


system_prompt = """
You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 based on the PDF document loaded into your knowledge base.
Use the retriever tool available to answer questions about the stock market performance data. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.
"""

tools_dict = {our_tool.name: our_tool for our_tool in tools} # Creating a dictionary of our tools

# LLM Agent
def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state."""
    messages = list(state['messages'])
    messages = [SystemMessage(content=system_prompt)] + messages
    message = llm.invoke(messages)
    return {'messages': [message]}

# Retriever Agent
def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""

    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"呼叫工具 Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")
        
        if not t['name'] in tools_dict: # Checks if a valid tool is present
            print(f"\nTool: {t['name']} does not exist.")
            result = "工具名字不正确，请从新在工具列表中选择."
        
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"结果长度: {len(str(result))}")
            

        # Appends the Tool Message
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    print("工具处理完毕，返回模型!")
    return {'messages': results}

# Define the state graph
graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "retriever_agent", False: END}
)
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()


def running_agent():
    print("\n=== RAG AGENT===")
    
    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        messages = [HumanMessage(content=user_input)] # converts back to a HumanMessage type

        result = rag_agent.invoke({"messages": messages})
        
        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)


running_agent()

