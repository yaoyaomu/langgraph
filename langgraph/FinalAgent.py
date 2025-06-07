import os
import asyncio
import json
from typing import Annotated, Sequence, TypedDict, List, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from crawl4ai.async_configs import LLMConfig
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import csv
from langchain_community.chat_models import ChatTongyi
from langchain_community.llms import Tongyi
from langchain_community.chat_models.tongyi import ChatTongyi
import re



load_dotenv()

# 状态定义
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    news_data: List[Dict[str, Any]]
    factor_data: List[Dict[str, Any]]
    url: str

# Pydantic models
class NewsURL(BaseModel):
    url: str = Field(..., description="提取新闻发布的链接,只返回链接,不要返回其他内容.")

class NewsContent(BaseModel):
    news_time: str = Field(..., description="新闻发布的时间,包括日期、时间,如 2023年10月5日 14:30.")
    news_title: str = Field(..., description="新闻原文的完整标题，禁止修改或缩写.")
    news_text: str = Field(..., description="新闻的全部正文内容.")
    company_involved: str = Field(..., description="新闻中所有关联的上市公司都要列出，多条公司多条记录，使用其法定注册全称.")
    stock_code: str = Field(..., description="与涉及公司对应的唯一官方证券代码.")
    stock_short_name: str = Field(..., description="与涉及公司对应的交易所标准化简称.")

class NewsImpact(BaseModel):
    company_name: str = Field(..., description="公司名称")
    stock_code: str = Field(..., description="股票代码")
    stock_short_name: str = Field(..., description="股票简称")
    news_time: str = Field(..., description="新闻时间")
    news_title: str = Field(..., description="新闻标题")
    impact_direction: int = Field(..., description="影响方向: 1(正面)/-1(负面)/0(中性)")
    news_summary: str = Field(..., description="新闻摘要")

class NewsInput(BaseModel):
    news_data: List[Dict[str, Any]] = Field(..., description="新闻数据列表")

class URLInput(BaseModel):
    user_query: str = Field(..., description="用户的查询或需求描述")

# Tools
@tool
async def crawl_and_save_news(url: str) -> List[Dict[str, Any]]:
    """
    使用 crawl4ai 从给定 URL 抓取新闻并提取内容，同时保存原始新闻数据。
    
    参数
        url： 要抓取新闻的 URL。默认为 eastmoney quick news。
        
    返回值
        包含提取新闻内容的字典列表。
    """
    async with AsyncWebCrawler() as crawler:
        # First layer: Extract news links
        links_result = await crawler.arun(
            url=url,
            config=CrawlerRunConfig(
                word_count_threshold=1,
                extraction_strategy=LLMExtractionStrategy(
                    llm_config=LLMConfig(
                        provider="deepseek/deepseek-chat",
                        api_token=os.getenv("DEEPSEEK_API_KEY"),
                    ),
                    schema=NewsURL.model_json_schema(),
                    extraction_type="schema",
                    instruction="""请提取所有新闻信息的url。示例：
                    14:11[**【为了"出海"和"还贷" 锦江酒店拟启动港股IPO】** 
                    从锦江酒店来看，其出海布局已进入实质阶段，而东南亚市场成为出海的重要落点。值得注意的是，公告还提到募资将用于偿还银行贷款。
                    [点击查看全文]](https://finance.eastmoney.com/a/202506053422839540.html).我们需要返回的只有https://finance.eastmoney.com/a/202506053422839540.html"""
                ),
                cache_mode=CacheMode.BYPASS
            )
        )
        
        if not links_result.success or not links_result.extracted_content:
            print("未找到任何新闻链接")
            return []
            
        content = json.loads(links_result.extracted_content) if isinstance(links_result.extracted_content, str) else links_result.extracted_content
        news_links = [item['url'] for item in content if isinstance(item, dict) and item.get('url')]
        
        total_links = len(news_links)
        print(f"\n总共找到 {total_links} 条新闻链接")
        print("开始提取新闻内容")        
        # # 限制处理的链接数量为20条
        # news_links = news_links[:2]
        # print(f"将处理前 20 条新闻\n")
        
        # Second layer: Extract news content
        all_news = []
        for i, news_url in enumerate(news_links, 1):
            print(f"正在处理第 {i}/{len(news_links)} 条新闻: {news_url}")
            content_result = await crawler.arun(
                url=news_url,
                config=CrawlerRunConfig(
                    word_count_threshold=1,
                    extraction_strategy=LLMExtractionStrategy(
                        llm_config=LLMConfig(
                            provider="deepseek/deepseek-chat",
                            api_token=os.getenv("DEEPSEEK_API_KEY"),
                        ),
                        schema=NewsContent.model_json_schema(),
                        extraction_type="schema",
                        instruction="""
                    请从新闻页面提取以下信息：
                                1. news_time: 新闻发布的具体时间（格式：YYYY年MM月DD日 HH:mm）
                                2. news_title: 新闻的完整标题
                                3. news_text: 新闻的完整正文内容
                                4. company_involved: 新闻中提到的上市公司全称
                                5. stock_code: 对应的股票代码（如：000001.SZ）
                                6. stock_short_name: 公司在交易所的简称

                                注意：
                                - 如果新闻涉及多家公司，请分别创建多条记录
                                - 每条记录的news_time、news_title和news_text保持相同
                                - company_involved、stock_code和stock_short_name对应每家公司的具体信息
                                - 如果找不到某个字段的信息，请返回空字符串""
                                - 返回格式必须是JSON数组"""
                    ),
                    cache_mode=CacheMode.BYPASS,
                )
            )
            
            if content_result.success and content_result.extracted_content:
                news_content = json.loads(content_result.extracted_content) if isinstance(content_result.extracted_content, str) else content_result.extracted_content

                if isinstance(news_content, list):
                    all_news.extend(news_content)
                else:
                    all_news.append(news_content)
                print("√ 提取成功")
                print(news_content)
            else:
                print("× 提取失败")

        
        print(f"\n新闻处理完成:")
        print(f"- 总链接数: {total_links}")
        print(f"- 处理链接数: {len(news_links)}")
        print(f"- 提取的公司记录数: {len(all_news)}")
        
        # Save raw news data
        if all_news:
            fieldnames = ['时间', '标题', '正文', '涉及公司', '股票代码', '股票简称']
            field_mapping = {
                'news_time': '时间',
                'news_title': '标题',
                'news_text': '正文',
                'company_involved': '涉及公司',
                'stock_code': '股票代码',
                'stock_short_name': '股票简称'
            }
            try:
                with open('news.csv', 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for item in all_news:
                        row = {field_mapping[k]: item.get(k, "") for k in field_mapping}
                        writer.writerow(row)
                print(f"- 原始新闻数据已保存到 news.csv")
            except Exception as e:
                print(f"保存CSV文件时出错: {str(e)}")
                    
        return all_news

@tool
def analyze_news_impact(input_data: NewsInput) -> List[Dict[str, Any]]:
    """分析新闻对公司的影响"""
    llm = ChatTongyi(model="qwen-plus",api_key=os.getenv("DASHSCOPE_API_KEY"))
    
    factor_data = []
    for news in input_data.news_data:
        prompt = f"""
        请分析以下新闻对公司的影响：
        
        新闻标题：{news['news_title']}
        新闻正文：{news['news_text']}
        涉及公司：{news['company_involved']}
        
        请提供：
        1. 一句话新闻摘要
        2. 影响方向评估：
           - 输出+1表示正面影响
           - 输出-1表示负面影响
           - 输出0表示中性影响
        
        请严格按照以下JSON格式返回，不要包含任何其他内容：
        {{"summary": "这里是新闻摘要", "impact": 影响方向数字}}
        """
        
        try:
            response = llm.invoke(prompt)
            # 从response.content中提取JSON字符串
            content = response.content
            # 查找JSON开始和结束的位置
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end != -1:
                json_str = content[start:end]
                # 移除JSON中数字前的+号，因为这不是有效的JSON格式
                json_str = json_str.replace('"+1"', '1').replace('"+0"', '0').replace('"+', '')
                analysis = json.loads(json_str)
                factor_data.append({
                    "company_name": news["company_involved"],
                    "stock_code": news["stock_code"],
                    "stock_short_name": news["stock_short_name"],
                    "news_time": news["news_time"],
                    "news_title": news["news_title"],
                    "impact_direction": analysis["impact"],
                    "news_summary": analysis["summary"]
                })
                print(f"成功分析新闻: {news['news_title']}")
            else:
                print(f"无法从响应中提取JSON: {content}")
        except Exception as e:
            print(f"分析新闻时出错: {str(e)}")
            print(f"错误的响应内容: {response.content if 'response' in locals() else 'No response'}")
            continue
            
    return factor_data

class SaveInput(BaseModel):
    data: List[Dict[str, Any]] = Field(..., description="要保存的数据")
    filename: str = Field(default="factor.csv", description="输出文件名")

@tool
def save_factor_data(input_data: SaveInput) -> str:
    """保存因子数据到CSV文件"""
    if not input_data.data:
        return "No data to save"

    fieldnames = ['公司名', '股票代码', '股票简称', '新闻时间', '新闻标题', '影响方向', '新闻摘要']
    field_mapping = {
        'company_name': '公司名',
        'stock_code': '股票代码',
        'stock_short_name': '股票简称',
        'news_time': '新闻时间',
        'news_title': '新闻标题',
        'impact_direction': '影响方向',
        'news_summary': '新闻摘要'
    }
    
    try:
        with open(input_data.filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for item in input_data.data:
                row = {field_mapping[k]: item[k] for k in field_mapping}
                writer.writerow(row)
        return f"Successfully saved {len(input_data.data)} records to {input_data.filename}"
    except Exception as e:
        return f"Error saving to CSV: {str(e)}"

@tool
def get_news_url(input_data: URLInput) -> str:
    """与用户对话，根据用户需求推荐合适的新闻URL"""
    # 如果输入已经是URL，直接返回
    if input_data.user_query.startswith('http'):
        return input_data.user_query
    # 否则返回默认URL
    return "https://kuaixun.eastmoney.com/ssgs.html"

# 节点函数
async def crawl_node(state: AgentState) -> AgentState:
    """新闻爬取节点"""
    url = state["url"]
    news_data = await crawl_and_save_news.ainvoke(url)
    state["news_data"] = news_data
    return state

async def analyze_node(state: AgentState) -> AgentState:
    """新闻分析节点"""
    news_input = NewsInput(news_data=state["news_data"])
    factor_data = await analyze_news_impact.ainvoke({"input_data": news_input.model_dump()})
    state["factor_data"] = factor_data
    print("analyze_node")
    return state

async def save_node(state: AgentState) -> AgentState:
    """数据保存节点"""
    input_data = SaveInput(data=state["factor_data"])
    await save_factor_data.ainvoke({"input_data": input_data.model_dump()})
    print("save_node")
    return state

async def get_url_node(state: AgentState) -> AgentState:
    """URL获取节点"""
    url_input = URLInput(user_query="我需要最新的上市公司相关新闻")
    url = await get_news_url.ainvoke({"input_data": url_input.model_dump()})
    state["url"] = url
    print(f"Selected URL based on user query: {url}")
    return state

# 工作流设置
workflow = StateGraph(
    AgentState,  # 使用我们定义的 AgentState 作为状态模式
)

# 添加节点
workflow.add_node("get_url", get_url_node)
workflow.add_node("crawl", crawl_node)
workflow.add_node("analyze", analyze_node)
workflow.add_node("save", save_node)

# 添加边
workflow.add_edge("get_url", "crawl")
workflow.add_edge("crawl", "analyze")
workflow.add_edge("analyze", "save")
workflow.add_edge("save", END)

# 设置入口
workflow.set_entry_point("get_url")

# 编译工作流
chain = workflow.compile()

# 运行工作流
async def run_workflow(user_query: str="我想了解金融财经的最新新闻"):
    # 初始化对话模型
    llm = ChatTongyi(model="qwen-plus", api_key=os.getenv("DASHSCOPE_API_KEY"))
    
    # 初始化对话历史
    conversation_history = []
    system_prompt = SystemMessage(content="""
    你是一个专业的金融新闻助手。当用户询问新闻时，请从以下URL中选择最合适的推荐给用户：
    1. https://kuaixun.eastmoney.com/ssgs.html (东方财富上市公司快讯)

    
    请直接在回答中包含选择的URL。对于其他问题，请正常回答。
    """)
    conversation_history.append(system_prompt)
    
    print("欢迎使用金融新闻助手！请输入您的问题（输入'exit'退出）：")
    
    while True:
        user_input = input("\n👤 用户: ")
        if user_input.lower() == 'exit':
            break
            
        # 添加用户消息到对话历史
        user_message = HumanMessage(content=user_input)
        conversation_history.append(user_message)
        
        # 获取AI响应
        response = llm.invoke(conversation_history)
        print(f"\n🤖 助手: {response.content}")
        conversation_history.append(response)
        
        # 检查响应中是否包含URL
        urls = re.findall(r'https?://[^\s<>"]+|www\.[^\s<>"]+', response.content)
        
        if urls:
            print("\n检测到新闻URL，是否要执行新闻爬取和分析任务？(y/n)")
            should_execute = input().lower()
            
            if should_execute == 'y':
                print("\n开始执行新闻爬取和分析任务...")
                initial_state: AgentState = {
                    "messages": [],
                    "news_data": [],
                    "factor_data": [],
                    "url": urls[0]  # 使用检测到的第一个URL
                }
                
                try:
                    result = await chain.ainvoke(initial_state)
                    print("\n✅ 任务执行完成！结果已保存到CSV文件中。")
                except Exception as e:
                    print(f"\n❌ 任务执行失败：{str(e)}")
                    
    print("\n感谢使用金融新闻助手！再见！")

if __name__ == "__main__":
    # 运行对话式工作流
    asyncio.run(run_workflow()) 