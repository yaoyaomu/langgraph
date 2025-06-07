import os
import asyncio
import json
from typing import Annotated, Sequence, TypedDict, List, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from crawl4ai.async_configs import LLMConfig
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import csv
from langchain_community.chat_models import ChatTongyi
from langchain_community.llms import Tongyi
from langchain_community.chat_models.tongyi import ChatTongyi



load_dotenv()

# 状态定义
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    news_data: List[Dict[str, Any]]
    factor_data: List[Dict[str, Any]]

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

# Tools
@tool
async def crawl_and_save_news(url: str = "https://kuaixun.eastmoney.com/ssgs.html") -> List[Dict[str, Any]]:
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
            fieldnames = ['发布时间', '新闻标题', '新闻内容', '相关公司', '股票代码', '股票简称']
            field_mapping = {
                'news_time': '发布时间',
                'news_title': '新闻标题',
                'news_text': '新闻内容',
                'company_involved': '相关公司',
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

    fieldnames = ['公司名称', '股票代码', '股票简称', '新闻时间', '新闻标题', '影响方向', '新闻摘要']
    field_mapping = {
        'company_name': '公司名称',
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

# 节点函数
async def crawl_node(state: AgentState) -> AgentState:
    """新闻爬取节点"""
    url = "https://kuaixun.eastmoney.com/ssgs.html"
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

# 工作流设置
workflow = StateGraph(
    AgentState,  # 使用我们定义的 AgentState 作为状态模式
)

# 添加节点
workflow.add_node("crawl", crawl_node)
workflow.add_node("analyze", analyze_node)
workflow.add_node("save", save_node)

# 添加边
workflow.add_edge("crawl", "analyze")
workflow.add_edge("analyze", "save")
workflow.add_edge("save", END)

# 设置入口
workflow.set_entry_point("crawl")

# 编译工作流
chain = workflow.compile()

# 运行工作流
async def run_workflow():
    initial_state: AgentState = {
        "messages": [],
        "news_data": [],
        "factor_data": []
    }
    result = await chain.ainvoke(initial_state)
    return result

if __name__ == "__main__":
    asyncio.run(run_workflow()) 