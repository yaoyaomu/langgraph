import os
import asyncio
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from crawl4ai.async_configs import LLMConfig
from langchain_core.tools import tool
import csv
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

# Pydantic models for data validation
class NewsURL(BaseModel):
    url: str = Field(..., description="提取新闻发布的链接,只返回链接,不要返回其他内容.")

class NewsContent(BaseModel):
    news_time: str = Field(..., description="新闻发布的时间,包括日期、时间,如 2023年10月5日 14:30.")
    news_title: str = Field(..., description="新闻原文的完整标题，禁止修改或缩写.")
    news_text: str = Field(..., description="新闻的全部正文内容.")
    company_involved: str = Field(..., description="新闻中所有关联的上市公司都要列出，多条公司多条记录，使用其法定注册全称.")
    stock_code: str = Field(..., description="与涉及公司对应的唯一官方证券代码.")
    stock_short_name: str = Field(..., description="与涉及公司对应的交易所标准化简称.")

# # API密钥配置
# DEEPSEEK_API_KEY = "sk-b647e4f60dd64587ad6f33db59d3ee6e"

@tool("crawl_news")
async def crawl_news(url: str = "https://kuaixun.eastmoney.com/ssgs.html") -> List[Dict[str, Any]]:
    """
    使用 crawl4ai 从给定 URL 抓取新闻并提取内容。
    
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
                        api_token=DEEPSEEK_API_KEY,
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
            return []
            
        content = json.loads(links_result.extracted_content) if isinstance(links_result.extracted_content, str) else links_result.extracted_content
        news_links = [item['url'] for item in content if isinstance(item, dict) and item.get('url')]
        
        # Second layer: Extract news content
        all_news = []
        for news_url in news_links:
            content_result = await crawler.arun(
                url=news_url,
                config=CrawlerRunConfig(
                    word_count_threshold=1,
                    extraction_strategy=LLMExtractionStrategy(
                        llm_config=LLMConfig(
                            provider="deepseek/deepseek-chat",
                            api_token=DEEPSEEK_API_KEY,
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
                                - 返回格式必须是JSON数组
                    """
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
                    
        return all_news

@tool("save_news")
def save_news(data: List[Dict[str, Any]], filename: str = "news.csv") -> str:
    """
    将提取的新闻数据保存为 CSV 文件。
    
    参数
        data： 包含新闻数据的词典列表
        filename：输出 CSV 文件的名称
        
    返回值
        显示成功或失败的状态信息
    """
    if not data:
        return "No data to save"

    fieldnames = ['news_time', 'news_title', 'news_text', 'company_involved', 'stock_code', 'stock_short_name']
    
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for item in data:
                row = {k: item.get(k, "") for k in fieldnames}
                writer.writerow(row)
                
        return f"Successfully saved {len(data)} records to {filename}"
    except Exception as e:
        return f"Error saving to CSV: {str(e)}"






# Example usage in a langgraph workflow:
"""
from langgraph.graph import StateGraph, END

# Define your graph
workflow = StateGraph()

# Add nodes for the tools
workflow.add_node("crawl", crawl_news)
workflow.add_node("save", save_news)

# Add edges
workflow.add_edge("crawl", "save")
workflow.add_edge("save", END)

# Set the entry point
workflow.set_entry_point("crawl")

# Create the runnable
chain = workflow.compile()

# Run the workflow
async def run_workflow():
    result = await chain.ainvoke({
        "url": "https://kuaixun.eastmoney.com/ssgs.html",
        "filename": "news.csv"
    })
    return result

if __name__ == "__main__":
    asyncio.run(run_workflow())
""" 