import os    
import asyncio      
import json  
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode    
from crawl4ai.extraction_strategy import LLMExtractionStrategy    
from crawl4ai.async_configs import LLMConfig    
from pydantic import BaseModel, Field    
import csv  


class NewsURL(BaseModel):    
    url: str = Field(..., description="提取新闻发布的链接,只返回链接,不要返回其他内容.")  

class NewsContent(BaseModel):    
    news_time: str = Field(..., description="新闻发布的时间,包括日期、时间,如 2023年10月5日 14:30.")    
    news_title: str = Field(..., description="新闻原文的完整标题，禁止修改或缩写.")    
    news_text: str = Field(..., description="新闻的全部正文内容.")    
    company_involved: str = Field(..., description="新闻中所有关联的上市公司都要列出，多条公司多条记录，使用其法定注册全称​.")    
    stock_code: str = Field(..., description="与涉及公司对应的唯一官方证券代码.")    
    stock_short_name: str = Field(..., description="与涉及公司对应的交易所标准化简称.")    
  
# API密钥配置  
DEEPSEEK_API_KEY = "sk-b647e4f60dd64587ad6f33db59d3ee6e"

url = "https://kuaixun.eastmoney.com/ssgs.html"    
  
def save_to_csv(data, filename="news.csv"):
    """将提取的数据保存为CSV文件"""
    if not data:
        print("没有数据可保存")
        return

    # CSV字段顺序
    fieldnames = ['news_time', 'news_title', 'news_text', 'company_involved', 'stock_code', 'stock_short_name']

    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            # 如果data是单个对象，转换为列表
            if isinstance(data, dict):
                data = [data]

            for item in data:
                # 只保留需要的字段，忽略error字段
                row = {k: item.get(k, "") for k in fieldnames}
                writer.writerow(row)

        print(f"数据已成功保存到 {filename}")
        print(f"共保存了 {len(data)} 条记录")
    except Exception as e:
        print(f"保存CSV文件时出错: {e}")

async def extract_news_links(crawler, base_url):
    """第一层爬取：提取新闻链接"""
    print("开始第一层爬取：提取新闻链接...")
    
    try:
        result = await crawler.arun(
            url=base_url,
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
        
        if result.success and result.extracted_content:
            content = result.extracted_content
            if isinstance(content, str):
                content = json.loads(content)
            
            if not isinstance(content, list):
                print("提取的内容格式不正确，期望是列表格式")
                return []
            
            links = []
            for item in content:
                if isinstance(item, dict) and item.get('url'):
                    url = item['url']
                    if url.startswith('//'):
                        url = 'https:' + url
                    if '/a/' in url:
                        links.append({
                            'url': url,
                            'title': ''  # 标题将在第二层爬取时获取
                        })
            
            print(f"找到 {len(links)} 个新闻链接")
            if len(links) > 0:
                print("示例链接：")
                for i, link in enumerate(links[:3], 1):
                    print(f"{i}. {link['url']}")
            return links
            
        print("提取失败：", result.error_message if hasattr(result, 'error_message') else "未知错误")
        return []
    except Exception as e:
        print(f"提取新闻链接时出错: {e}")
        if 'result' in locals():
            print("原始响应内容：", result.extracted_content if result.success else "无内容")
        return []

async def extract_news_content(crawler, news_url):
    """从新闻详情页提取内容"""
    try:
        result = await crawler.arun(
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
                    instruction="""请从新闻页面提取以下信息：
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
        
        if result.success and result.extracted_content:
            try:
                # 尝试解析JSON字符串
                if isinstance(result.extracted_content, str):
                    content = json.loads(result.extracted_content)
                else:
                    content = result.extracted_content
                
                # 确保内容是列表格式
                if not isinstance(content, list):
                    content = [content]
                
                # 验证每个记录是否包含所需字段
                valid_records = []
                required_fields = {'news_time', 'news_title', 'news_text', 'company_involved', 'stock_code', 'stock_short_name'}
                
                for record in content:
                    if isinstance(record, dict) and all(field in record for field in required_fields):
                        valid_records.append(record)
                
                return valid_records if valid_records else None
                
            except json.JSONDecodeError as e:
                print(f"JSON解析错误: {e}")
                return None
            except Exception as e:
                print(f"数据处理错误: {e}")
                return None
        else:
            print("提取失败：", result.error_message if hasattr(result, 'error_message') else "未知错误")
            return None
    except Exception as e:
        print(f"提取新闻内容时出错 {news_url}: {e}")
        return None

async def main():      
    async with AsyncWebCrawler() as crawler:      
        try:
            # 第一层：提取新闻链接
            news_links = await extract_news_links(crawler, url)
            
            if not news_links:
                print("未找到任何新闻链接")
                return
                
            print(f"\n开始第二层爬取：提取新闻详细内容...")
            
            # 第二层：提取每个URL的新闻内容
            all_news = []
            for i, link in enumerate(news_links, 1):
                news_url = link['url']
                print(f"\n处理第 {i}/{len(news_links)} 条新闻:")
                print(f"URL: {news_url}")
                
                news_content = await extract_news_content(crawler, news_url)
                if news_content:
                    if isinstance(news_content, list):
                        all_news.extend(news_content)
                    else:
                        all_news.append(news_content)
                    print("√ 提取成功")
                    print(news_content)
                else:
                    print("× 提取失败")
                    print(news_content)

            # 保存结果
            if all_news:
                save_to_csv(all_news)
                print(f"\n成功提取并保存了 {len(all_news)} 条新闻记录")
            else:
                print("\n没有成功提取到任何新闻内容")
                
        except Exception as e:  
            print(f"爬取过程中出现错误: {e}")  

if __name__ == "__main__":
    asyncio.run(main())  