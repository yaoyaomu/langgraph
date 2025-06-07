# import os  
# import asyncio    
# from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode  
# from crawl4ai.extraction_strategy import LLMExtractionStrategy  
# from crawl4ai.async_configs import LLMConfig  
# from pydantic import BaseModel, Field  
# import csv
  
# class OpenAIModelFee(BaseModel):  
#     news_time: str = Field(..., description="新闻发布的时间,包括日期、时间,如 2023年10月5日 14:30.")  
#     news_title: str = Field(..., description="新闻原文的完整标题，禁止修改或缩写.")  
#     news_text: str = Field(..., description="新闻的全部正文内容.")  
#     company_involved: str = Field(..., description="新闻中所有关联的上市公司都要列出，多条公司多条记录，使用其法定注册全称​.")  
#     stock_code: str = Field(..., description="与涉及公司对应的唯一官方证券代码.")  
#     stock_short_name: str = Field(..., description="与涉及公司对应的交易所标准化简称.")  
  
# DASHSCOPE_API_KEY = "sk-b647e4f60dd64587ad6f33db59d3ee6e"  
# url="https://kuaixun.eastmoney.com/ssgs.html"  
  
# async def main():    
#     async with AsyncWebCrawler() as crawler:    
#         result = await crawler.arun(  
#             url=url,  
#             config=CrawlerRunConfig(  
#                 word_count_threshold=1,  
#                 extraction_strategy=LLMExtractionStrategy(  
#                     llm_config=LLMConfig(  
#                         provider="deepseek/deepseek-chat",  ## 尝试 LiteLLM 支持的格式 
#                         api_token=os.getenv('DASHSCOPE_API_KEY', DASHSCOPE_API_KEY),
                        
#                     ),  
#                     schema=OpenAIModelFee.model_json_schema(),  
#                     extraction_type="schema",  
#                     instruction="""从抓取的内容中，提取抓取与A股上市公司相关的最新财经新闻。对每条新闻，提取以下字段新闻时间、新闻标题、新闻正文、涉及公司（单个公司名）、股票代码（单个公司代码）、股票简称（单个公司简称）。  
#                     如果一条新闻涉及多家公司，则为每个公司分别保存一行，  
#                     其他字段内容相同，将结构化后的新闻数据保存为本地CSV文件（如news.csv），字段顺序如下： 时间, 标题, 正文, 涉及公司, 股票代码, 股票简称。"""  
#                 ),  
#                 cache_mode=CacheMode.BYPASS,  
#             )  
#         )  
#         print(result.extracted_content)  

    
# if __name__ == "__main__":    
#     asyncio.run(main())



import os    
import asyncio      
import json  
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode    
from crawl4ai.extraction_strategy import LLMExtractionStrategy    
from crawl4ai.async_configs import LLMConfig    
from pydantic import BaseModel, Field    
import csv  
  
class OpenAIModelFee(BaseModel):    
    news_time: str = Field(..., description="新闻发布的时间,包括日期、时间,如 2023年10月5日 14:30.")    
    news_title: str = Field(..., description="新闻原文的完整标题，禁止修改或缩写.")    
    news_text: str = Field(..., description="新闻的全部正文内容.")    
    company_involved: str = Field(..., description="新闻中所有关联的上市公司都要列出，多条公司多条记录，使用其法定注册全称​.")    
    stock_code: str = Field(..., description="与涉及公司对应的唯一官方证券代码.")    
    stock_short_name: str = Field(..., description="与涉及公司对应的交易所标准化简称.")    
  
# 修正API密钥配置  
DEEPSEEK_API_KEY = "sk-b647e4f60dd64587ad6f33db59d3ee6e"  # 使用正确的DeepSeek API密钥  
url = "https://finance.eastmoney.com/a/202506053423050771.html"    
  
def save_to_csv(data, filename="news.csv"):
    """将提取的数据保存为CSV文件"""
    if not data:
        print("没有数据可保存")
        return

    # CSV字段顺序
    fieldnames = ['news_time', 'news_title', 'news_text', 'company_involved', 'stock_code', 'stock_short_name']

    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # 如果data是单个对象，转换为列表
        if isinstance(data, dict):
            data = [data]

        for item in data:
            # 只保留fieldnames中的字段
            row = {k: item.get(k, "") for k in fieldnames}
            writer.writerow(row)

    print(f"数据已保存到 {filename}") 
  
async def main():      
    async with AsyncWebCrawler() as crawler:      
        try:  
            result = await crawler.arun(    
                url=url,    
                config=CrawlerRunConfig(    
                    word_count_threshold=1,    
                    extraction_strategy=LLMExtractionStrategy(    
                        llm_config=LLMConfig(    
                            provider="deepseek/deepseek-chat",  
                            api_token=os.getenv('DEEPSEEK_API_KEY', DEEPSEEK_API_KEY),  # 修正API密钥  
                        ),    
                        schema=OpenAIModelFee.model_json_schema(),    
                        extraction_type="schema",    
                        instruction="""从抓取的内容中，提取与A股上市公司相关的最新财经新闻。对每条新闻，提取以下字段：新闻时间、新闻标题、新闻正文、涉及公司（单个公司名）、股票代码（单个公司代码）、股票简称（单个公司简称）。  
                        如果一条新闻涉及多家公司，则为每个公司分别创建一条记录，其他字段内容相同。  
                        返回JSON格式的数据。"""    
                    ),    
                    cache_mode=CacheMode.BYPASS,    
                )    
            )    
              
            if result.success and result.extracted_content:  
                print("提取成功！")  
                print(result.extracted_content)  
                  
                # 解析JSON并保存为CSV  
                try:  
                    extracted_data = json.loads(result.extracted_content)  
                    save_to_csv(extracted_data)  
                except json.JSONDecodeError as e:  
                    print(f"JSON解析错误: {e}")  
                    print("原始提取内容:", result.extracted_content)  
            else:  
                print("提取失败:", result.error_message if hasattr(result, 'error_message') else "未知错误")  
                  
        except Exception as e:  
            print(f"爬取过程中出现错误: {e}")  
  
if __name__ == "__main__":      
    asyncio.run(main())