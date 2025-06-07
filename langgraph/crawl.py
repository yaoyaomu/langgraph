# from crawl4ai import AsyncWebCrawler
# import asyncio

# async def simple_crawl():
#     async with AsyncWebCrawler(verbose=True) as crawler:
#         result = await crawler.arun(url="https://finance.eastmoney.com/a/202506053422904053.html")
#         print(result.markdown[:500])  

# async def main():
#     await simple_crawl()

# if __name__ == "__main__":
#     asyncio.run(main())

# from crawl4ai import AsyncWebCrawler
# import asyncio
# from bs4 import BeautifulSoup

# async def simple_crawl():
#     async with AsyncWebCrawler(verbose=True) as crawler:
#         result = await crawler.arun(url="https://finance.eastmoney.com/a/202506053422904053.html")
#         # 手动解析 HTML，提取所有 <p> 标签内容
#         soup = BeautifulSoup(result.html, "html.parser")
#         paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
#         content = "\n".join(paragraphs)
#         print(content[:1000])  # 打印前1000字符

# async def main():
#     await simple_crawl()

# if __name__ == "__main__":
#     asyncio.run(main())

import asyncio  
from crawl4ai import AsyncWebCrawler  
  
async def main():  
    async with AsyncWebCrawler() as crawler:  
        # 直接使用 arun，不需要手动调用 awarmup  
        result = await crawler.arun(url="https://kuaixun.eastmoney.com/ssgs.html")  
        print(result.markdown)  
  
if __name__ == "__main__":  
    asyncio.run(main())

# 成功调用crawl.py
# 原来的库 

# import asyncio  
# from crawl4ai import AsyncWebCrawler  
# from bs4 import BeautifulSoup

# async def main():  
#     async with AsyncWebCrawler() as crawler:  
#         result = await crawler.arun(url="https://kuaixun.eastmoney.com/ssgs.html")  
#         soup = BeautifulSoup(result.html, "html.parser")
#         news_items = soup.find_all(class_="news_item")
#         for item in news_items:
#             # 提取时间
#             time_tag = item.find(class_="news_time")
#             news_time = time_tag.get_text(strip=True) if time_tag else ""
#             # 提取正文/标题
#             detail_tag = item.find(class_="news_detail_text")
#             news_text = detail_tag.get_text(strip=True) if detail_tag else ""
#             # 提取公司名
#             stock_name_tag = item.find(class_="stock_name")
#             company_name = stock_name_tag.get_text(strip=True) if stock_name_tag else ""
#             # 提取股票代码
#             stock_item_tag = item.find("div", class_="stock_item", attrs={"data-show": "true"})
#             stock_code = ""
#             if stock_item_tag and stock_item_tag.has_attr("data-mc"):
#                 code = stock_item_tag["data-mc"]
#                 # 代码一般为 1.601992，取点后面的部分
#                 stock_code = code.split(".")[-1]
#             print(f"时间: {news_time}\n标题/正文: {news_text}\n公司: {company_name}\n股票代码: {stock_code}\n{'-'*40}")

#         print(f"共提取到 {len(news_items)} 条新闻。")

# if __name__ == "__main__":  
#     asyncio.run(main())