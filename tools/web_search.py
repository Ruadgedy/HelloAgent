import os
import re
from datetime import datetime

from dotenv import load_dotenv
from serpapi import SerpApiClient

load_dotenv(verbose=True)

def _rewrite_query_with_time(query: str) -> str:
    """
    将query中的时间相关词汇替换为具体日期。
    例如：【最新、最近、近期】 -> '2026年'
    """
    now = datetime.now()
    current_year = now.strftime("%Y年")
    current_year_short = now.strftime("%Y")
    current_year_2digit = now.strftime("%y")

    replacements = [
        (r"最新", current_year),
        (r"近期", current_year),
        (r"最近", current_year),
        (r"今年", current_year),
    ]

    rewritten = query
    for pattern, replacement in replacements:
        rewritten = re.sub(pattern, replacement, rewritten)

    return rewritten

def search(query: str, recency_days: int = 365) -> str:
    """
    一个基于SerpApi的实战网页搜索引擎工具。
    它会智能地解析搜索结果，优先返回直接答案或知识图谱信息。
    """
    query = _rewrite_query_with_time(query)
    print(f"🔍 正在执行 [SerpApi] 网页搜索: {query}")
    try:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            return "错误:SERPAPI_API_KEY 未在 .env 文件中配置。"

        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "gl": "cn", # 国家代码
            "hl": "zh-CN", # 语言代码
        }

        client = SerpApiClient(params)
        result = client.get_dict()
        if "answer_box_list" in result:
            return "\n".join(result["answer_box_list"])
        if "answer_box" in result and "answer" in result["answer_box"]:
            return result["answer_box"]["answer"]
        if "knowledge_graph" in result and "description" in result["knowledge_graph"]:
            return result["knowledge_graph"]["description"]
        if "organic_results" in result and result["organic_results"]:
            # 如果没有直接答案，则返回前三个有机结果的摘要
            snippets = [
                f"[{i + 1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                for i, res in enumerate(result["organic_results"][:3])
            ]
            return "\n\n".join(snippets)

        return f"对不起，没有找到关于 '{query}' 的信息。"
    except Exception as e:
        return f"搜索时发生错误: {e}"
