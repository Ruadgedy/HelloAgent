import os
import time
from typing import List, Dict

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(verbose=True)


class HelloAgentLLM:
    """
    为本书 "Hello Agents" 定制的LLM客户端。
    它用于调用任何兼容OpenAI接口的服务，并默认使用流式响应。
    """
    def __init__(self, model: str = None, apiKey: str = None, baseUrl: str = None, timeout: int = None, max_retries: int = 3):
        """
        初始化客户端。优先使用传入参数，如果未提供，则从环境变量加载。

        Args:
            model: 模型 ID
            apiKey: API 密钥
            baseUrl: 服务地址
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
        """
        self.model = model or os.getenv("LLM_MODEL_ID")
        apiKey = apiKey or os.getenv("LLM_API_KEY")
        baseUrl = baseUrl or os.getenv("LLM_BASE_URL")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", "60"))
        self.max_retries = max_retries

        if not all([self.model, apiKey, baseUrl]):
            raise ValueError("模型ID、API密钥和服务地址必须被提供或在.env文件中定义。")

        self.client = OpenAI(api_key=apiKey, base_url=baseUrl, timeout=timeout)

    def think(self, message: List[Dict[str, str]], temperature: float = 0) -> str:
        """调用大语言模型进行思考，并返回其响应。失败时自动重试。"""
        print(f"🧠 正在调用 {self.model} 模型...")

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=message,
                    temperature=temperature,
                    stream=True,
                    extra_body={"reasoning_split": True}  # 屏蔽<think>标签
                )

                # 处理流式响应
                print("✅ 大语言模型响应成功！")
                collection_content = []
                for chunk in response:
                    content = chunk.choices[0].delta.content or ""
                    collection_content.append(content)
                return "".join(collection_content)

            except Exception as e:
                if attempt == self.max_retries - 1:
                    print(f"❌ 调用LLM API时发生错误: {e}")
                    return None
                wait_time = 2 ** attempt  # 指数退避
                print(f"⚠️ API 调用失败，{wait_time}秒后重试 ({attempt + 1}/{self.max_retries})...")
                time.sleep(wait_time)

        return None

if __name__ == "__main__":
    # 演示：使用 LLM 客户端调用模型
    try:
        agent = HelloAgentLLM()

        # 构造对话消息：系统提示 + 用户问题
        exampleMessages = [
            {"role": "system", "content": "You are a helpful assistant that writes Python code."},
            {"role": "user", "content": "写一个快速排序算法"}
        ]

        print("--- 调用LLM ---")
        responseText = agent.think(exampleMessages)
        if responseText:
            print("\n\n--- 完整模型响应 ---")
            print(responseText)

    except ValueError as e:
        print(e)