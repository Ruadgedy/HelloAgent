"""
工具执行器模块
管理和执行各种可调用工具的注册与调用
"""

from typing import Dict, Any, Callable

from tools.web_search import search



class ToolExecutor:
    """
    一个工具执行器，负责管理和执行工具。
    """
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}

    def registerTool(self, name: str, description: str, func: Callable):
        """
        向工具箱中注册一个新工具。
        """
        if name in self.tools:
            print(f"警告:工具 '{name}' 已存在，将被覆盖。")

        self.tools[name] = {"description": description, "func": func}
        print(f"工具 '{name}' 已注册。")

    def getTool(self, name: str) -> Callable:
        """根据名称获取一个工具的执行函数，不存在时返回 None"""
        if name not in self.tools:
            return None
        return self.tools[name]["func"]

    def getAvailableTools(self) -> str:
        """获取所有可用工具的格式化描述字符串"""
        return "\n".join([
            f"- {name}: {info['description']}"
            for name, info in self.tools.items()
        ])


if __name__ == "__main__":
    # 1. 初始化工具执行器
    tool_executor = ToolExecutor()

    # 2. 注册我们的实战搜索工具
    search_description = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    tool_executor.registerTool("Search", search_description, search)

    # 3. 打印可用的工具
    print("\n--- 可用的工具 ---")
    print(tool_executor .getAvailableTools())

    # 4. 智能体的Action调用，这次我们问一个实时性的问题
    tool_name = "Search"
    tool_input = "什么是AI本体"
    print(f"\n--- 执行 Action: Search['{tool_input}'] ---")

    tool_function = tool_executor.getTool(tool_name)
    if tool_function:
        observation = tool_function(tool_input)
        print("--- 观察 (Observation) ---")
        print(observation)
    else:
        print(f"错误:未找到名为 '{tool_name}' 的工具。")