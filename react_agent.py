"""
ReAct (Reasoning + Acting) 智能体实现
通过思考-行动-观察的循环来回答复杂问题
"""

import re
from pathlib import Path

import tools.web_search
from hello_agents import HelloAgentLLM
from tool_executor import ToolExecutor

# 基于脚本位置获取 prompt 目录
PROMPT_DIR = Path(__file__).parent / "prompt"


class ReActAgent:
    """
    ReAct 智能体：结合推理与行动的问答智能体
    通过 Thought -> Action -> Observation 循环，直到得到最终答案
    """

    def __init__(self, llm_client: HelloAgentLLM, tool_executor: ToolExecutor, max_steps: int = 5):
        """初始化 ReAct 智能体

        Args:
            llm_client: LLM 客户端
            tool_executor: 工具执行器
            max_steps: 最大循环步数，防止无限循环
        """
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []  # 存储思考过程中的 Action 和 Observation


    def run(self, question: str):
        """
        运行ReAct智能体来回答一个问题。
        """
        self.history = [] # 每次运行时重置历史记录
        current_step = 0

        while current_step < self.max_steps:
            current_step += 1
            print(f"--- 第 {current_step} 步 ---")

            # 1.格式化提示词
            tools_desc = self.tool_executor.getAvailableTools()
            history_str = "\n".join(self.history)

            prompt_path = PROMPT_DIR / "react_prompt.txt"
            prompt = open(prompt_path, "r").read()
            prompt = prompt.format(
                tools=tools_desc,
                question=question,
                history=history_str
            )

            # 2.调用LLM进行思考
            messages = [
                {"role": "user", "content": prompt}
            ]
            response_text = self.llm_client.think(messages)

            if not response_text:
                print("错误:LLM未能返回有效响应。")
                break

            # 3. 解析LLM的输出
            thought, action = self._parse_output(response_text)

            if thought:
                print(f"思考：{thought}")

            if not action:
                print("警告:未能解析出有效的Action，流程终止。")
                break

            # 4. 执行Action
            if action.startswith("Finish"):
                # 如果是Finish指令，提取最终答案并结束
                final_answer = re.match(r"Finish\[(.*)\]", action, re.DOTALL).group(1)
                print(f"🎉 最终答案: {final_answer}")
                return final_answer

            tool_name, tool_input = self._parse_action(action)
            if not tool_name or not tool_input:
                # ... 处理无效Action格式 ...
                print("无效的Tool name & desc")
                continue

            print(f"🎬 行动: {tool_name}[{tool_input}]")
            tool_func = self.tool_executor.getTool(tool_name)
            if not tool_func:
                observation = f"错误:未找到名为 '{tool_name}' 的工具。"
            else:
                observation = tool_func(tool_input)  # 调用真实工具

            print(f"👀 观察: {observation}\n")

            # 将本轮的Action和Observation添加到历史记录中
            self.history.append(f"Action:{action}")
            self.history.append(f"Observation:{observation}")

        # 循环结束
        print("已达到最大步数，流程终止。")
        return None

    def _parse_output(self, text: str):
        """解析LLM的输出，提取Thought和Action。
        """
        # Thought: 匹配到 Action: 或文本末尾
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL)
        # Action: 匹配到文本末尾
        action_match = re.search(r"Action:\s*(.*?)$", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    def _parse_action(self, action_text: str):
        """解析Action字符串，提取工具名称和输入。
        """
        match = re.match(r"(\w+)\[(.*)\]", action_text, re.DOTALL)
        if match:
            return match.group(1), match.group(2)
        return None, None


if __name__ == "__main__":
    llm_agent = HelloAgentLLM()
    tool_executor = ToolExecutor()
    search_description = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    tool_executor.registerTool("Search",search_description,tools.web_search.search)
    agent = ReActAgent(llm_agent, tool_executor, 5)
    query = "小米手机26年新机型"
    result = agent.run(query)
    print(result)
