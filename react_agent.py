"""
ReAct (Reasoning + Acting) 智能体实现
通过思考-行动-观察的循环来回答复杂问题
"""

import json
import re
from pathlib import Path
from string import Template
from typing import Optional

import tools.web_search
import tools.time_query
from hello_agents import HelloAgentLLM
from tool_executor import ToolExecutor

PROMPT_DIR = Path(__file__).parent / "prompt"


class ReActAgent:
    """
    ReAct 智能体：结合推理与行动的问答智能体
    通过 Thought -> Action -> Observation 循环，直到得到最终答案
    """

    def __init__(self, llm_client: HelloAgentLLM, tool_executor: ToolExecutor, max_steps: int = 5):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []

    def run(self, question: str):
        self.history = []
        current_step = 0

        while current_step < self.max_steps:
            current_step += 1
            print(f"--- 第 {current_step} 步 ---")

            tools_desc = self.tool_executor.getAvailableTools()
            history_str = "\n".join(self.history)

            time_hint = self._inject_current_time(question)

            prompt_path = PROMPT_DIR / "react_prompt.txt"
            prompt_template = Template(open(prompt_path, "r").read())
            prompt = prompt_template.safe_substitute(
                tools=tools_desc,
                question=question,
                history=history_str,
                time_hint=time_hint
            )

            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.think(messages)

            if not response_text:
                print("错误:LLM未能返回有效响应。")
                break

            parsed = self._parse_json_output(response_text)
            if not parsed:
                print("错误:LLM未能返回有效响应。")
                break

            thought = parsed.get("thought")
            action_type = parsed.get("action_type")

            if thought:
                print(f"思考：{thought}")

            if action_type == "finish":
                final_answer = parsed.get("final_answer", "")
                print(f"🎉 最终答案: {final_answer}")
                return final_answer

            if action_type != "tool":
                print(f"警告:未知的action_type: {action_type}，流程终止。")
                break

            tool_name = parsed.get("tool_name")
            tool_input = parsed.get("tool_input", "")
            if isinstance(tool_input, dict):
                tool_input = tool_input.get("query", "")

            if not tool_name:
                print("无效的Tool name")
                continue

            print(f"🎬 行动: {tool_name}[{tool_input}]")

            if tool_name == "GetCurrentTime":
                observation = tools.time_query.get_current_time(tool_input)
            else:
                tool_func = self.tool_executor.getTool(tool_name)
                if not tool_func:
                    observation = f"错误:未找到名为 '{tool_name}' 的工具。"
                else:
                    observation = tool_func(tool_input)

            print(f"👀 观察: {observation}\n")

            self.history.append(f"Action:{action_type}:{tool_name}[{tool_input}]")
            self.history.append(f"Observation:{observation}")

        print("已达到最大步数，流程终止。")
        return None

    def _contains_time_keywords(self, text: str) -> bool:
        time_keywords = ["最新", "当前", "今年", "何时", "现在", "最近", "最近"]
        return any(keyword in text for keyword in time_keywords)

    def _inject_current_time(self, question: str) -> str:
        if not self._contains_time_keywords(question):
            return ""
        try:
            current_time = tools.time_query.get_current_time()
            return f"【系统提示】当前时间是 {current_time}，请以此为基准计算日期。"
        except Exception:
            return ""

    def _parse_json_output(self, text: str) -> Optional[dict]:
        try:
            match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
            json_str = match.group(1) if match else text

            data = json.loads(json_str)

            required_fields = ["thought", "action_type"]
            if not all(field in data for field in required_fields):
                print(f"错误:JSON输出缺少必需字段。需要: {required_fields}")
                return None

            if data["action_type"] not in ("tool", "finish"):
                print(f"错误:action_type必须为'tool'或'finish'，实际为: {data['action_type']}")
                return None

            if data["action_type"] == "tool" and not data.get("tool_name"):
                print("错误:action_type为'tool'但缺少tool_name")
                return None

            if data["action_type"] == "finish" and not data.get("final_answer"):
                print("错误:action_type为'finish'但缺少final_answer")
                return None

            return data

        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")
            return None
        except Exception as e:
            print(f"解析LLM输出时发生未知错误: {e}")
            return None


if __name__ == "__main__":
    llm_agent = HelloAgentLLM()
    tool_executor = ToolExecutor()
    search_description = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    tool_executor.registerTool("Search", search_description, tools.web_search.search)
    time_description = "获取当前日期和时间。当你需要知道'现在'是几号、几月、几年时调用此工具。"
    tool_executor.registerTool("GetCurrentTime", time_description, tools.time_query.get_current_time)
    agent = ReActAgent(llm_agent, tool_executor, 5)
    query = "小米手机新机型"
    result = agent.run(query)
    print(result)
