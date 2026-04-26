"""
ReAct (Reasoning + Acting) 智能体实现
通过思考-行动-观察的循环来回答复杂问题
"""

import json
import re
from pathlib import Path
from string import Template
from datetime import datetime
from typing import Optional

import tools.web_search
import tools.time_query
import tools.calculator
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
        self.tool_call_failures = [] # 记录工具调用每次失败信息

    def run(self, question: str):
        self.history = []
        self.tool_call_failures = []
        current_step = 0
        MAX_CONSECUTIVE_FAILURES = 3

        while current_step < self.max_steps:
            if len(self.tool_call_failures) >= MAX_CONSECUTIVE_FAILURES:
                last_failure = self.tool_call_failures[-1]["reason"]
                print(f"已达到最大失败次数限制 ({MAX_CONSECUTIVE_FAILURES})，任务终止。最后错误: {last_failure}")
                return None

            current_step += 1
            print(f"--- 第 {current_step} 步 ---")

            tools_desc = self.tool_executor.getAvailableTools()
            history_str = "\n".join(self.history)
            error_hint = self._build_error_hint()

            time_hint = self._inject_current_time(question)

            prompt_path = PROMPT_DIR / "react_prompt.txt"
            prompt_template = Template(open(prompt_path, "r").read())
            prompt = prompt_template.safe_substitute(
                tools=tools_desc,
                question=question,
                history=history_str,
                time_hint=time_hint,
                error_hint=error_hint
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

            observation = ""
            failure_reason = None

            if tool_name == "GetCurrentTime":
                observation = tools.time_query.get_current_time(tool_input)
            else:
                tool_func = self.tool_executor.getTool(tool_name)
                if not tool_func:
                    available = self.tool_executor.getAvailableToolNames()
                    observation = f"错误:未找到名为 '{tool_name}' 的工具。可用工具: {available}"
                    failure_reason = f"工具'{tool_name}'不存在"
                elif not tool_input:
                    observation = f"错误:工具 '{tool_name}' 的参数不能为空"
                    failure_reason = f"工具'{tool_name}'参数为空"
                else:
                    try:
                        observation = tool_func(tool_input)
                    except Exception as e:
                        observation = f"错误:工具 '{tool_name}' 执行失败，原因: {e}"
                        failure_reason = f"工具'{tool_name}'执行异常"

            # 失败原因只可能有：工具不存在、参数为空、执行异常
            if failure_reason:
                self.tool_call_failures.append({
                    "step": current_step,
                    "tool": tool_name,
                    "reason": failure_reason
                })

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

    def _build_error_hint(self) -> str:
        """
        根据失败历史构建错误提示，实现渐进式干预。

        干预策略：
        - 1次失败：仅提醒检查工具名和参数
        - 2次失败：列出具体失败的工具，并给出可用工具列表
        - 3次及以上：
            - 如果是同一工具反复失败（如Search调用3次都报错），明确指出并要求停止
            - 否则提示即将终止任务

        Returns:
            格式化的错误提示字符串，供注入到prompt中
        """
        if not self.tool_call_failures:
            return ""

        hints = []
        # 只看最近2次失败（更早的失败已不重要）
        recent_failures = self.tool_call_failures[-2:]

        # 第1次失败：轻微提醒
        if len(self.tool_call_failures) == 1:
            hints.append("【注意】上次工具调用失败了，请检查工具名称和参数是否正确。")

        # 第2次失败：给出可用工具列表，帮助纠正
        elif len(self.tool_call_failures) == 2:
            failed_tools = [f["tool"] for f in recent_failures]
            hints.append(f"【警告】连续2次工具调用失败 ({', '.join(failed_tools)})。")
            hints.append(f"可用工具列表: {self.tool_executor.getAvailableToolNames()}")

        # 第3次及以上：严重警告，区分重复调用同一工具 vs 调用不同工具都失败
        else:
            # 检查最近2次失败是否都在调用同一个工具
            repeated = self._find_repeated_failures()
            if repeated:
                # 重复调用同一工具 → 明确要求停止
                hints.append(f"【严重】工具'{repeated}'已被连续调用{len(recent_failures)}次但均失败，请停止调用该工具。")
            else:
                # 调用不同工具都失败 → 即将终止
                hints.append(f"【严重】已连续失败{len(recent_failures)}次，任务即将终止。")

        return "\n".join(hints)

    def _find_repeated_failures(self) -> Optional[str]:
        """
        查找连续失败中是否重复调用了同一个工具。

        仅检查最近2次失败记录。如果这2次都在调用同一个工具，
        说明LLM在反复尝试一个错误的工具，需要特别提醒。

        Returns:
            如果2次失败都是同一工具，返回工具名；否则返回None
        """
        # 失败记录少于2条时，无法判断"重复"
        if len(self.tool_call_failures) < 2:
            return None

        # 取出最近2条失败记录
        recent = self.tool_call_failures[-2:]
        tools = [f["tool"] for f in recent]

        # 比较第一次和第二次的tool名是否相同
        return tools[0] if tools[0] == tools[1] else None

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
    # 注册搜索工具
    search_description = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    tool_executor.registerTool("Search", search_description, tools.web_search.search)
    # 注册时间查询工具
    time_description = "获取当前日期和时间。当你需要知道'现在'是几号、几月、几年时调用此工具。"
    tool_executor.registerTool("GetCurrentTime", time_description, tools.time_query.get_current_time)
    calc_description = "计算器工具，用于执行数学运算。支持加减乘除、括号、指数等标准表达式，如 '(123+456)*789/12'。"
    tool_executor.registerTool("Calculator", calc_description, tools.calculator.calculate)

    agent = ReActAgent(llm_agent, tool_executor, 5)
    # query = "小米手机新机型"
    query = "计算 (321+456)*789/0"
    result = agent.run(query)
    print(result)
