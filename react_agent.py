"""
ReAct (Reasoning + Acting) 智能体实现
通过思考-行动-观察的循环来回答复杂问题

架构说明:
    本实现采用三层工具管理架构:
    1. ToolRegistry: 工具注册表，管理所有工具的元数据和函数
    2. ToolSelector: 工具选择器，根据用户意图选择合适的工具
    3. ToolExecutor: 工具执行器，负责调用工具并处理重试和异常

    ReActAgent 是编排层，协调三者工作:
    - 使用 Selector 选择工具
    - 使用 Executor 执行工具
    - 使用 Registry 获取工具描述

循环流程:
    while 未达到最大步数:
        1. 构建Prompt (含工具描述、历史、错误提示)
        2. 调用LLM获取思考和行动
        3. 解析LLM的JSON输出
        4. 执行工具 (通过Executor)
        5. 记录执行结果到历史
        6. 检查是否达到最终答案 (Finish)
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
from tools.registry import ToolRegistry
from tools.selector import ToolSelector
from tools.executor import ToolExecutor, ExecutionStatus

PROMPT_DIR = Path(__file__).parent / "prompt"


def create_default_registry() -> ToolRegistry:
    """
    创建默认的工具注册表

    预注册工具:
    - Search: 网页搜索工具，分类为 search
    - GetCurrentTime: 时间查询工具，分类为 time
    - Calculator: 计算器工具，分类为 calculate

    每个工具都包含:
    - name: 工具名
    - description: 功能描述
    - func: Python函数
    - category: 业务分类（用于意图路由）
    - tags: 标签（用于关键词检索）
    - parameters: 参数定义
    - examples: 调用示例（供LLM学习）
    """
    registry = ToolRegistry()

    registry.register(
        name="Search",
        description="网页搜索引擎。用于搜索实时信息、新闻、事件等。",
        func=tools.web_search.search,
        category="search",
        tags=["搜索", "查询", "信息", "新闻"],
        parameters={"query": "搜索关键词"},
        examples=["Search[小米手机2026年新机型]", "Search[今天天气]"]
    )

    registry.register(
        name="GetCurrentTime",
        description="获取当前日期和时间。用于回答'现在几点了'、'今天是哪天'等问题。",
        func=tools.time_query.get_current_time,
        category="time",
        tags=["时间", "日期", "现在"],
        parameters={},
        examples=["GetCurrentTime[]"]
    )

    registry.register(
        name="Calculator",
        description="数学计算器。用于计算数学表达式，支持加减乘除、括号、指数等。",
        func=tools.calculator.calculate,
        category="calculate",
        tags=["计算", "数学", "运算"],
        parameters={"expression": "数学表达式"},
        examples=["Calculator[(123+456)*789/12]", "Calculator[2**10]"]
    )

    return registry


class ReActAgent:
    """
    ReAct 智能体

    核心属性:
        llm_client: LLM客户端，用于生成思考和行动
        registry: 工具注册表，提供工具元数据和描述
        selector: 工具选择器，根据意图筛选工具（当前未充分使用）
        executor: 工具执行器，执行工具调用并处理重试
        max_steps: 最大循环步数，防止无限循环
        history: 对话历史，记录Action和Observation
        tool_call_failures: 工具调用失败记录，用于渐进式错误提示

    使用示例:
        registry = create_default_registry()
        selector = ToolSelector(registry)
        executor = ToolExecutor(registry)

        agent = ReActAgent(llm_client, registry, selector, executor, max_steps=5)
        result = agent.run("小米手机最新型号")
    """

    def __init__(self, llm_client: HelloAgentLLM, registry: ToolRegistry,
                 selector: ToolSelector, executor: ToolExecutor, max_steps: int = 5):
        self.llm_client = llm_client
        self.registry = registry
        self.selector = selector
        self.executor = executor
        self.max_steps = max_steps
        self.history = []
        self.tool_call_failures = []

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

            tools_desc = self.registry.describe_all()
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

            # 大小写纠正：如果LLM输出的工具名不存在，尝试大小写不敏感匹配
            # 例如LLM输出"search"但实际注册的是"Search"
            actual_tool_name = self._correct_tool_name(tool_name)
            if actual_tool_name != tool_name:
                print(f"⚠️ 工具名大小写纠正: '{tool_name}' -> '{actual_tool_name}'")
                tool_name = actual_tool_name

            print(f"🎬 行动: {tool_name}[{tool_input}]")

            result = self.executor.execute(tool_name, tool_input)
            observation = result.observation

            if result.status == ExecutionStatus.SUCCESS:
                failure_reason = None
            else:
                failure_reason = f"工具'{tool_name}'执行失败: {result.error}"

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

    def _correct_tool_name(self, tool_name: str) -> str:
        """
        纠正工具名的大小写错误

        当LLM输出的工具名与注册的工具名存在大小写差异时，
        自动纠正为注册的正确名称。

        例如:
            LLM输出: "search" -> 实际注册: "Search" -> 纠正为: "Search"

        Args:
            tool_name: LLM输出的工具名

        Returns:
            纠正后的工具名（如果存在大小写差异）
            原始工具名（如果无需纠正或不存在）
        """
        # 如果工具名直接存在（区分大小写），无需纠正
        if self.registry.get(tool_name):
            return tool_name

        # 尝试大小写不敏感匹配
        # get_all()返回ToolMetadata列表，需访问.name属性
        for metadata in self.registry.get_all():
            if metadata.name.lower() == tool_name.lower():
                return metadata.name

        # 未找到匹配，返回原始名称
        return tool_name

    def _contains_time_keywords(self, text: str) -> bool:
        """
        检查文本是否包含时间相关关键词

        用于判断是否需要自动注入当前时间提示。
        当用户问题包含"最新"、"当前"等词时，需要告知LLM当前实际时间。

        Args:
            text: 用户问题

        Returns:
            True if 包含时间关键词
        """
        time_keywords = ["最新", "当前", "今年", "何时", "现在", "最近"]
        return any(keyword in text for keyword in time_keywords)

    def _inject_current_time(self, question: str) -> str:
        """
        如果问题涉及时间关键词，自动注入当前时间到prompt

        当检测到用户问题涉及时间时（如"最新"），调用time_query工具
        获取当前时间，并将时间信息注入prompt，让LLM基于正确的时间回答。

        Args:
            question: 用户问题

        Returns:
            时间提示字符串，如"【系统提示】当前时间是 2026年04月26日..."
            如果不涉及时间，返回空字符串
        """
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

        干预策略:
        - 1次失败: 提醒检查工具名和参数（轻微提示）
        - 2次失败: 列出失败的工具和可用工具清单（帮助纠正）
        - 3次及以上:
            - 同一工具反复失败 → 明确要求停止调用该工具
            - 不同工具都失败 → 警告即将终止任务

        设计考虑:
        - 只看最近2次失败（更早的失败已不重要）
        - 区分"重复调用同一工具"和"调用不同工具都失败"两种情况
        - 渐进式干预：给予LLM自我纠正的机会，而非一开始就终止

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
            tool_names = [t.name for t in self.registry.get_all()]
            hints.append(f"可用工具列表: {tool_names}")

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

        设计考虑:
        - 为什么要检查"重复"？因为LLM可能会固执地重复调用同一失败的工具
        - 为什么要只看2次？因为更早的失败模式已不重要，重要的是"最近在重复"

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

    registry = create_default_registry()
    selector = ToolSelector(registry)
    executor = ToolExecutor(registry, max_retries=2)

    agent = ReActAgent(llm_agent, registry, selector, executor, 5)
    query = "小米手机新机型"
    result = agent.run(query)
    print(result)
