"""
Planner-Executor 模式的智能体实现
将复杂问题分解为计划步骤，然后逐步执行
"""

import ast
import re
from pathlib import Path
from typing import List

from hello_agents import HelloAgentLLM

# 基于脚本位置获取 prompt 目录，避免硬编码路径
PROMPT_DIR = Path(__file__).parent / "prompt"


class Planner:
    """规划器：根据用户问题生成行动计划"""

    def __init__(self, llm_client: HelloAgentLLM):
        self.llm_client = llm_client

    def plan(self, question: str) -> List[str]:
        """根据用户提问生成行动计划"""
        prompt_path = PROMPT_DIR / "planner_prompt.txt"
        with open(prompt_path, "r") as f:
            prompt = f.read()
        prompt = prompt.format(question=question)

        messages = [
            {"role": "user", "content": prompt}
        ]

        print("---正在生成计划---")

        # 使用流式输出来获取计划内容
        response_text = self.llm_client.think(messages) or ''

        print(f"✅计划已生成: {response_text}")

        # 解析 LLM 输出的字符串（使用正则增强容错性）
        try:
            match = re.search(r"```python\s*(.*?)\s*```", response_text, re.DOTALL)
            if match:
                plan_str = match.group(1).strip()
            else:
                # Fallback: 尝试直接解析
                plan_str = response_text.strip()
            plan = ast.literal_eval(plan_str)
            return plan if isinstance(plan, list) else []
        except (ValueError, SyntaxError, IndexError) as e:
            print(f"❌ 解析计划时出错: {e}")
            print(f"原始响应: {response_text}")
            return []
        except Exception as e:
            print(f"❌ 解析计划时发生未知错误: {e}")
            return []


class Executor:
    """执行器：按照计划逐步执行任务"""

    def __init__(self, llm_client: HelloAgentLLM):
        """初始化执行器，绑定 LLM 客户端"""
        self.llm_client = llm_client

    def execute(self, question: str, plan: List[str]) -> str:
        """执行行动计划"""
        history = ""

        print("\n---正在执行计划---")

        prompt_path = PROMPT_DIR / "executor_prompt.txt"
        with open(prompt_path, "r") as f:
            origin_prompt = f.read()

        for i, step in enumerate(plan):
            prompt = origin_prompt.format(question=question, plan=plan, history=history, current_step=step)

            messages = [
                {"role": "user", "content": prompt}
            ]

            response_text = self.llm_client.think(messages) or ''

            history += f"步骤 {i+1}: {step}\n结果: {response_text}\n\n"

            print(f"步骤 {i+1}: {step} ------> 已完成，结果: {response_text}")

        # 循环结束后，最后一步的响应就是答案
        final_answer = response_text
        return final_answer


class PlanAndSolveAgent:
    """Plan-and-Solve 智能体：先规划后执行"""

    def __init__(self, llm_client: HelloAgentLLM):
        """初始化智能体，同时创建 Planner 和 Executor"""
        self.llm_client = llm_client
        self.planner = Planner(llm_client)
        self.executor = Executor(llm_client)

    def run(self, question: str):
        """运行智能体，根据用户提问生成行动计划并执行"""
        print(f"\n--- 开始处理问题 ---\n问题: {question}")

        # 1.调用规划器生成计划
        plan = self.planner.plan(question)
        if not plan:
            print("\n--- 任务终止 --- \n无法生成有效的行动计划。")
            return

        # 2.调用执行器执行计划
        final_answer = self.executor.execute(question, plan)

        print(f"\n--- 任务完成 ---\n最终答案: {final_answer}")


if __name__ == "__main__":
    llm_client = HelloAgentLLM()
    agent = PlanAndSolveAgent(llm_client)
    agent.run("一个水果店周一卖出了15个苹果。周二卖出的苹果数量是周一的两倍。周三卖出的数量比周二少了5个。请问这三天总共卖出了多少个苹果？")
