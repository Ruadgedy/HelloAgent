"""
Planner-Executor 模式的智能体实现
将复杂问题分解为计划步骤，然后逐步执行

动态重规划机制:
- 支持在执行过程中检测步骤失败
- 当步骤失败时，自动调用planner重新生成剩余计划
- 限制最大重规划次数，防止无限循环
"""

import ast
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from hello_agents import HelloAgentLLM

PROMPT_DIR = Path(__file__).parent / "prompt"


@dataclass
class ExecutionResult:
    """
    步骤执行结果

    属性:
        success: 是否成功
        step_name: 步骤名称
        observation: 执行观察/结果
        error: 错误信息（失败时）
    """
    success: bool
    step_name: str
    observation: str
    error: Optional[str] = None


@dataclass
class ReplanContext:
    """
    重规划上下文

    用于将执行情况传递给planner，生成新的计划

    属性:
        question: 用户原始问题
        completed_steps: 已完成步骤列表 [(step_name, observation)]
        failed_step: 失败的步骤名称
        failure_reason: 失败原因
        replan_count: 当前重规划次数
    """
    question: str
    completed_steps: List[tuple] = field(default_factory=list)
    failed_step: Optional[str] = None
    failure_reason: Optional[str] = None
    replan_count: int = 0


class Planner:
    """
    规划器：根据用户问题生成行动计划

    核心方法:
        plan(): 生成初始计划
        replan(): 根据执行上下文重新生成计划
    """

    def __init__(self, llm_client: HelloAgentLLM):
        self.llm_client = llm_client

    def plan(self, question: str) -> List[str]:
        """
        根据用户提问生成初始行动计划

        Args:
            question: 用户问题

        Returns:
            步骤列表，如 ["步骤1", "步骤2", "步骤3"]
        """
        prompt_path = PROMPT_DIR / "planner_prompt.txt"
        with open(prompt_path, "r") as f:
            prompt = f.read()
        prompt = prompt.format(question=question)

        messages = [{"role": "user", "content": prompt}]

        print("---正在生成计划---")

        response_text = self.llm_client.think(messages) or ''

        print(f"✅计划已生成: {response_text}")

        return self._parse_plan(response_text)

    def replan(self, context: ReplanContext) -> List[str]:
        """
        根据执行上下文重新生成计划（动态重规划）

        当某个步骤执行失败时，调用此方法生成新的剩余计划。
        注意：此方法只返回新生成的剩余步骤，已完成步骤由调用方管理。

        Args:
            context: 重规划上下文，包含问题、已完成步骤、失败信息等

        Returns:
            新生成的剩余计划列表（不包含已完成步骤）
        """
        prompt_path = PROMPT_DIR / "replanner_prompt.txt"
        with open(prompt_path, "r") as f:
            prompt_template = f.read()

        # 构建已完成步骤的描述
        completed_desc = ""
        if context.completed_steps:
            completed_lines = []
            for i, (step, obs) in enumerate(context.completed_steps, 1):
                completed_lines.append(f"  步骤{i}: {step} -> 结果: {obs[:100]}...")
            completed_desc = "\n".join(completed_lines)
        else:
            completed_desc = "  （无已完成步骤）"

        # 构建失败步骤的描述
        failed_desc = f"  失败步骤: {context.failed_step}"
        failed_desc += f"\n  失败原因: {context.failure_reason}"

        prompt = prompt_template.format(
            question=context.question,
            completed_steps=completed_desc,
            failed_step=failed_desc,
            replan_count=context.replan_count
        )

        messages = [{"role": "user", "content": prompt}]

        print(f"---正在重新生成计划（第{context.replan_count}次重规划）---")

        response_text = self.llm_client.think(messages) or ''

        print(f"✅新计划已生成: {response_text}")

        # 解析新计划
        new_plan = self._parse_plan(response_text)

        # 解析新计划（只返回新生成的剩余计划，不包含已完成步骤）
        # 注意：completed_steps由_execute_with_replanning中的current_plan管理
        # 通过current_plan[1:]逐步移除已完成步骤，避免重复执行
        return new_plan if new_plan else []

    def _parse_plan(self, response_text: str) -> List[str]:
        """
        解析LLM输出，提取计划步骤列表

        优先从代码块中解析，失败则尝试直接解析

        Args:
            response_text: LLM响应文本

        Returns:
            步骤列表，解析失败返回空列表
        """
        try:
            # 尝试从代码块中提取
            match = re.search(r"```python\s*(.*?)\s*```", response_text, re.DOTALL)
            if match:
                plan_str = match.group(1).strip()
            else:
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
    """
    执行器：按照计划逐步执行任务

    支持动态重规划:
    - 执行每个步骤
    - 检测步骤是否失败
    - 失败时通知Agent触发重规划
    """

    def __init__(self, llm_client: HelloAgentLLM):
        self.llm_client = llm_client

    def execute_step(self, question: str, step: str, history: str) -> ExecutionResult:
        """
        执行单个步骤

        Args:
            question: 用户问题
            step: 当前步骤描述
            history: 历史执行记录

        Returns:
            ExecutionResult: 包含成功与否、观察结果、错误信息
        """
        prompt_path = PROMPT_DIR / "executor_prompt.txt"
        with open(prompt_path, "r") as f:
            origin_prompt = f.read()

        prompt = origin_prompt.format(
            question=question,
            plan="",  # 动态规划模式下plan可能不完整
            history=history,
            current_step=step
        )

        messages = [{"role": "user", "content": prompt}]

        try:
            response_text = self.llm_client.think(messages) or ''

            # 简单的失败检测：检查响应中是否包含明显的错误标记
            if self._is_failure_response(response_text):
                return ExecutionResult(
                    success=False,
                    step_name=step,
                    observation=response_text,
                    error="步骤执行结果不符合预期"
                )

            return ExecutionResult(
                success=True,
                step_name=step,
                observation=response_text
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                step_name=step,
                observation="",
                error=str(e)
            )

    def _is_failure_response(self, response: str) -> bool:
        """
        检测响应是否表示失败

        当前策略：检查是否包含明确的失败标记词
        可扩展：LLM自我评估、规则匹配等

        Args:
            response: LLM响应文本

        Returns:
            True if 检测到失败标记
        """
        failure_markers = [
            "无法完成",
            "无法执行",
            "执行失败",
            "无法完成此步骤",
            "步骤无法完成",
            "操作失败",
            "无法完成这个任务",
        ]
        return any(marker in response for marker in failure_markers)


class PlanAndSolveAgent:
    """
    Plan-and-Solve 智能体：先规划后执行

    支持动态重规划机制:
    - 生成初始计划
    - 按计划执行步骤
    - 步骤失败时自动重规划
    - 限制最大重规划次数
    """

    def __init__(self, llm_client: HelloAgentLLM, max_replans: int = 3):
        """
        初始化智能体

        Args:
            llm_client: LLM客户端
            max_replans: 最大重规划次数，防止无限循环
        """
        self.llm_client = llm_client
        self.planner = Planner(llm_client)
        self.executor = Executor(llm_client)
        self.max_replans = max_replans

    def run(self, question: str):
        """
        运行智能体，支持动态重规划

        流程:
            1. 生成初始计划
            2. 执行计划
            3. 如果步骤失败且未超过重规划上限，重新生成剩余计划
            4. 重复直到计划完成或重规划次数用尽
        """
        print(f"\n--- 开始处理问题 ---\n问题: {question}")

        # 1.生成初始计划
        initial_plan = self.planner.plan(question)
        if not initial_plan:
            print("\n--- 任务终止 --- \n无法生成有效的行动计划。")
            return

        # 2.执行计划（带重规划支持）
        final_answer = self._execute_with_replanning(question, initial_plan)

        print(f"\n--- 任务完成 ---\n最终答案: {final_answer}")

    def _execute_with_replanning(self, question: str, initial_plan: List[str]) -> str:
        """
        执行计划，支持动态重规划

        Args:
            question: 用户问题
            initial_plan: 初始计划

        Returns:
            最终答案文本
        """
        current_plan = initial_plan
        completed_steps: List[tuple] = []  # [(step_name, observation)]
        replan_count = 0
        history = ""

        while current_plan:
            current_step = current_plan[0]

            print(f"\n--- 执行步骤: {current_step} ---")

            # 执行当前步骤
            result = self.executor.execute_step(question, current_step, history)

            if result.success:
                # 步骤成功
                completed_steps.append((current_step, result.observation))
                history += f"步骤 {len(completed_steps)}: {current_step}\n结果: {result.observation}\n\n"

                print(f"✅ 步骤完成: {current_step}")
                print(f"   结果: {result.observation[:100]}...")

                # 从计划中移除已完成的步骤
                current_plan = current_plan[1:]

            else:
                # 步骤失败
                replan_count += 1

                if replan_count > self.max_replans:
                    print(f"\n❌ 已超过最大重规划次数（{self.max_replans}），任务终止。")
                    print(f"   失败步骤: {current_step}")
                    print(f"   失败原因: {result.error or result.observation}")
                    return f"任务执行失败：{current_step}无法完成，已重规划{replan_count-1}次仍未能解决。"

                # 构建重规划上下文
                context = ReplanContext(
                    question=question,
                    completed_steps=completed_steps.copy(),
                    failed_step=current_step,
                    failure_reason=result.error or "执行结果不符合预期",
                    replan_count=replan_count
                )

                # 触发重规划
                print(f"\n⚠️ 步骤失败，准备重规划（第{replan_count}次）")
                print(f"   失败步骤: {current_step}")
                print(f"   失败原因: {result.error or '执行结果不符合预期'}")

                new_plan = self.planner.replan(context)

                if not new_plan:
                    print("❌ 重规划失败，无法生成新计划")
                    return f"任务执行失败：步骤'{current_step}'失败，重规划后仍无法生成有效计划。"

                # 更新当前计划（保留已完成步骤）
                current_plan = new_plan
                print(f"✅ 重规划完成，新计划共{len(current_plan)}个步骤（已完成{len(completed_steps)}个）")

        # 所有步骤执行完成，返回最后一步的结果
        if completed_steps:
            return completed_steps[-1][1]
        return ""


if __name__ == "__main__":
    llm_client = HelloAgentLLM()
    agent = PlanAndSolveAgent(llm_client, max_replans=3)
    agent.run("一个水果店周一卖出了15个苹果。周二卖出的苹果数量是周一的两倍。周三卖出的数量比周二少了5个。请问这三天总共卖出了多少个苹果？")
