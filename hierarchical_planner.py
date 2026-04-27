"""
分层规划模块

核心概念:
- Level1Planner: 高层抽象规划器，生成粗粒度的抽象计划
- Level2SubPlanner: 子计划规划器，为每个高层步骤生成详细子计划
- SubPlanCache: 子计划缓存，避免重复生成相同的子计划

执行流程:
    1. Level1Planner 生成抽象计划 ["步骤A", "步骤B", "步骤C"]
    2. 执行到某高层步骤时，Level2SubPlanner 生成对应的子计划
    3. 执行子计划，用结果更新上下文
    4. 继续下一个高层步骤
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

from hello_agents import HelloAgentLLM

PROMPT_DIR = Path(__file__).parent / "prompt"


@dataclass
class HighLevelStep:
    """
    高层抽象步骤

    表示一个需要进一步分解才能执行的粗粒度步骤

    属性:
        id: 步骤编号
        name: 步骤名称，如 "预订机票"
        description: 步骤描述，补充说明
        status: 执行状态，"pending" | "executing" | "completed" | "failed"
        context_updates: 执行后需要更新到全局上下文的字段
    """
    id: int
    name: str
    description: str = ""
    status: str = "pending"
    context_updates: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LowLevelStep:
    """
    低层详细步骤

    表示一个可以直接执行的细粒度步骤

    属性:
        id: 步骤编号
        name: 步骤名称，如 "搜索北京到上海航班"
        high_level_id: 所属的高层步骤ID
        status: 执行状态
        observation: 执行结果
    """
    id: int
    name: str
    high_level_id: int
    status: str = "pending"
    observation: str = ""


class SubPlanCache:
    """
    子计划缓存

    复用机制:
    - 相同名称的高层步骤可以复用相同的子计划
    - 例如: "预订机票" 这种通用步骤，第一次生成后会被缓存
    - 下次遇到相同名称时直接返回缓存的子计划
    """

    def __init__(self):
        self._cache: Dict[str, List[str]] = {}

    def has(self, high_level_name: str) -> bool:
        """
        检查是否存在缓存的子计划

        Args:
            high_level_name: 高层步骤名称

        Returns:
            True if 存在缓存
        """
        return high_level_name in self._cache

    def get(self, high_level_name: str) -> List[str]:
        """
        获取缓存的子计划

        Args:
            high_level_name: 高层步骤名称

        Returns:
            子计划列表，不存在则返回空列表
        """
        return self._cache.get(high_level_name, [])

    def store(self, high_level_name: str, subplan: List[str]):
        """
        存储子计划到缓存

        Args:
            high_level_name: 高层步骤名称
            subplan: 子计划列表
        """
        self._cache[high_level_name] = subplan

    def clear(self):
        """清空缓存"""
        self._cache.clear()


class Level1Planner:
    """
    高层抽象规划器

    负责生成粗粒度的抽象计划，每个抽象步骤需要进一步分解才能执行

    生成结果示例:
        ["确定出行日期", "预订机票", "预订酒店", "预订租车", "安排行程"]
    """

    def __init__(self, llm_client: HelloAgentLLM):
        self.llm_client = llm_client

    def plan(self, question: str) -> List[HighLevelStep]:
        """
        根据问题生成高层抽象计划

        Args:
            question: 用户问题

        Returns:
            HighLevelStep 列表
        """
        prompt_path = PROMPT_DIR / "level1_planner_prompt.txt"
        with open(prompt_path, "r") as f:
            prompt_template = f.read()

        prompt = prompt_template.format(question=question)

        messages = [{"role": "user", "content": prompt}]

        print("--- [L1] 正在生成抽象计划 ---")

        response_text = self.llm_client.think(messages) or ''

        print(f"--- [L1] 抽象计划: {response_text[:100]} ---")

        step_names = self._parse_response(response_text)

        # 转换为 HighLevelStep 对象
        return [
            HighLevelStep(id=i + 1, name=name, description="")
            for i, name in enumerate(step_names)
        ]

    def _parse_response(self, response_text: str) -> List[str]:
        """
        解析LLM响应，提取高层步骤列表

        优先从代码块中解析，失败则尝试直接解析

        Args:
            response_text: LLM响应文本

        Returns:
            步骤名称列表
        """
        import ast
        import re

        try:
            # 尝试从代码块中提取
            match = re.search(r"```python\s*(.*?)\s*```", response_text, re.DOTALL)
            if match:
                plan_str = match.group(1).strip()
            else:
                plan_str = response_text.strip()

            plan = ast.literal_eval(plan_str)
            if isinstance(plan, list):
                return [str(step) for step in plan if step]
            return []

        except Exception as e:
            print(f"❌ 解析抽象计划失败: {e}")
            return []


class Level2SubPlanner:
    """
    子计划规划器

    为每个高层步骤生成详细的子计划

    复用机制:
    - 相同名称的高层步骤会复用缓存的子计划
    - 避免重复调用LLM生成相同的子计划
    """

    def __init__(self, llm_client: HelloAgentLLM):
        self.llm_client = llm_client
        self.cache = SubPlanCache()

    def generate_subplan(self, high_level_step: HighLevelStep,
                         context: Dict[str, Any]) -> List[LowLevelStep]:
        """
        为高层步骤生成详细子计划

        如果缓存中存在该高层步骤的子计划，直接返回缓存版本

        Args:
            high_level_step: 高层步骤
            context: 全局上下文，包含问题、已完成步骤的结果等

        Returns:
            LowLevelStep 列表
        """
        # 检查缓存
        if self.cache.has(high_level_step.name):
            print(f"--- [L2] 复用缓存的子计划: {high_level_step.name} ---")
            cached_names = self.cache.get(high_level_step.name)
            return [
                LowLevelStep(id=i + 1, name=name, high_level_id=high_level_step.id)
                for i, name in enumerate(cached_names)
            ]

        # 生成新的子计划
        prompt_path = PROMPT_DIR / "level2_subplanner_prompt.txt"
        with open(prompt_path, "r") as f:
            prompt_template = f.read()

        # 构建上下文描述
        context_desc = self._format_context(context)

        prompt = prompt_template.format(
            high_level_step=high_level_step.name,
            context=context_desc
        )

        messages = [{"role": "user", "content": prompt}]

        print(f"--- [L2] 为 '{high_level_step.name}' 生成子计划 ---")

        response_text = self.llm_client.think(messages) or ''

        print(f"--- [L2] 子计划: {response_text[:100]} ---")

        step_names = self._parse_response(response_text)

        # 转换为 LowLevelStep 对象
        low_level_steps = [
            LowLevelStep(id=i + 1, name=name, high_level_id=high_level_step.id)
            for i, name in enumerate(step_names)
        ]

        # 缓存子计划
        self.cache.store(high_level_step.name, step_names)

        return low_level_steps

    def _format_context(self, context: Dict[str, Any]) -> str:
        """
        格式化上下文为可读字符串

        Args:
            context: 上下文字典

        Returns:
            格式化后的字符串
        """
        if not context:
            return "（暂无上下文）"

        lines = []
        for key, value in context.items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)

    def _parse_response(self, response_text: str) -> List[str]:
        """
        解析LLM响应，提取子计划步骤列表

        Args:
            response_text: LLM响应文本

        Returns:
            步骤名称列表
        """
        import ast
        import re

        try:
            match = re.search(r"```python\s*(.*?)\s*```", response_text, re.DOTALL)
            if match:
                plan_str = match.group(1).strip()
            else:
                plan_str = response_text.strip()

            plan = ast.literal_eval(plan_str)
            if isinstance(plan, list):
                return [str(step) for step in plan if step]
            return []

        except Exception as e:
            print(f"❌ 解析子计划失败: {e}")
            return []


class HierarchicalExecutor:
    """
    分层执行器

    执行流程:
    1. 按顺序执行高层步骤
    2. 每个高层步骤执行前，先展开为子计划
    3. 执行子计划，用结果更新上下文
    4. 继续下一个高层步骤

    支持两种展开策略:
    - 惰性展开: 执行到某高层步骤时才展开子计划
    - 提前展开: 开始前将所有高层步骤展开为完整计划
    """

    def __init__(self, llm_client: HelloAgentLLM,
                 tool_executor=None):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.level1_planner = Level1Planner(llm_client)
        self.level2_subplanner = Level2SubPlanner(llm_client)

    def execute(self, question: str,
                high_level_steps: List[HighLevelStep],
                context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行分层计划

        Args:
            question: 用户问题
            high_level_steps: 高层步骤列表
            context: 初始上下文

        Returns:
            最终上下文（包含所有执行结果）
        """
        current_context = context.copy()

        for high_level_step in high_level_steps:
            print(f"\n{'='*60}")
            print(f"--- [L1] 执行高层步骤: {high_level_step.name} ---")
            print(f"{'='*60}")

            # 更新高层步骤状态
            high_level_step.status = "executing"

            # 检查是否需要子计划（简单步骤可能不需要）
            if self._needs_subplan(high_level_step):
                # 惰性展开：生成并执行子计划
                result_context = self._execute_with_subplan(high_level_step, current_context)
            else:
                # 直接执行简单步骤
                result_context = self._execute_single(high_level_step, current_context)

            # 更新上下文
            if result_context:
                current_context.update(result_context)

            # 更新高层步骤状态
            high_level_step.status = "completed"

            print(f"\n--- [L1] 步骤 '{high_level_step.name}' 完成 ---")
            print(f"--- [L1] 当前上下文: {list(current_context.keys())} ---")

        return current_context

    def _needs_subplan(self, high_level_step: HighLevelStep) -> bool:
        """
        判断高层步骤是否需要子计划

        简单的高层步骤（如"确认"、"结束"等）可能不需要子计划

        Args:
            high_level_step: 高层步骤

        Returns:
            True if 需要子计划
        """
        # 简单步骤关键词，这些步骤不需要进一步分解
        simple_keywords = ["确认", "结束", "完成", "提交", "保存"]

        for keyword in simple_keywords:
            if keyword in high_level_step.name:
                return False
        return True

    def _execute_with_subplan(self, high_level_step: HighLevelStep,
                              context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行需要子计划的高层步骤

        Args:
            high_level_step: 高层步骤
            context: 当前上下文

        Returns:
            执行结果上下文
        """
        # 生成子计划
        subplan = self.level2_subplanner.generate_subplan(high_level_step, context)

        if not subplan:
            print(f"⚠️ 无法为 '{high_level_step.name}' 生成子计划，尝试直接执行")
            return self._execute_single(high_level_step, context)

        print(f"--- [L2] 子计划共 {len(subplan)} 个步骤 ---")

        result_context = {}
        for sub_step in subplan:
            print(f"\n--- [L2] 执行: {sub_step.name} ---")

            # 调用LLM执行子步骤（这里简化处理，实际可能需要工具调用）
            result = self._execute_low_level_step(sub_step, context, result_context)

            sub_step.observation = result
            sub_step.status = "completed"

            # 尝试从结果中提取上下文更新
            if result:
                # 简化处理：直接将结果存入上下文
                key = f"step_{sub_step.id}_result"
                result_context[key] = result

            print(f"--- [L2] 结果: {result[:50] if result else 'None'}... ---")

        return result_context

    def _execute_single(self, high_level_step: HighLevelStep,
                       context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行不需要子计划的简单高层步骤

        Args:
            high_level_step: 高层步骤
            context: 当前上下文

        Returns:
            执行结果上下文
        """
        print(f"--- [L1] 直接执行简单步骤: {high_level_step.name} ---")

        # 调用LLM直接执行
        prompt = f"问题: {context.get('question', '')}\n当前步骤: {high_level_step.name}\n请执行该步骤并给出结果。"

        messages = [{"role": "user", "content": prompt}]
        response_text = self.llm_client.think(messages) or ''

        return {"result": response_text}

    def _execute_low_level_step(self, sub_step: LowLevelStep,
                                context: Dict[str, Any],
                                local_context: Dict[str, Any]) -> str:
        """
        执行单个低层步骤

        Args:
            sub_step: 低层步骤
            context: 全局上下文
            local_context: 当前高层步骤内的局部上下文

        Returns:
            执行结果字符串
        """
        prompt = f"""问题: {context.get('question', '')}
当前上下文:
{self._format_context(context)}

当前子步骤: {sub_step.name}
请执行该步骤，给出具体结果。
"""

        messages = [{"role": "user", "content": prompt}]
        response_text = self.llm_client.think(messages) or ''

        return response_text

    def _format_context(self, context: Dict[str, Any]) -> str:
        """格式化上下文为字符串"""
        if not context:
            return "（空）"
        return "\n".join([f"  {k}: {v}" for k, v in context.items()])


class HierarchicalPlannerAgent:
    """
    分层规划智能体

    结合 Level1Planner、Level2SubPlanner 和 HierarchicalExecutor
    实现完整的分层规划-执行流程
    """

    def __init__(self, llm_client: HelloAgentLLM,
                 tool_executor=None,
                 max_high_level_steps: int = 10):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_high_level_steps = max_high_level_steps
        self.executor = HierarchicalExecutor(llm_client, tool_executor)

    def run(self, question: str) -> str:
        """
        运行分层规划智能体

        Args:
            question: 用户问题

        Returns:
            最终答案
        """
        print(f"\n{'='*60}")
        print(f"=== 分层规划智能体启动 ===")
        print(f"=== 问题: {question}")
        print(f"{'='*60}\n")

        # 初始化上下文
        context = {"question": question}

        # L1: 生成高层抽象计划
        level1_planner = Level1Planner(self.llm_client)
        high_level_steps = level1_planner.plan(question)

        if not high_level_steps:
            print("❌ 无法生成抽象计划")
            return "无法生成有效的计划，请重试。"

        print(f"\n--- [L1] 生成的抽象计划（共 {len(high_level_steps)} 个步骤）---")
        for step in high_level_steps:
            print(f"  {step.id}. {step.name}")

        # 检查步骤数量
        if len(high_level_steps) > self.max_high_level_steps:
            print(f"⚠️ 抽象计划步骤过多（{len(high_level_steps)}），截断到 {self.max_high_level_steps}")
            high_level_steps = high_level_steps[:self.max_high_level_steps]

        # 执行分层计划
        final_context = self.executor.execute(question, high_level_steps, context)

        # 提取最终答案
        final_answer = final_context.get("result", "")
        if not final_answer and final_context:
            # 如果没有明确的result字段，取最后一个有结果的高层步骤的结果
            for key in sorted(final_context.keys(), reverse=True):
                if "result" in key.lower():
                    final_answer = final_context[key]
                    break

        print(f"\n{'='*60}")
        print(f"=== 分层规划智能体完成 ===")
        print(f"=== 最终答案: {final_answer[:100]}...")
        print(f"{'='*60}\n")

        return final_answer


if __name__ == "__main__":
    from hello_agents import HelloAgentLLM

    llm_client = HelloAgentLLM()
    agent = HierarchicalPlannerAgent(llm_client)

    question = "预订一次从北京到上海的商务旅行，包括机票、酒店和租车"
    result = agent.run(question)
    print(f"\n最终结果:\n{result}")
