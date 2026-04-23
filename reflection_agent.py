from typing import List, Dict, Any, Optional

from hello_agents import HelloAgentLLM


class Memory:
    """
    一个简单的短期记忆模块，用于存储智能体的行动与反思轨迹。
    """

    def __init__(self):
        """初始化记忆模块，清空历史记录"""
        self.records: List[Dict[str, Any]] = []

    def add_record(self, record_type: str, content: str):
        """
        向记忆中添加一条新记录。

        参数:
        - record_type (str): 记录的类型 ('execution' 或 'reflection')。
        - content (str): 记录的具体内容 (例如，生成的代码或反思的反馈)。
        """
        self.records.append({"type": record_type, "content": content})
        print(f"📝 记忆已更新，新增一条 '{record_type}' 记录。")

    def get_trajectory(self) -> str:
        """
        将所有记忆记录格式化为一个连贯的字符串文本，用于构建提示词。
        """
        trajectory_parts = []
        for record in self.records:
            if record["type"] == "execution":
                trajectory_parts.append(f"--- 上一轮尝试 (代码) ---\n{record['content']}")
            elif record["type"] == "reflection":
                trajectory_parts.append(f"--- 上一轮反思 (反馈) ---\n{record['content']}")

        print(f"📝 已生成记忆轨迹，共 {len(trajectory_parts)} 条记录。")
        return "\n\n".join(trajectory_parts)

    def get_last_execution(self) -> Optional[str]:
        """
        获取最近一次的执行结果 (例如，最新生成的代码)。
        如果不存在，则返回 None。
        """
        for record in reversed(self.records):
            if record["type"] == "execution":
                return record["content"]
        return None


class ReflectionAgent:

    def __init__(self, llm_client: HelloAgentLLM, max_iterations=3):
        self.llm_client = llm_client
        self.memory = Memory()
        self.max_iterations = max_iterations

    def run(self, task: str):
        print(f"\n--- 开始处理任务 ---\n任务: {task}")

        # --- 1. 初始执行 ---
        print("\n--- 正在进行初始尝试 ---")
        with open("./prompt/execution_prompt.txt", "r") as f:
            initial_prompt = f.read().strip()

        initial_prompt = initial_prompt.format(task=task)
        initial_code = self._get_llm_response(initial_prompt)

        self.memory.add_record("execution", initial_code)

        # --- 2. 迭代循环:反思与优化 ---
        for i in range(self.max_iterations):
            print(f"\n第 {i+1}/{self.max_iterations} 轮反思")

            # a. 反思
            last_code = self.memory.get_last_execution()
            with open("./prompt/reflection_prompt.txt", "r") as f:
                reflection_prompt = f.read().strip()
            reflection_prompt = reflection_prompt.format(task=task, code=last_code)
            feedback = self._get_llm_response(reflection_prompt)
            self.memory.add_record("reflection", feedback)

            # b. 检查是否需要停止
            if "无需改进" in feedback:
                print("\n✅ 反思认为代码已无需改进，任务完成。")
                break

            # c. 继续优化
            print("\n✅ 反思认为代码需要改进，继续优化。")
            with open("./prompt/refine_prompt.txt", "r") as f:
                refine_prompt = f.read().strip()
            refine_prompt = refine_prompt.format(task=task, last_code_attempt=last_code, feedback=feedback)
            refine_code = self._get_llm_response(refine_prompt)
            self.memory.add_record("execution", refine_code)

        final_code = self.memory.get_last_execution()
        print(f"\n--- 任务完成 ---\n最终生成的代码:\n```python\n{final_code}\n```")
        return final_code

    def _get_llm_response(self, prompt: str) -> str:
        """一个辅助方法，用于调用LLM并获取完整的流式响应。"""
        messages = [{"role": "user", "content": prompt}]
        response_text = self.llm_client.think(message=messages) or ""
        return response_text

if __name__ == "__main__":
    agent = ReflectionAgent(llm_client=HelloAgentLLM())
    task = "编写一个Python函数，找出1到n之间所有的素数"
    agent.run(task)
