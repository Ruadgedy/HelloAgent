text = """
临时测试文件：验证正则表达式匹配 Thought 和 Action 的效果
用于调试 ReAct Agent 中的输出解析逻辑
"""

# 测试文本，模拟 LLM 的原始输出格式
text = """
Thought: 先思考第一步。
Action: step1
Thought: 再思考第二步。
Action: step2
"""

import re

# 正则模式：匹配 "Thought:" 后到 "Action:" 或字符串末尾之间的内容
# (?=\nAction:|$) 是 lookahead，匹配但不消费
pattern = r"Thought:\s*(.*)(?=\nAction:|$)"

# re.search: 从字符串任意位置查找匹配（找到第一个即可）
match = re.search(pattern, text, re.DOTALL)

# re.match: 从字符串开头匹配（此场景下与 search 等价）
match2 = re.match(pattern, text, re.DOTALL)

# group(0): 完整匹配内容
# group(1): 捕获组内容，即 Thought 后面的实际文本
print("--- re.search 结果 ---")
print(f"group(0): {match.group(0)}")
print(f"group(1): {match.group(1)}")

print("\n--- re.match 结果 ---")
print(match2)