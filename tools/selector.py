"""
工具选择器模块
提供意图分类、工具检索和候选排序功能

核心概念:
- IntentClassifier: 意图分类器，根据用户问题判断所属业务域
- ToolSelector: 工具选择器，根据意图从注册表中选择最合适的工具候选集

工作流程:
    1. 用户问题输入
    2. IntentClassifier 识别问题意图（domain + sub_intent + confidence）
    3. ToolSelector 根据意图从 Registry 中筛选候选工具
    4. 对候选工具进行关键词匹配打分
    5. 返回得分最高的 top_k 个工具

使用示例:
    registry = ToolRegistry()
    selector = ToolSelector(registry)
    intent = selector.classifier.classify("小米手机最新型号")
    # {'domain': 'search', 'sub_intent': '信息检索', 'confidence': 'high'}

    selected = selector.select("计算 (123+456)*789", top_k=3)
    # ['Calculator']
"""

import re
from typing import List, Dict, Any, Optional


class IntentClassifier:
    """
    意图分类器

    功能:
        根据用户输入的文本，判断其所属的业务域和子意图。

    分类体系:
        - time: 时间日期相关（"现在几点了"、"今天是哪天"）
        - search: 信息检索相关（"搜索"、"查找"、"最新消息"）
        - calculate: 数学计算相关（"计算"、"等于多少"）
        - weather: 天气相关（"天气"、"温度"）
        - code: 编程相关（"代码"、"写程序"）
        - finance: 金融相关（"股票"、"基金"）
        - news: 新闻相关（"新闻"、"热点"）
        - general: 通用问答（无法归类时）

    返回结构:
        {
            "domain": str,      # 业务域，如 "search"
            "sub_intent": str,  # 子意图，如 "信息检索"
            "confidence": str   # 置信度，"high" 或 "low"
        }
    """

    def __init__(self):
        # 领域关键词映射: domain -> 关键词列表
        # 用于快速判断用户问题属于哪个领域
        self._intent_keywords = {
            "time": ["时间", "日期", "现在", "几点", "哪天", "什么时候"],
            "search": ["搜索", "查找", "查询", "新闻", "信息", "最新", "最近"],
            "calculate": ["计算", "等于", "多少", "加减乘除", "运算", "数学"],
            "weather": ["天气", "温度", "下雨", "气温"],
            "code": ["代码", "编程", "写程序", "函数", "程序"],
            "finance": ["股票", "基金", "行情", "股价", "涨跌"],
            "news": ["新闻", "热点", "事件", "报道"],
        }

        # 同义词映射: 标准词 -> 同义词列表
        # 用于扩展关键词匹配的覆盖范围
        self._intent_synonyms = {
            "search": ["搜索", "查找", "查询", "找"],
            "calculate": ["算", "计算", "求"],
        }

    def classify(self, text: str) -> Dict[str, str]:
        """
        对文本进行意图分类

        算法步骤:
            1. 遍历所有领域的关键词
            2. 检查文本中是否包含某领域的关键词
            3. 一旦匹配立即返回（优先级按定义顺序）

        Args:
            text: 用户输入的文本

        Returns:
            包含 domain、sub_intent、confidence 的字典

        示例:
            classify("现在几点了")
            -> {"domain": "time", "sub_intent": "获取当前时间", "confidence": "high"}

            classify("小米手机最新型号")
            -> {"domain": "search", "sub_intent": "信息检索", "confidence": "high"}
        """
        text_lower = text.lower()

        # 按优先级检查各领域关键词
        # time 优先级最高（如果问时间，先返回时间）
        for kw in self._intent_keywords.get("time", []):
            if kw in text:
                return {"domain": "time", "sub_intent": "获取当前时间", "confidence": "high"}

        # calculate 次之（如果问计算，先返回计算）
        for kw in self._intent_keywords.get("calculate", []):
            if kw in text:
                return {"domain": "calculate", "sub_intent": "数学计算", "confidence": "high"}

        # search 通用搜索
        for kw in self._intent_keywords.get("search", []):
            if kw in text:
                return {"domain": "search", "sub_intent": "信息检索", "confidence": "high"}

        # 未能匹配任何领域，返回通用域
        return {"domain": "general", "sub_intent": "通用问答", "confidence": "low"}

    def expand_synonyms(self, word: str) -> List[str]:
        """
        获取词汇的同义词列表

        Args:
            word: 关键词

        Returns:
            原词 + 所有同义词的列表

        示例:
            expand_synonyms("搜索")
            -> ["搜索", "查找", "查询", "找"]
        """
        synonyms = [word]
        # 遍历所有同义词映射，找包含该词的条目
        for intent, words in self._intent_synonyms.items():
            if word in words:
                synonyms.extend(words)
        # 去重
        return list(set(synonyms))


class ToolSelector:
    """
    工具选择器

    核心功能:
        1. 根据用户问题选择最相关的工具
        2. 按相关度对候选工具排序
        3. 返回 top_k 个候选工具

    选择算法:
        1. 意图分类: 判断问题属于哪个业务域
        2. 候选生成: 从Registry获取该域的工具（或全部工具）
        3. 打分排序: 根据工具名匹配(+10)、标签匹配(+5)、描述匹配(+2)计算得分
        4. 结果返回: 返回得分>0的前k个工具

    使用示例:
        selector = ToolSelector(registry)
        selected = selector.select("计算 (123+456)*789", top_k=5)
        # ['Calculator']
    """

    def __init__(self, registry):
        """
        初始化工具选择器

        Args:
            registry: 工具注册表实例
        """
        self.registry = registry
        self.classifier = IntentClassifier()

    def select(self, question: str, top_k: int = 5) -> List[str]:
        """
        根据问题选择最相关的工具

        Args:
            question: 用户问题
            top_k: 最多返回k个候选工具

        Returns:
            工具名列表，按相关度从高到低排序
        """
        # Step 1: 意图分类
        intent_info = self.classifier.classify(question)
        domain = intent_info["domain"]

        # Step 2: 获取候选工具
        if domain != "general":
            # 如果能归类，只从该分类取工具
            candidates = self.registry.get_by_category(domain)
        else:
            # 无法归类，从所有工具中选
            candidates = self.registry.get_all()

        # Step 3: 关键词匹配打分
        scored = self._score_tools(question, candidates)

        # Step 4: 排序并返回top_k
        scored.sort(key=lambda x: x[1], reverse=True)  # 按分数降序
        return [name for name, score in scored[:top_k] if score > 0]

    def _score_tools(self, question: str, candidates: List) -> List[tuple]:
        """
        根据关键词匹配度对工具打分

        打分规则:
            - 工具名称出现在问题中: +10分
            - 工具标签出现在问题中: +5分/标签
            - 工具描述的单词出现在问题中: +2分/单词

        Args:
            question: 用户问题
            candidates: 候选工具列表

        Returns:
            [(tool_name, score), ...] 列表
        """
        scores = []
        question_lower = question.lower()

        for tool in candidates:
            score = 0

            # 工具名精确匹配
            if tool.name.lower() in question_lower:
                score += 10

            # 标签匹配
            for tag in tool.tags:
                if tag.lower() in question_lower:
                    score += 5

            # 描述关键词匹配（分词后匹配）
            desc_lower = tool.description.lower()
            for kw in re.findall(r'\w+', question_lower):  # 分词
                if kw in desc_lower:
                    score += 2

            scores.append((tool.name, score))

        return scores

    def get_examples(self, tool_name: str) -> List[str]:
        """
        获取工具的调用示例

        用于在prompt中提供few-shot示例，帮助LLM学习正确调用方式。

        Args:
            tool_name: 工具名称

        Returns:
            示例列表，如 ["Search[小米手机]", "Search[今天天气]"]
        """
        metadata = self.registry.get(tool_name)
        return metadata.examples if metadata else []
