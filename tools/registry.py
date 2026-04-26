"""
工具注册与管理模块
提供工具的注册、存储、分类和元数据管理功能

核心概念:
- ToolMetadata: 工具的元数据结构，包含名称、描述、分类、标签、参数、示例等信息
- ToolRegistry: 工具注册表，统一管理所有工具的注册、存储和检索

使用示例:
    registry = ToolRegistry()
    registry.register(
        name="Search",
        description="搜索引擎",
        func=search_function,
        category="search",
        tags=["搜索", "查询"],
        parameters={"query": "搜索关键词"},
        examples=["Search[小米手机]"]
    )
"""

from typing import Dict, Any, Callable, List, Optional


class ToolMetadata:
    """
    工具元数据结构

    属性说明:
        name: 工具名称，唯一标识符，如 "Search"、"Calculator"
        description: 工具的功能描述，供LLM理解工具用途
        func: 工具对应的Python函数
        category: 工具所属分类，用于按业务域分组，如 "search"、"time"、"calculate"
        tags: 工具的标签列表，用于关键词检索，如 ["搜索", "查询", "信息"]
        parameters: 工具参数定义，key为参数名，value为参数描述
        examples: 工具调用示例列表，供LLM学习正确调用方式
    """

    def __init__(self, name: str, description: str, func: Callable,
                 category: str = "general", tags: List[str] = None,
                 parameters: Dict[str, Any] = None, examples: List[str] = None):
        self.name = name
        self.description = description
        self.func = func
        self.category = category
        self.tags = tags or []
        self.parameters = parameters or {}
        self.examples = examples or []

    def to_dict(self) -> dict:
        """
        转换为字典格式

        Returns:
            包含工具所有元数据的字典
        """
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "tags": self.tags,
            "parameters": self.parameters,
        }

    def to_llm_format(self) -> str:
        """
        转换为LLM可读的描述格式

        格式示例:
            - Search(query): 网页搜索引擎，用于搜索实时信息

        Returns:
            格式化字符串，供LLM理解工具接口
        """
        # 拼接参数名列表，如 "query, recency_days?"
        params_str = ", ".join(self.parameters.keys()) if self.parameters else "无"
        return f"- {self.name}({params_str}): {self.description}"


class ToolRegistry:
    """
    工具注册表

    核心功能:
        1. 工具注册: 通过register()方法注册新工具
        2. 工具获取: 通过get()/get_func()获取工具元数据或函数
        3. 分类管理: 通过get_by_category()获取特定分类的工具
        4. 工具描述: 通过describe_all()/describe_category()生成LLM可读的描述

    使用示例:
        registry = ToolRegistry()
        registry.register(name="Search", description="搜索工具", func=search_func, category="search")
        tools = registry.get_by_category("search")
        desc = registry.describe_all()
    """

    def __init__(self):
        # 工具字典: name -> ToolMetadata
        self._tools: Dict[str, ToolMetadata] = {}

        # 分类字典: category -> [tool_names]
        # 用于快速获取某分类下的所有工具
        self._categories: Dict[str, List[str]] = {}

    def register(self, name: str, description: str, func: Callable,
                 category: str = "general", tags: List[str] = None,
                 parameters: Dict[str, Any] = None, examples: List[str] = None) -> None:
        """
        注册一个新工具到注册表

        Args:
            name: 工具名称，必须唯一
            description: 工具功能描述
            func: 工具对应的Python函数
            category: 所属分类，默认"general"
            tags: 标签列表，用于检索
            parameters: 参数定义字典
            examples: 调用示例列表
        """
        # 创建元数据对象
        metadata = ToolMetadata(
            name=name,
            description=description,
            func=func,
            category=category,
            tags=tags,
            parameters=parameters,
            examples=examples
        )

        # 存储到工具字典
        self._tools[name] = metadata

        # 更新分类字典
        if category not in self._categories:
            self._categories[category] = []
        if name not in self._categories[category]:
            self._categories[category].append(name)

    def get(self, name: str) -> Optional[ToolMetadata]:
        """
        根据名称获取工具元数据

        Args:
            name: 工具名称

        Returns:
            工具元数据对象，不存在返回None
        """
        return self._tools.get(name)

    def get_func(self, name: str) -> Optional[Callable]:
        """
        根据名称获取工具函数

        Args:
            name: 工具名称

        Returns:
            工具函数，不存在返回None
        """
        metadata = self._tools.get(name)
        return metadata.func if metadata else None

    def get_by_category(self, category: str) -> List[ToolMetadata]:
        """
        获取指定分类下的所有工具

        Args:
            category: 分类名称

        Returns:
            该分类下的工具元数据列表
        """
        # 从分类字典获取该分类的工具名列表
        tool_names = self._categories.get(category, [])
        # 转换为元数据对象列表
        return [self._tools[name] for name in tool_names if name in self._tools]

    def get_all(self) -> List[ToolMetadata]:
        """
        获取所有已注册的工具

        Returns:
            所有工具的元数据列表
        """
        return list(self._tools.values())

    def get_all_categories(self) -> List[str]:
        """
        获取所有已注册的分类

        Returns:
            分类名称列表
        """
        return list(self._categories.keys())

    def get_category_map(self) -> Dict[str, List[str]]:
        """
        获取分类到工具名的映射关系

        Returns:
            {category_name: [tool_names]} 字典
        """
        return dict(self._categories)

    def describe_all(self) -> str:
        """
        获取所有工具的LLM可读描述

        格式示例:
            【搜索类 - 关键词: search】
              - Search(query): 网页搜索引擎

        注意: 分类标题中的【】内是分类信息，
              下面的工具名（如Search）才是调用时使用的tool_name。

        Returns:
            格式化字符串
        """
        if not self._tools:
            return "暂无可用工具"

        lines = []
        # 按分类遍历
        for category, tool_names in self._categories.items():
            # 分类标题，强调category和tool_name的区别
            lines.append(f"\n【{category}类 - 关键词: {category}】")
            # 该分类下的每个工具
            for name in tool_names:
                metadata = self._tools[name]
                lines.append(f"  - {metadata.to_llm_format()}")

        lines.append("\n注意: 请使用上述工具名（如Search、Calculator）作为tool_name，")
        lines.append("       不要使用分类关键词（如search、calculate）")

        return "\n".join(lines)

    def describe_category(self, category: str) -> str:
        """
        获取指定分类下工具的描述

        Args:
            category: 分类名称

        Returns:
            格式化字符串
        """
        tools = self.get_by_category(category)
        if not tools:
            return f"分类 '{category}' 下暂无工具"

        lines = [f"【{category}】"]
        for metadata in tools:
            lines.append(f"  {metadata.to_llm_format()}")
        return "\n".join(lines)
