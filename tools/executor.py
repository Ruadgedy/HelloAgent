"""
工具执行层模块
提供工具调用、结果验证和失败重试功能

核心概念:
- ExecutionStatus: 执行状态枚举，表示工具调用的结果
- ExecutionResult: 执行结果数据结构，包含状态、观察、错误等信息
- ToolExecutor: 工具执行器，负责实际调用工具并处理异常和重试

重试机制:
    当工具调用失败时，执行器会自动重试，最多重试 max_retries 次。
    重试间隔采用指数退避策略: 0.5s, 1s, 2s, ...

使用示例:
    registry = ToolRegistry()
    executor = ToolExecutor(registry, max_retries=2)

    result = executor.execute("Search", "小米手机")
    if result.status == ExecutionStatus.SUCCESS:
        print(result.observation)
    else:
        print(f"执行失败: {result.error}")
"""

import time
from typing import Any, Callable, Optional, Dict, List
from dataclasses import dataclass
from enum import Enum


class ExecutionStatus(Enum):
    """
    工具执行状态枚举

    状态说明:
        - SUCCESS: 执行成功
        - FAILURE: 执行失败（可能是工具不存在、参数错误、运行时异常等）
        - RETRY_EXHAUSTED: 重试次数用尽后仍然失败
    """
    SUCCESS = "success"
    FAILURE = "failure"
    RETRY_EXHAUSTED = "retry_exhausted"


@dataclass
class ExecutionResult:
    """
    工具执行结果数据结构

    属性说明:
        status: 执行状态，ExecutionStatus枚举值
        observation: 执行结果或错误信息，供LLM理解的观察
        tool_name: 被调用的工具名称
        tool_input: 调用时传入的输入参数
        attempts: 尝试次数（包含首次调用和重试）
        error: 错误详情，仅在失败时有值
    """
    status: ExecutionStatus
    observation: str
    tool_name: str
    tool_input: Any
    attempts: int
    error: Optional[str] = None


class ToolExecutor:
    """
    工具执行器

    核心职责:
        1. 根据工具名从Registry获取工具函数
        2. 执行工具调用，捕获异常
        3. 失败时自动重试，采用指数退避策略
        4. 记录执行历史

    重试策略:
        - 最大重试次数: max_retries（默认2次）
        - 重试间隔: 0.5 * (attempt + 1) 秒，即 0.5s, 1s, 2s...
        - 只要最终成功，status仍为SUCCESS

    使用示例:
        executor = ToolExecutor(registry, max_retries=2)
        result = executor.execute("Calculator", "(123+456)*2")

        if result.status == ExecutionStatus.SUCCESS:
            print(f"结果: {result.observation}")
        else:
            print(f"失败原因: {result.error}")
            print(f"尝试次数: {result.attempts}")
    """

    def __init__(self, registry, max_retries: int = 2, timeout: float = 30.0):
        """
        初始化工具执行器

        Args:
            registry: 工具注册表，用于获取工具函数
            max_retries: 最大重试次数，默认2次（即最多调用3次）
            timeout: 超时时间（暂未使用），默认30秒
        """
        self.registry = registry
        self.max_retries = max_retries
        self.timeout = timeout
        # 执行历史，记录每次调用的情况
        self._execution_history: List[Dict] = []

    def execute(self, tool_name: str, tool_input: Any) -> ExecutionResult:
        """
        执行工具调用，带重试机制

        执行流程:
            1. 从Registry获取工具函数
            2. 如果工具不存在，立即返回失败
            3. 执行函数调用
            4. 如果成功，返回结果
            5. 如果失败且还有重试次数，等待后重试
            6. 重试用尽仍失败，返回失败结果

        Args:
            tool_name: 工具名称
            tool_input: 工具输入参数

        Returns:
            ExecutionResult 对象，包含执行状态和所有详细信息
        """
        # Step 1: 获取工具函数
        func = self.registry.get_func(tool_name)
        if not func:
            # 工具不存在，快速失败
            return ExecutionResult(
                status=ExecutionStatus.FAILURE,
                observation=f"错误: 未找到工具 '{tool_name}'",
                tool_name=tool_name,
                tool_input=tool_input,
                attempts=0,
                error=f"工具不存在: {tool_name}"
            )

        # Step 2: 尝试执行，最多重试 max_retries 次
        attempts = 0
        last_error = None

        for attempt in range(self.max_retries + 1):
            attempts = attempt + 1
            try:
                # 执行工具函数
                observation = func(tool_input)

                # 执行成功，记录历史并返回
                self._execution_history.append({
                    "tool": tool_name,
                    "input": tool_input,
                    "status": "success",
                    "attempts": attempts
                })

                return ExecutionResult(
                    status=ExecutionStatus.SUCCESS,
                    observation=observation,
                    tool_name=tool_name,
                    tool_input=tool_input,
                    attempts=attempts
                )

            except Exception as e:
                # 执行失败，记录错误
                last_error = str(e)

                # 如果还有重试机会
                if attempt < self.max_retries:
                    # 指数退避等待: 0.5s, 1s, 2s...
                    sleep_time = 0.5 * (attempt + 1)
                    time.sleep(sleep_time)

        # Step 3: 重试用尽，仍然失败
        self._execution_history.append({
            "tool": tool_name,
            "input": tool_input,
            "status": "failure",
            "attempts": attempts,
            "error": last_error
        })

        return ExecutionResult(
            status=ExecutionStatus.FAILURE,
            observation=f"错误: 工具 '{tool_name}' 执行失败，已重试{attempts}次。最后错误: {last_error}",
            tool_name=tool_name,
            tool_input=tool_input,
            attempts=attempts,
            error=last_error
        )

    def get_history(self) -> List[Dict]:
        """
        获取执行历史

        记录格式:
            [
                {"tool": "Search", "input": "...", "status": "success", "attempts": 1},
                {"tool": "Calculator", "input": "...", "status": "failure", "attempts": 3, "error": "..."}
            ]

        Returns:
            执行历史列表
        """
        return self._execution_history
