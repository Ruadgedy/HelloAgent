# 代码规范

## 新增代码必须包含注释

### 1. 模块级 docstring
说明模块目的、核心概念、使用示例

```python
"""
模块名称

核心概念:
- 概念A: 说明
- 概念B: 说明

使用示例:
    from module import func
    result = func(input)
"""
```

### 2. 类级 docstring
属性说明、功能概述、使用示例

```python
class MyClass:
    """
    类功能概述

    属性说明:
        attr1: 属性1的含义
        attr2: 属性2的含义

    使用示例:
        obj = MyClass()
        obj.method()
    """
```

### 3. 方法级 docstring
参数说明、返回值、异常处理、算法步骤

```python
def my_method(self, arg1: str, arg2: int) -> bool:
    """
    方法功能简述

    Args:
        arg1: 参数1的含义
        arg2: 参数2的含义

    Returns:
        True if 成功, False otherwise

    Raises:
        ValueError: 当arg2为0时

    算法步骤:
        1. 验证输入
        2. 执行核心逻辑
        3. 返回结果
    """
```

### 4. 关键逻辑注释
解释"为什么这样做"，而非"做什么"

```python
# 为什么要用指数退避而非线性等待
# 因为指数退避可以快速失败又不会过度频繁重试
sleep_time = base * (2 ** attempt)
```
