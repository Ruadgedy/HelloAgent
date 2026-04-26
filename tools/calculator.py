import numexpr


def calculate(expression: str) -> str:
    """
    计算数学表达式的值。
    支持加减乘除、括号、指数等标准数学表达式。

    Args:
        expression: 数学表达式字符串，如 "(123 + 456) * 789 / 12"

    Returns:
        计算结果的字符串形式
    """
    try:
        expression = expression.strip()
        result = numexpr.evaluate(expression)
        return str(result)
    except ZeroDivisionError:
        return "错误: 除数不能为零"
    except Exception as e:
        return f"错误: 无法计算表达式 '{expression}'，原因: {e}"
