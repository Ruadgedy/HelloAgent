from datetime import datetime

def get_current_time(info: str = "") -> str:
    """
    获取当前日期和时间的工具。
    当用户问题涉及"最新"、"当前"、"今年"、"何时"等时间相关词汇时，
    应先调用此工具获取准确的当前时间，再进行后续搜索或回答。
    """
    now = datetime.now()
    return now.strftime("%Y年%m月%d日 %H:%M:%S")