# utils.py
def parse_mcp_result(result):
    """
    解析 MCP 工具调用结果，支持文本、文件、图片等内容。
    返回一个列表，每个元素为（类型, 内容）。
    """
    parsed_contents = []
    if hasattr(result, 'isError') and result.isError:
        parsed_contents.append(("error", "调用 MCP 工具失败"))
    elif hasattr(result, 'content'):
        for item in result.content:
            if hasattr(item, "text"):
                parsed_contents.append(("text", item.text))
            elif hasattr(item, "url"):
                parsed_contents.append(("file", item.url))
            elif hasattr(item, "image"):
                parsed_contents.append(("image", item.image))
            else:
                parsed_contents.append(("unknown", str(item)))
    else:
        parsed_contents.append(("unknown", str(result)))
    return parsed_contents
