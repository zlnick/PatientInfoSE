# utils.py
import requests
import json
import base64

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


def get_result_value(data):
    """
    假设经parse_mcp_result处理过的result一定会采用[('text', '24')]这样的结构，但text标签会变。
    从中取出变量的值，如上例中的24。
    """
    if isinstance(data, list) and len(data) > 0:
        item = data[0]
        if isinstance(item, tuple) and len(item) > 1:
            return item[1]
    return None


def get_practitioner(id):
    """
    使用 HTTP GET请求调用REST API并返回解析后的JSON对象
    
    参数:
        url: API的URL地址
        params: 可选，请求参数（字典形式）
        headers: 可选，请求头信息（字典形式）
        
    返回:
        解析后的JSON对象（字典或列表），如果请求失败则返回None
    """
    url = 'http://localhost:52880/csp/healthshare/fhirserver/fhir/r4/Practitioner/'+id
    # 请求头
    """
    # 准备认证头
    headers = {}
    username = "superuser"
    password = "SYS"
    if username and password:
        credentials = f"{{username}}:{{password}}"
        encoded_credentials = base64.b64encode(credentials.encode('utf-8')).decode('utf-8')
        headers["Authorization"] = f"Basic {{encoded_credentials}}"
    """
    headers = {
        "Content-Type": "application/fhir+json;charset=utf-8",
        "User-Agent": "Python HTTP Client"
    }
    
    try:
        # 发送GET请求
        response = requests.get(url, headers=headers)
        
        # 检查响应状态码，200表示成功
        response.raise_for_status()
        
        # 解析JSON响应
        json_data = response.json()
        
        print("FHIR Practitioner API请求成功！")
        return json_data
        
    except requests.exceptions.HTTPError as e:
        print(f"HTTP错误: {e}")
    except requests.exceptions.ConnectionError:
        print("连接错误：无法连接到服务器")
    except requests.exceptions.Timeout:
        print("超时错误：请求超时")
    except requests.exceptions.RequestException as e:
        print(f"请求异常: {e}")
    except json.JSONDecodeError:
        print("JSON解析错误：响应内容不是有效的JSON格式")
    
    return None

def get_official_name(result):
    """
    从结果中获取name列表里use为official的text值
    
    参数:
        result: API返回的JSON对象（Python字典）
        
    返回:
        str: 符合条件的text值，如果未找到则返回提示信息
    """
    # 安全获取name列表，若不存在则返回空列表
    name_list = result.get('name', [])
    
    # 遍历name列表，寻找use为official的条目
    for name_item in name_list:
        # 检查当前条目是否有use字段且值为official
        if name_item.get('use') == 'official':
            # 返回对应的text值，若text不存在则返回默认提示
            return name_item.get('text', '未找到姓名文本')
    
    # 若遍历完未找到符合条件的条目
    return '未找到use为official的name信息'

# 示例用法
if __name__ == "__main__":
    # 示例API地址（使用JSONPlaceholder的测试API）
    id = "1"
    
    
    
    # 调用API并获取数据
    result = get_practitioner(id)
    
    if result:
        print("\n返回的JSON数据:")
        # 格式化打印JSON数据
        #print(json.dumps(result, indent=2))
        print(result.get('name', []))
        print(get_official_name(result))
