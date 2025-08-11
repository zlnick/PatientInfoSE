# server.py
import os
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from openapi_parser import generate_tool_list
from rest_api_tool_generator import RESTAPIToolGenerator
import httpx
import asyncio
import base64
from typing import Dict, List, Any
import json
import requests

load_dotenv()

# Create an MCP server
mcp = FastMCP("MCP Server on IRIS")

# 连接IRIS上被暴露的表元数据
def get_table_meta(url,namespace,scheme):
    try:
        # 准备认证头
        headers = {}
        username = os.getenv("IRIS_USERNAME")
        password = os.getenv("IRIS_PASSWORD")
        headers = {}
        if username and password:
            # 创建基本认证头
            credentials = f"{username}:{password}"
            encoded_credentials = base64.b64encode(credentials.encode('utf-8')).decode('utf-8')
            headers["Authorization"] = f"Basic {encoded_credentials}"
        # 发送GET请求
        response = requests.get(url+'/'+namespace+'/'+scheme,headers=headers)
        
        # 检查请求是否成功
        response.raise_for_status()  # 如果响应状态码不是200，会抛出HTTPError异常
        print(f"获取IRIS表元数据成功! 状态码: {response.status_code}")
        # 解析JSON响应
        tables = response.json()
        return tables
        # 打印JSON数据
        #print("API返回的表元数据：")
        #print(json.dumps(json_data, ensure_ascii=False))
    except requests.exceptions.HTTPError as errh:
        print(f"HTTP错误: {errh}")
    except requests.exceptions.ConnectionError as errc:
        print(f"连接错误: {errc}")
    except requests.exceptions.Timeout as errt:
        print(f"超时错误: {errt}")
    except requests.exceptions.RequestException as err:
        print(f"请求异常: {err}")
    except json.JSONDecodeError:
        print("无法解析响应为JSON格式")
    return None

async def get_iris_apis() -> Dict :
    """在服务器启动前执行REST API检查"""
    spec_url = os.getenv("IRIS_OPENAPI_SPEC")
    if not spec_url:
        print("未设置 IRIS_OPENAPI_SPEC 环境变量，跳过启动检查")
        return
    
    print(f"正在获取IRIS API Spec，访问URL: {spec_url}")
    # 获取认证凭据
    username = os.getenv("IRIS_USERNAME")
    password = os.getenv("IRIS_PASSWORD")
    # 准备请求头
    headers = {}
    if username and password:
        # 创建基本认证头
        credentials = f"{username}:{password}"
        encoded_credentials = base64.b64encode(credentials.encode('utf-8')).decode('utf-8')
        headers["Authorization"] = f"Basic {encoded_credentials}"
        #print("已添加基本认证凭据")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                spec_url, 
                headers=headers, 
                timeout=10
            )
            response.raise_for_status()
            print(f"获取IRIS API Spec成功! 状态码: {response.status_code}")
            spec = response.text
            return spec
        except Exception as e:
            print(f"启动检查失败: {str(e)}")


@mcp.tool()
async def query_fhir(resource_type: str, filters: dict) -> dict:
    """
    查询FHIR服务器上的指定资源，支持传入过滤条件。
    患者用药可以用MedicationStatement来查询。
    :param resource_type: FHIR资源类型（如 'Observation', 'Patient'）
    :param filters: 查询过滤条件（如 {'subject': 'Patient/794', 'code': '85354-9', 'date': 'ge2015-06-03'}。）
        注意，id、资源id这样的条件不对应参数identifier，而应该使用参数id
        在检验检查项目中，其类型遵循SNOMED-CT术语，如下：
            {"血压":"85354-9"}
    :return: 查询结果（FHIR Bundle JSON）
    """
    fhir_base_url = os.getenv("FHIR_BASE_URL")  # 你可以在 .env 文件中设置 FHIR API 地址
    if not fhir_base_url:
        raise Exception("未设置 FHIR_BASE_URL 环境变量")

    # 构造查询URL
    url = f"{fhir_base_url}/{resource_type}"
    query_params = ''
    for k, v in filters.items():
        if k == 'id':
            url = url + f"/{v}" 
        else:
            query_params = '&'.join([f"{k}={v}" for k, v in filters.items()])
    url = f"{url}?{query_params}"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data
        except Exception as e:
            raise Exception(f"FHIR 查询失败: {e}")

sql_query_Desc = """
    在IRIS服务器上执行SQL语句查询，传入待执行SQL语句，返回查询结果。
    这些表只用于记录收费方面的数据，不能当作临床数据。
    :param sqlStatement: SQL语句（如 SELECT * FROM Data.OrderItem WHERE OrderID IN ( SELECT ID FROM Data.Order WHERE Patient = 'Patient/794')）
    :return: 查询结果，格式为JSON，如下：
        {"status":{"errors":[],"summary":""},"console":[],"result":{"content":[{"ID":1,"Currency":"CNY","ItemName":"阿奇霉素","OrderID":"1","Price":1666},{"ID":2,"Currency":"CNY","ItemName":"曲马多","OrderID":"1","Price":983}]}}
        要注意结果集在result的content结点中。
        当前可用的表包括："""
async def query_sql(sqlStatement: str)-> dict:
    sql_base_url = os.getenv("SQL_BASE_URL")
    if not sql_base_url:
        raise Exception("未设置 SQL_BASE_URL 环境变量")
    payload = {"query":sqlStatement}
    # 获取认证凭据
    username = os.getenv("IRIS_USERNAME")
    password = os.getenv("IRIS_PASSWORD")
    # 准备请求头
    headers = {}
    if username and password:
        # 创建基本认证头
        credentials = f"{username}:{password}"
        encoded_credentials = base64.b64encode(credentials.encode('utf-8')).decode('utf-8')
        headers["Authorization"] = f"Basic {encoded_credentials}"
        #print("已添加基本认证凭据")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                sql_base_url, 
                headers=headers, 
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            print(f"执行SQL查询! 状态码: {response.status_code}")
            spec = response.text
            return spec
        except Exception as e:
            print(f"执行SQL失败: {str(e)}")


if __name__ == "__main__":

    # 动态获取IRIS上的API定义
    spec = json.loads(asyncio.run(get_iris_apis()))
    #补丁：由于IRIS会自动以域名+端口作为host的根路径（如mcpdemo:52773），暂时需要手动将其替换为docker环境下可访问的地址如(localhost:52880)
    spec['host'] = 'localhost:52880'
    #print(spec)
    # 将OpenAI 2.0版本的REST API规范转换为如下格式的Python JSON对象
    """
     [
        {
            "name": "addNumbers",
            "description": "两个数相加，返回两数之和。",
            "api_path": "http://localhost:52880/MCP/MCPTools/Math/add",
            "method": "post",
            "input_schema": {
                "type": "object",
                "title": "addNumbersArguments",
                "properties": {
                    "a": {
                        "type": "number",
                        "title": "",
                        "description": "First number"
                    },
                    "b": {
                        "type": "number",
                        "title": "",
                        "description": "Second number"
                    }
                },
                "required": ["a", "b"]
            }
        }
     ]
    """
    api_dict = generate_tool_list(spec)
    #print(api_dict)
    # 创建工具生成器
    generator = RESTAPIToolGenerator(api_dict)
    # 获取并注册生成的工具函数
    tools = generator.get_tool_functions()
    for func_name, func in tools.items():
        mcp.add_tool(func, name=func_name)
        print(f"动态工具已添加：{func_name}")
    # 将SQL表可读性注入SQL查询工具的注释中，便于大模型使用
    table_desc = get_table_meta(os.getenv("TABLE_META_ENDPOINT"),os.getenv("TABLE_NS"),os.getenv("TABLE_SCHEME"))
    desc = sql_query_Desc+json.dumps(table_desc, separators=(',', ':'),ensure_ascii=False)
    #print(desc)
    func = query_sql
    func.__doc__ = desc
    mcp.add_tool(func, name=func.__name__)
    # 以sse模式启动MCP服务器
    mcp.run(
        transport="sse"
    )