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

load_dotenv()

# Create an MCP server
mcp = FastMCP("MCP Server on IRIS")

# 连接IRIS获取OpenAPI规范
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
        print("已添加基本认证凭据")
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
    查询FHIR服务器上的指定资源，支持传入过滤条件
    :param resource_type: FHIR资源类型（如 'Observation', 'Patient'）
    :param filters: 查询过滤条件（如 {'subject': 'Patient/794', 'code': '85354-9', 'date': 'ge2015-06-03'}）
    :return: 查询结果（FHIR Bundle JSON）
    """
    fhir_base_url = os.getenv("FHIR_BASE_URL")  # 你可以在 .env 文件中设置 FHIR API 地址
    if not fhir_base_url:
        raise Exception("未设置 FHIR_BASE_URL 环境变量")

    # 构造查询URL
    query_params = '&'.join([f"{k}={v}" for k, v in filters.items()])
    url = f"{fhir_base_url}/{resource_type}?{query_params}"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data
        except Exception as e:
            raise Exception(f"FHIR 查询失败: {e}")


if __name__ == "__main__":

    # 动态获取IRIS上的API定义
    spec = json.loads(asyncio.run(get_iris_apis()))
    print(spec)
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
    print(api_dict)
    # 创建工具生成器
    generator = RESTAPIToolGenerator(api_dict)
    # 获取并注册生成的工具函数
    tools = generator.get_tool_functions()
    for func_name, func in tools.items():
        mcp.add_tool(func, name=func_name)
        print(f"动态工具已添加：{func_name}")
    mcp.run(
        transport="sse"
    )