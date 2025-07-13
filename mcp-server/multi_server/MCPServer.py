# server.py
import os
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
import httpx
import asyncio

load_dotenv()

# Create an MCP server
mcp = FastMCP("IRIS MCP wrapper server")
mcp._tool_manager.add_tool

# Add an addition tool
@mcp.tool()
async def add(a: int, b: int) -> int:
    """两个数字相加"""
    return a+b

@mcp.tool()
async def multiply(a: int, b: int) -> int:
    """两个数字相乘"""
    return a*b

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
    #mcp.run()
    mcp.run(
        transport="sse"
    )