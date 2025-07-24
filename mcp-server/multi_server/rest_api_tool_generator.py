import httpx
import inspect
import json
import types
import asyncio
import os
from typing import Dict, Any, Callable
import base64

class RESTAPIToolGenerator:
    def __init__(self, api_metadata: Dict[str, Any]):
        """
        初始化 REST API 工具生成器
        :param api_metadata: REST API 元数据描述
        """
        self.api_metadata = api_metadata
        self.tool_functions = {}
        self._generate_functions()
    
    def _generate_functions(self):
        """为每个 API 元数据生成对应的 Python 函数"""
        for api in self.api_metadata:
            # 创建函数名称和描述
            func_name = api["name"]
            description = api["description"]
            
            # 创建函数签名
            signature = self._create_function_signature(api["input_schema"])
            
            # 创建函数实现
            func_impl = self._create_function_implementation(
                api["api_path"],
                api["method"],
                signature
            )
            
            # 创建函数对象
            func = types.FunctionType(
                code=func_impl.__code__,
                globals=globals(),
                name=func_name,
                argdefs=func_impl.__defaults__,
                closure=func_impl.__closure__
            )
            
            # 设置函数属性
            func.__doc__ = description
            func.__annotations__ = signature["annotations"]
            
            # 添加到工具字典
            self.tool_functions[func_name] = func
    
    def _create_function_signature(self, input_schema: Dict) -> Dict:
        """
        根据输入模式创建函数签名
        :param input_schema: 输入模式定义
        :return: 包含签名信息的字典
        """
        properties = input_schema["properties"]
        required_params = input_schema.get("required", [])
        
        # 创建参数列表和类型注解
        parameters = []
        annotations = {}
        defaults = []
        
        for param_name, param_def in properties.items():
            # 映射 JSON 类型到 Python 类型
            param_type = self._map_json_type_to_python(param_def["type"])
            
            # 添加类型注解
            annotations[param_name] = param_type
            
            # 如果是必需参数
            if param_name in required_params:
                parameters.append(param_name)
            else:
                # 非必需参数添加默认值
                parameters.append(param_name)
                defaults.append(None)  # 默认值为 None
        
        return {
            "parameters": parameters,
            "annotations": annotations,
            "defaults": tuple(defaults) if defaults else None
        }
    
    def _map_json_type_to_python(self, json_type: str) -> type:
        """将 JSON 类型映射到 Python 类型"""
        type_mapping = {
            "string": str,
            "number": float,
            "integer": int,
            "boolean": bool,
            "object": dict,
            "array": list,
            "null": type(None)
        }
        return type_mapping.get(json_type, Any)
    
    def _create_function_implementation(self, api_path: str, method: str, signature: Dict) -> Callable:
        """
        创建函数实现
        :param api_path: API 路径
        :param method: HTTP 方法
        :param signature: 函数签名
        :return: 可调用的函数
        """
        method = method.lower()
        params = signature["parameters"]
        
        # 创建函数定义字符串
        func_def = f"async def {signature.get('name', 'api_function')}({', '.join(params)}):\n"
        
        # 添加函数体 - 完整的 HTTP 请求实现
        # 注意：这里我们直接在函数体中生成 headers
        func_body = f"""
    # 准备请求参数
    payload = {{"""
        
        for param in params:
            func_body += f"""
        '{param}': {param},"""
        
        func_body += f"""
    }}
    
    # 准备认证头
    headers = {{}}
    username = os.getenv("IRIS_USERNAME")
    password = os.getenv("IRIS_PASSWORD")
    if username and password:
        credentials = f"{{username}}:{{password}}"
        encoded_credentials = base64.b64encode(credentials.encode('utf-8')).decode('utf-8')
        headers["Authorization"] = f"Basic {{encoded_credentials}}"
    
    # 发送 HTTP 请求
    async with httpx.AsyncClient() as client:
        try:
            response = await client.request(
                method="{method}",
                url="{api_path}",
                headers=headers,  # 直接传递字典变量
                json=payload,
                timeout=10.0
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {{
                "error": f"HTTP错误: {{e.response.status_code}}",
                "details": e.response.text
            }}
        except Exception as e:
            return {{
                "error": "请求失败",
                "details": str(e)
            }}
"""
        
        # 合并函数定义和函数体
        full_func_code = func_def + func_body
        
        # 执行函数代码
        local_vars = {}
        # 确保 os 模块在生成的函数中可用
        exec_globals = {"os": os, "base64": base64}
        exec(full_func_code, exec_globals, local_vars)
        
        return local_vars[signature.get('name', 'api_function')]
    
    def get_tool_functions(self) -> Dict[str, Callable]:
        """获取生成的工具函数字典"""
        return self.tool_functions
    
    def register_tools(self, mcp_server):
        """将生成的工具注册到 MCP 服务器"""
        for func_name, func in self.tool_functions.items():
            mcp_server._tool_manager.add_tool(func, name=func_name)
            print(f"已注册工具: {func_name}")

# 示例使用
if __name__ == "__main__":
    # 示例 API 元数据
    api_metadata = [
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
        },
        {
            "name": "multiplyNumbers",
            "description": "两个数相乘，返回两数之积。",
            "api_path": "http://localhost:52880/MCP/MCPTools/Math/multiply",
            "method": "post",
            "input_schema": {
                "type": "object",
                "title": "multiplyNumbersArguments",
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
    
    # 创建工具生成器
    generator = RESTAPIToolGenerator(api_metadata)
    
    # 获取生成的工具函数
    tools = generator.get_tool_functions()
    
    # 打印工具信息
    print("生成的工具函数:")
    for name, func in tools.items():
        print(f"\n函数名: {name}")
        print(f"描述: {func.__doc__}")
        print(f"签名: {inspect.signature(func)}")
        print(f"类型注解: {func.__annotations__}")
    
    # 设置环境变量用于测试
    os.environ["IRIS_USERNAME"] = "superuser"
    os.environ["IRIS_PASSWORD"] = "SYS"
    
    # 测试调用
    async def test_tools():
        print("\n测试工具调用:")
        
        # 调用加法工具
        add_result = await tools["addNumbers"](a=1, b=3)
        print(f"加法结果: {add_result}")
        
        # 调用乘法工具
        multiply_result = await tools["multiplyNumbers"](a=4, b=6)
        print(f"乘法结果: {multiply_result}")
    
    # 运行测试
    asyncio.run(test_tools())