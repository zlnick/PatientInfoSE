import json
import os
from typing import Dict, List, Any

def convert_swagger_type_to_json_schema_type(swagger_type: str, swagger_format: str = None) -> str:
    """将 Swagger 类型转换为 JSON Schema 类型"""
    type_mapping = {
        "integer": "integer",
        "long": "integer",
        "float": "number",
        "double": "number",
        "string": "string",
        "byte": "string",
        "binary": "string",
        "boolean": "boolean",
        "date": "string",
        "date-time": "string",
        "password": "string",
        "object": "object",
        "array": "array"
    }
    
    # 特殊处理格式
    if swagger_format:
        if swagger_format == "int32" or swagger_format == "int64":
            return "integer"
        elif swagger_format == "float" or swagger_format == "double":
            return "number"
    
    return type_mapping.get(swagger_type, "object")

def resolve_ref(spec: Dict, ref_path: str) -> Dict:
    """解析 $ref 引用"""
    if not ref_path.startswith("#/"):
        return {}
    
    parts = ref_path[2:].split("/")  # 去掉开头的 "#/"
    current = spec
    for part in parts:
        current = current.get(part, {})
    return current

def process_schema(spec: Dict, schema: Dict) -> Dict:
    """递归处理 schema 及其引用"""
    if not schema:
        return {}
    
    # 处理引用
    if "$ref" in schema:
        ref_schema = resolve_ref(spec, schema["$ref"])
        return process_schema(spec, ref_schema)
    
    # 处理数组类型
    if schema.get("type") == "array":
        items = schema.get("items", {})
        return {
            "type": "array",
            "items": process_schema(spec, items)
        }
    
    # 处理对象类型
    if schema.get("type") == "object" or "properties" in schema:
        properties = {}
        for prop_name, prop_def in schema.get("properties", {}).items():
            properties[prop_name] = process_schema(spec, prop_def)
        
        return {
            "type": "object",
            "properties": properties,
            "required": schema.get("required", [])
        }
    
    # 基本类型
    return {
        "type": convert_swagger_type_to_json_schema_type(
            schema.get("type", "object"),
            schema.get("format")
        ),
        "title": schema.get("title", ""),
        "description": schema.get("description", "")
    }

def generate_tool_list(openapi_spec: Dict) -> List[Dict]:
    """从 OpenAPI 2.0 规范生成工具列表"""
    tools = []

    definitions = openapi_spec.get("definitions", {})
    # 这里还有点问题，经IRIS注册后，host会自动变成域名+端口形态，对于没有进行域名绑定的服务器来说是无效的，暂时写死
    host = "localhost:52880"
    # openapi_spec.get("host", {})
    basePath = openapi_spec.get("basePath", {})
    schemes = openapi_spec.get("schemes", {})
    # 遍历所有路径和方法
    for path, path_item in openapi_spec.get("paths", {}).items():
        for method, operation in path_item.items():
            if method.lower() not in ["get", "post", "put", "delete", "patch"]:
                continue
                
            # 获取操作信息
            operation_id = operation.get("operationId", f"{method}_{path.replace('/', '_')}")
            api_path = schemes[0]+"://"+host+basePath+path
            description = operation.get("description", operation.get("summary", ""))
            
            # 准备输入模式
            properties = {}
            required = []
            schema = None
            
            # 处理参数
            for param in operation.get("parameters", []):
                param_in = param.get("in")
                
                # 只处理 body 参数（对于 POST/PUT/PATCH）
                if param_in == "body":
                    if "schema" in param:
                        schema = process_schema(openapi_spec, param["schema"])
                        properties = schema.get("properties", {})
                        required = schema.get("required", [])
                    break
            
            # 如果没有 body 参数，尝试从 operation 获取
            if not schema:
                if "requestBody" in operation:
                    content = operation["requestBody"].get("content", {})
                    for content_type, media_type in content.items():
                        if "schema" in media_type:
                            schema = process_schema(openapi_spec, media_type["schema"])
                            properties = schema.get("properties", {})
                            required = schema.get("required", [])
                            break
            
            # 为操作创建输入模式
            input_schema = {
                "type": "object",
                "title": f"{operation_id}Arguments",
                "properties": properties,
                "required": required
            }
            
            # 添加到工具列表
            tools.append({
                "name": operation_id,
                "description": description.strip(),
                "api_path": api_path,
                "method": method.lower(),
                "input_schema": input_schema
            })
    
    return tools

# 示例使用
if __name__ == "__main__":
    # 加载 OpenAPI 规范
    file_path = "MCPToolsAPI.json"
    
    try:
        # 明确指定 UTF-8 编码打开文件
        with open(file_path, "r", encoding="utf-8") as f:
            openapi_spec = json.load(f)
        
        # 生成工具列表
        tool_list = generate_tool_list(openapi_spec)
        
        # 打印结果
        print(json.dumps(tool_list, indent=2, ensure_ascii=False))
    
    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 不存在")
    except json.JSONDecodeError as e:
        print(f"JSON 解析错误: {e}")
    except Exception as e:
        print(f"发生错误: {e}")