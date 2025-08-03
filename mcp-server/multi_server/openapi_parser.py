import re
import json
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
    
    if swagger_format and swagger_format in type_mapping:
        return type_mapping[swagger_format]
    
    return type_mapping.get(swagger_type, "object")

def resolve_ref(spec: Dict, ref_path: str) -> Dict:
    """解析 $ref 引用"""
    if not ref_path.startswith("#/"):
        return {}
    
    parts = ref_path[2:].split("/")
    current = spec
    for part in parts:
        current = current.get(part, {})
    return current

def extract_parameters(operation: Dict, spec: Dict) -> Dict:
    """
    提取所有参数（路径、查询、body）
    返回格式: {
        "path_params": {param_name: param_schema},
        "query_params": {param_name: param_schema},
        "body_params": {param_name: param_schema},
        "required": [param_name]
    }
    """
    parameters = {
        "path_params": {},
        "query_params": {},
        "body_params": {},
        "required": []
    }
    
    for param in operation.get("parameters", []):
        param_in = param.get("in")
        param_name = param.get("name")
        required = param.get("required", False)
        
        if required:
            parameters["required"].append(param_name)
        
        # 获取参数模式
        schema = param.get("schema", {})
        if "$ref" in schema:
            schema = resolve_ref(spec, schema["$ref"])
        
        if not schema:
            # 简单参数
            schema = {
                "type": param.get("type", "string"),
                "format": param.get("format"),
                "description": param.get("description", "")
            }
        
        # 根据位置分类
        if param_in == "path":
            parameters["path_params"][param_name] = schema
        elif param_in == "query":
            parameters["query_params"][param_name] = schema
        elif param_in == "body":
            # 处理 body 参数（可能是嵌套对象）
            if "properties" in schema:
                for prop_name, prop_def in schema["properties"].items():
                    if "$ref" in prop_def:
                        ref_def = resolve_ref(spec, prop_def["$ref"])
                        parameters["body_params"][prop_name] = ref_def
                    else:
                        parameters["body_params"][prop_name] = prop_def
                
                # 添加必填字段
                if "required" in schema:
                    parameters["required"].extend(schema["required"])
    
    return parameters

def generate_tool_list(openapi_spec: Dict) -> List[Dict]:
    """从 OpenAPI 2.0 规范生成工具列表（修复路径参数问题）"""
    tools = []
    definitions = openapi_spec.get("definitions", {})
    base_url = f"{openapi_spec['schemes'][0]}://{openapi_spec['host']}{openapi_spec['basePath']}"
    
    # 遍历所有路径和方法
    for path, path_item in openapi_spec.get("paths", {}).items():
        for method, operation in path_item.items():
            if method.lower() not in ["get", "post", "put", "delete", "patch"]:
                continue
                
            # 获取操作信息
            operation_id = operation.get("operationId", f"{method}_{path.replace('/', '_')}")
            description = operation.get("description", operation.get("summary", ""))
            
            # 提取所有参数
            params_info = extract_parameters(operation, openapi_spec)
            
            # 创建统一的属性集合
            properties = {}
            properties.update(params_info["path_params"])
            properties.update(params_info["query_params"])
            properties.update(params_info["body_params"])
            
            # 为属性添加类型信息
            for prop_name, prop_def in properties.items():
                prop_type = convert_swagger_type_to_json_schema_type(
                    prop_def.get("type", "string"),
                    prop_def.get("format")
                )
                properties[prop_name] = {
                    "type": prop_type,
                    "description": prop_def.get("description", "")
                }
            
            # 创建输入模式
            input_schema = {
                "type": "object",
                "title": f"{operation_id}Arguments",
                "properties": properties,
                "required": params_info["required"]
            }
            
            # 完整的 API 路径
            api_path = f"{base_url}{path}"
            
            # 添加到工具列表
            tools.append({
                "name": operation_id,
                "description": description.strip(),
                "api_path": api_path,
                "method": method.lower(),
                "input_schema": input_schema,
                # 额外信息用于请求构造
                "path_params": list(params_info["path_params"].keys()),
                "query_params": list(params_info["query_params"].keys())
            })
    
    return tools