import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
import asyncio
import json

load_dotenv()

# 初始化大模型客户端
client = AsyncOpenAI(
    api_key=os.getenv("Qwen_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

async def generate_plan(history, user_input, tools):
    """
    自动生成执行计划，包含多步tool和llm_answer。
    返回JSON: { plan: [step1, step2, ...], explanation: "" }
    """
    # 组织tools描述
    tool_list_str = ""
    for t in tools:
        input_schema = t.get('input_schema', {})
        input_fields = ", ".join(
            [f"{k}({(v.get('type') if isinstance(v, dict) else v) or '未知类型'})" for k, v in input_schema.items()]
        ) if isinstance(input_schema, dict) and input_schema else "无输入字段"
        tool_list_str += f"- {t['name']}: {t['description']} | 输入字段: {input_fields}\n"

    # 多轮历史
    history_str = ""
    if history:
        for msg in history:
            history_str += f"{msg['role']}: {msg['content']}\n"

    prompt = f"""
你是一个多步计划Agent，请根据用户历史与当前问题，结合可用工具，规划详细执行计划。

对话历史（如有）：
{history_str}

用户最新问题："{user_input}"

可用工具：
{tool_list_str}

**要求**：
1. 若能用工具解决任何子任务，必须优先用tool。否则用LLM自身作答，类型为llm_answer。
2. plan须输出为JSON数组，每步如下格式：
   {{
     "action": "call_tool" 或 "llm_answer",
     "tool": 工具名 (如action为call_tool时填写，否则为null),
     "input": 输入参数 (dict，字段名与工具定义严格一致。llm_answer时为字符串),
     "result_var": 本步结果变量名（如有依赖，用$var名引用）,
     "description": "本步意图说明"
   }}
3. 多步plan间如有依赖，用result_var实现数据流。
4. 最终输出格式：
{{
  "plan": [step1, step2, ...],
  "explanation": "简要说明你的计划拆解思路"
}}
仅输出严格JSON格式plan和explanation，不要输出其他内容。
"""

    completion = await client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "你是一个计划生成Agent，只负责输出JSON结构计划"},
            {"role": "user", "content": prompt},
        ]
    )
    # 抽取完整的JSON体
    response = completion.choices[0].message.content
    # 提取json（防止模型外包一层说明，可用正则或手动strip）
    try:
        # 尽可能找第一个大括号及其后内容
        json_start = response.find('{')
        json_obj = json.loads(response[json_start:])
        return json_obj
    except Exception as e:
        return {
            "raw": response,
            "plan": [],
            "explanation": f"解析LLM计划输出失败: {e}, 原始输出: {response}"
        }

# =========== 用法举例 ===========
if __name__ == "__main__":
    # 假定从Chainlit侧收到：
    #history = [
    #    {"role": "user", "content": "先把2和3加起来，再和5相乘"},
    #]
    history = []
    user_input = "请帮我算(2+3)*(5+7)"
    tools = [
        {
            "name": "add",
            "description": "两个数字相加",
            "input_schema": {"a": {"type": "int"}, "b": {"type": "int"}}
        },
        {
            "name": "multiply",
            "description": "两个数字相乘",
            "input_schema": {"a": {"type": "int"}, "b": {"type": "int"}}
        }
    ]
    plan = asyncio.run(generate_plan(history, user_input, tools))
    print(json.dumps(plan, ensure_ascii=False, indent=2))
