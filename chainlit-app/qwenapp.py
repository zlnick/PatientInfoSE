import os
import time
from openai import AsyncOpenAI
from mcp import ClientSession
import chainlit as cl
from utils import parse_mcp_result
import json

client = AsyncOpenAI(
    api_key=os.getenv("Qwen_API_KEY"), base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

@cl.on_chat_start
def on_chat_start():
    cl.user_session.set("counter", 0)

@cl.on_mcp_connect
async def on_mcp_connect(connection, session: ClientSession):
    cl.logger.info(f"Chainlit 正在尝试连接 MCP Server: {connection.name}")
    try:
        result = await session.list_tools()
        tools = [{"name": t.name, "description": t.description, "input_schema": t.inputSchema} for t in result.tools]
        cl.user_session.set("mcp_tools", {connection.name: tools})
        cl.user_session.set("mcp_session", session)
        cl.logger.info(f"MCP 连接成功，已获取 {len(tools)} 个工具")
    except Exception as e:
        cl.logger.error(f"获取 MCP 工具列表时出错: {e}")

@cl.on_message
async def on_message(msg: cl.Message):
    start = time.time()
    mcp_tools = cl.user_session.get("mcp_tools")
    session = cl.user_session.get("mcp_session")
    
    process_steps = []

    if mcp_tools and session:
        tool_list = mcp_tools[list(mcp_tools.keys())[0]]
        # 构造包含输入字段说明的工具描述，防止 'type' 缺失
        tool_descriptions = ""
        for t in tool_list:
            input_schema = t.get('input_schema', {})
            if isinstance(input_schema, dict):
                input_fields = ", ".join([
                    f"{k}({(v.get('type') if isinstance(v, dict) else v) or '未知类型'})"
                    for k, v in input_schema.items()
                ]) if input_schema else "无输入字段"
            else:
                input_fields = "未知输入结构"
            tool_descriptions += f"- {t['name']}: {t['description']} | 输入字段: {input_fields}\n"
        process_steps.append(f"已连接到 MCP Server，发现工具：\n{tool_descriptions}")
    else:
        tool_descriptions = ""
        process_steps.append("未连接 MCP Server 或未加载任何工具")

    # 完善版提示词模板
    prompt = f"""
你是一个智能助手。用户提出的问题是："{msg.content}"。

以下是当前可用 MCP 工具列表，包括输入字段说明：
{tool_descriptions}

请遵循以下规则：
1️⃣ 若需要调用 MCP 工具，严格输出：
Tool: 工具名称（与上列表完全匹配）
Input: 工具输入参数（JSON格式，字段名严格对应工具定义）

例如：
Tool: add
Input: {{"a": 1, "b": 2}}

2️⃣ 若无需调用工具，请输出：
Answer: 直接回答内容

切记：
✅ 工具名称和字段必须严格匹配
✅ 不要生成无关字段，如 num1/num2 等
✅ 如果没有合适工具，请直接输出 Answer
"""

    process_steps.append("提示词构造完毕，开始调用 Qwen 推理...")

    completion = await client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "你是一个智能助手"},
            {"role": "user", "content": prompt},
        ]
    )

    response = completion.choices[0].message.content
    process_steps.append(f"Qwen 返回推理结果：\n{response}")

    # 分析推理结果
    if "Tool:" in response:
        tool_name = response.split("Tool:")[1].splitlines()[0].strip()
        input_str = response.split("Input:")[1].strip()
        try:
            tool_input = json.loads(input_str)
            process_steps.append(f"解析到工具调用计划：\n工具: {tool_name}\n输入: {tool_input}")
            if session:
                try:
                    result = await session.call_tool(tool_name, tool_input)
                    parsed = parse_mcp_result(result)
                    process_steps.append(f"调用 MCP 工具 {tool_name} 成功，解析结果:")
                    for content_type, content_value in parsed:
                        process_steps.append(f"类型: {content_type}, 内容: {content_value}")
                        if content_type == "text":
                            await cl.Message(content=content_value).send()
                        elif content_type == "file":
                            await cl.Message(content=f"文件链接: {content_value}").send()
                        elif content_type == "image":
                            await cl.Message(content="收到图片:").send()
                            await cl.Message(image=content_value).send()
                        elif content_type == "error":
                            await cl.Message(content=f"❌ {content_value}").send()
                        else:
                            await cl.Message(content=f"其他内容: {content_value}").send()
                except Exception as e:
                    process_steps.append(f"调用 MCP 工具失败: {e}")
                    await cl.Message(content=f"调用 MCP 工具失败: {e}").send()
            else:
                process_steps.append("未连接 MCP Server，无法调用工具。")
                await cl.Message(content="未连接 MCP Server，无法调用工具。").send()
        except Exception as e:
            process_steps.append(f"输入解析失败: {e}")
            await cl.Message(content=f"工具输入解析失败: {e}").send()
    elif "Answer:" in response:
        answer = response.split("Answer:")[1].strip()
        process_steps.append(f"Qwen 判断无需调用工具，直接回答: {answer}")
        await cl.Message(content=answer).send()
    else:
        process_steps.append("Qwen 返回无法识别，使用默认大模型回答...")
        fallback_completion = await client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": "你是一个智能助手"},
                {"role": "user", "content": msg.content},
            ]
        )
        fallback_response = fallback_completion.choices[0].message.content
        process_steps.append(f"默认大模型回答: {fallback_response}")
        await cl.Message(content=fallback_response).send()

    reasoning_output = "\n".join(process_steps)
    await cl.Message(content=f"📝 推理与调用过程:\n{reasoning_output}").send()

    counter = cl.user_session.get("counter")
    counter += 1
    cl.user_session.set("counter", counter)
    await cl.Message(content=f"你已经发送了 {counter} 条消息！").send()
