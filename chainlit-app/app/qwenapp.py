from http import HTTPStatus
import re
from dotenv import load_dotenv
import os
import uuid
import time
from openai import AsyncOpenAI
from mcp import ClientSession
import chainlit as cl
import json
from utils import parse_mcp_result
from context_manager import IRISContextManager
from planner_agent import generate_plan
import asyncio
import requests
import dashscope
from chainlit.element import ElementBased
from dashscope.audio.tts_v2 import SpeechSynthesizer
from dashscope.common.constants import FilePurpose
from io import BytesIO
import tempfile


load_dotenv()

# 初始化IRIS上下文管理器
ctx = IRISContextManager(
    host=os.getenv("IRIS_HOSTNAME"),
    port=int(os.getenv("IRIS_PORT")),
    namespace=os.getenv("IRIS_NAMESPACE"),
    username=os.getenv("IRIS_USERNAME"),
    password=os.getenv("IRIS_PASSWORD"),
)

# LLM客户端
client = AsyncOpenAI(
    api_key=os.getenv("Qwen_API_KEY"), base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

@cl.on_chat_start
def on_chat_start():
    # 每次新会话分配一个session_id并创建doc
    session_id = str(uuid.uuid4())
    cl.user_session.set("session_id", session_id)
    cl.user_session.set("counter", 0)
    cl.logger.info(f"新会话分配 session_id: {session_id}")

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

# context_manager和其他import省略

def get_or_create_session(cl, ctx):
    session_id = cl.user_session.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        cl.user_session.set("session_id", session_id)
    # 判断IRIS是否已有此session，没有才创建
    if ctx.get_session(session_id) is None:
        ctx.create_session(session_id)  # meta参数可以省略
    return session_id

def save_user_message(ctx, session_id, msg):
    ctx.append_history(session_id, "user", msg.content)

def get_history_str(ctx, session_id):
    history = ctx.get_history(session_id)
    return "\n".join([f"{item['role']}: {item['content']}" for item in history]), history

def build_tool_descriptions(mcp_tools):
    if not mcp_tools:
        return ""
    tool_list = list(mcp_tools.values())[0]
    descriptions = []
    for t in tool_list:
        input_schema = t.get('input_schema', {})
        if isinstance(input_schema, dict):
            input_fields = ", ".join([
                f"{k}({(v.get('type') if isinstance(v, dict) else v) or '未知类型'})"
                for k, v in input_schema.items()
            ]) if input_schema else "无输入字段"
        else:
            input_fields = "未知输入结构"
        descriptions.append(f"- {t['name']}: {t['description']} | 输入字段: {input_fields}")
    return "\n".join(descriptions)

def compose_prompt(history_str, msg_content, tool_descriptions):
    return f"""
你是一个智能助手。以下是本轮对话历史：
{history_str}

用户最新问题是："{msg_content}"

以下是当前可用 MCP 工具列表，包括输入字段说明：
{tool_descriptions}

请遵循以下规则：
1️⃣ 若需要调用 MCP 工具，严格输出：
Tool: 工具名称（与上列表完全匹配）
Input: 工具输入参数（JSON格式，字段名严格对应工具定义）

2️⃣ 若无需调用工具，请输出：
Answer: 直接回答内容

切记：
✅ 工具名称和字段必须严格匹配
✅ 不要生成无关字段
✅ 如果没有合适工具，请直接输出 Answer
"""

async def call_mcp_tool(session, tool_name, tool_input):
    try:
        result = await session.call_tool(tool_name, tool_input)
        return str(result)
    except Exception as e:
        return f"调用 MCP 工具失败: {e}"

def save_assistant_message(ctx, session_id, answer):
    ctx.append_history(session_id, "assistant", answer)

async def send_messages(cl, answer, reasoning_output, counter):
    await cl.Message(content=answer).send()
    await cl.Message(content=f"📝 推理与调用过程:\n{reasoning_output}").send()
    await cl.Message(content=f"你已经发送了 {counter} 条消息！").send()

# ===================
# 主 on_message 方法
# ===================

@cl.on_message
async def on_message(msg: cl.Message):
    session_id = get_or_create_session(cl, ctx)
    save_user_message(ctx, session_id, msg)
    history_str, history = get_history_str(ctx, session_id)

    mcp_tools = cl.user_session.get("mcp_tools")
    session = cl.user_session.get("mcp_session")

    # === 新增：调用 planner_agent 生成 plan ===
    # 提取原始tools
    tool_list = []
    if mcp_tools:
        for v in mcp_tools.values():
            tool_list.extend(v)

    # 生成多步 plan
    plan_json = await generate_plan(history, msg.content, tool_list)
    #cl.logger.error(plan_json)
    plan = plan_json.get("plan", [])
    explanation = plan_json.get("explanation", "")

    process_steps = [f"多步执行计划：{json.dumps(plan, ensure_ascii=False, indent=2)}", f"计划说明：{explanation}"]

    answer_texts = []

    for idx, step in enumerate(plan):
        action = step.get("action")
        tool = step.get("tool")
        input_data = step.get("input")
        result_var = step.get("result_var")
        step_desc = step.get("description", "")

        process_steps.append(f"Step {idx+1}: {step_desc}")

        if action == "call_tool" and tool and session:
            # 调用 MCP 工具
            try:
                raw_result = await session.call_tool(tool, input_data)
                parsed_contents = parse_mcp_result(raw_result)
                for content_type, content_value in parsed_contents:
                    if content_type == "text":
                        answer_texts.append(content_value)
                        await cl.Message(content=f"[{tool}] {content_value}").send()
                    elif content_type == "file":
                        await cl.Message(content=f"文件链接: {content_value}").send()
                    elif content_type == "image":
                        await cl.Message(content="收到图片:").send()
                        await cl.Message(image=content_value).send()
                    elif content_type == "error":
                        answer_texts.append(f"❌ {content_value}")
                        await cl.Message(content=f"❌ {content_value}").send()
                    else:
                        await cl.Message(content=f"其他内容: {content_value}").send()
            except Exception as e:
                err_msg = f"调用工具 {tool} 失败: {e}"
                answer_texts.append(err_msg)
                await cl.Message(content=err_msg).send()
        elif action == "llm_answer":
            # 直接用LLM自身能力生成回答
            llm_prompt = input_data if isinstance(input_data, str) else str(input_data)
            completion = await client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {"role": "system", "content": "你是一个智能助手"},
                    {"role": "user", "content": llm_prompt},
                ]
            )
            llm_reply = completion.choices[0].message.content
            answer_texts.append(llm_reply)
            await cl.Message(content=llm_reply).send()
        else:
            # 计划格式不对
            process_steps.append(f"无法识别的计划类型：{step}")

    # 汇总本轮对话内容落IRIS
    answer = "\n".join(answer_texts)
    if answer:
        save_assistant_message(ctx, session_id, answer)
    counter = cl.user_session.get("counter")
    counter += 1
    cl.user_session.set("counter", counter)
    process_steps.append(f"你已经发送了 {counter} 条消息！")
    await cl.Message(content="📝 本轮多步推理/执行过程：\n" + "\n".join(process_steps)).send()


async def speech_to_text(audio_file):
    result = dashscope.Files.upload(file_path=audio_file,
                                    purpose=FilePurpose.assistants)
    file_id = result.output['uploaded_files'][0]['file_id']
    file_res = dashscope.Files.get(file_id)
    task_response = dashscope.audio.asr.Transcription.async_call(
        model='sensevoice-v1',
        file_urls=[file_res.output['url']],
        language_hints=['zh', 'en'],
    )
    transcribe_response = dashscope.audio.asr.Transcription.wait(task=task_response.output.task_id)
    if transcribe_response.status_code == HTTPStatus.OK:
        transcription_url = transcribe_response.output["results"][0]["transcription_url"]
        # 发送GET请求
        response = requests.get(transcription_url)
        # 检查请求是否成功
        if response.status_code == 200:
            # 使用内置的json()方法将响应体解析为字典
            data = response.json()
            print(data)
            print(data['transcripts'][0]['text'])
            text = data['transcripts'][0]['text']
            # 使用正则表达式提取标签之间的文本
            pattern = r'<\|Speech\|>(.*?)<\|/Speech\|>'
            match = re.search(pattern, text, re.DOTALL)

            if match:
                text = match.group(1).strip()
            else:
                text = ''
            return text
        else:
            print(f"Failed to retrieve data: {response.status_code}")
    return "fail"

@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk):
    if chunk.isStart:
        buffer = BytesIO()
        buffer.name = f"input_audio.{chunk.mimeType.split('/')[1]}"
        # Initialize the session for a new audio stream
        cl.user_session.set("audio_buffer", buffer)
        cl.user_session.set("audio_mime_type", chunk.mimeType)

    # For now, write the chunks to a buffer and transcribe the whole audio at the end
    cl.user_session.get("audio_buffer").write(chunk.data)


@cl.on_audio_end
async def on_audio_end(elements: list[ElementBased]):
    # Get the audio buffer from the session
    audio_buffer: BytesIO = cl.user_session.get("audio_buffer")
    audio_mime_type: str = cl.user_session.get("audio_mime_type")
    audio_buffer.seek(0)  # 将文件指针移到开头
    # 使用pydub处理音频
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmpFile:
            tmpFile.write(audio_buffer.read())
            tmpFile_path = tmpFile.name
    except Exception as e:
        print(f"Error processing audio: {e}")

    print('tmpFile_path', tmpFile_path)
    transcription = await speech_to_text(tmpFile_path)
    input_audio_el = cl.Audio(
        mime=audio_mime_type, path=tmpFile_path, name="",
    )
    message = await cl.Message(
        author="You",
        type="user_message",
        content=transcription,
        elements=[input_audio_el, *elements]
    ).send()
    print('transcription', transcription)
    
