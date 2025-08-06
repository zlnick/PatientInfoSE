from dotenv import load_dotenv
import os
import uuid
from openai import AsyncOpenAI
from mcp import ClientSession
import chainlit as cl
import json
from utils import parse_mcp_result,get_result_value,get_practitioner,get_official_name,get_table_meta
from context_manager import IRISContextManager
from planner_agent import generate_plan
from context_aware_agent import can_answer_from_context, generate_context_answer

load_dotenv()
prac_id = os.getenv("Practioner_ID")
practioner = get_practitioner(prac_id)
prac_name = get_official_name(practioner)
table_meta = get_table_meta(os.getenv("TABLE_META_ENDPOINT"),os.getenv("TABLE_NS"),os.getenv("TABLE_SCHEME"))
assistant_name = os.getenv("Assistant_NAME")
assistant_prompt = f"""
你是一个临床医生的门诊助手。你将用医生容易阅读的自然语言和医生用中文交流。不要向医生返回对FHIR、SQL表等数据的技术信息，你将会将这些信息用自然语言描述后再向医生反馈。
你非常了解HL7 FHIR协议，知道资源id指的是id参数。
你还知道如下临床知识：
血压的字典码是85354-9。
当接收到一批同一患者的数据时，除了向医生描述患者数据，还应从临床角度进行总结。
"""

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
    api_key=os.getenv("Qwen_API_KEY"), 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

@cl.on_chat_start
async def on_chat_start():
    # 每次新会话分配一个session_id并创建doc
    session_id = str(uuid.uuid4())
    cl.user_session.set("session_id", session_id)
    cl.user_session.set("counter", 0)
    cl.user_session.set("temp_values",{})
    cl.logger.info(f"医生{prac_id}的名字是{prac_name}")
    #cl.user_session.set("practitioner",)
    cl.logger.info(f"新会话分配 session_id: {session_id}")
    await cl.Message(
        content=f"欢迎您，{prac_name}医生，我是您的门诊助手{assistant_name}。我会协助您完成门诊，欢迎您向我提出任何问题。"
    ).send()

@cl.on_mcp_connect
async def on_mcp_connect(connection, session: ClientSession):
    cl.logger.info(f"Chainlit 正在尝试连接 MCP Server: {connection.name}")
    try:
        result = await session.list_tools()
        tools = [{"name": t.name, "description": t.description, "input_schema": t.inputSchema} for t in result.tools]
        cl.user_session.set("mcp_tools", {connection.name: tools})
        cl.user_session.set("mcp_session", session)
        cl.logger.info(f"MCP 连接成功，已获取 {len(tools)} 个工具")
        #cl.logger.info(tools)
    except Exception as e:
        cl.logger.error(f"获取 MCP 工具列表时出错: {e}")


def get_or_create_session(cl, ctx):
    session_id = cl.user_session.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        cl.user_session.set("session_id", session_id)
    # 判断IRIS是否已有此session，没有才创建
    if ctx.get_session(session_id) is None:
        ctx.create_session(session_id)  # meta参数可以省略
        # 创建session时绑定医生身份
        save_assistant_message(ctx,session_id,f"我是一个临床医生的门诊助手。现在登录的临床医生的资源id是{prac_id},他的姓名是{prac_name}。我们用中文交流。")
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

async def call_mcp_tool(session, tool_name, tool_input):
    try:
        cl.logger.info(f"基于输入 {tool_input} 调用工具 {tool_name} ")
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
    #save_user_message(ctx, session_id, msg)
    history_str, history = get_history_str(ctx, session_id)

    # === 上下文优先判断 ===
    # 先判断上下文中的数据是否足够回答问题
    can_answer, reasoning = await can_answer_from_context(history, msg.content, client)
    #cl.logger.info(f"历史信息1： {history_str}")
    #cl.logger.info(f"历史信息2： {history}")
    #cl.logger.info(f"判断状态： {can_answer}, 原因是: {reasoning}")
    save_user_message(ctx, session_id, msg)
    # 用Step显示判断结果    
    async with cl.Step("判断结果显示") as step:
        step.output = f"🧠 上下文判断: {'可以' if can_answer else '不可以'}直接回答\n"+ f"📝 判断理由: {reasoning}"

    if can_answer:
        # 直接基于上下文生成回答
        answer = await generate_context_answer(history, msg.content, client)
        
        # 保存并返回回答
        save_assistant_message(ctx, session_id, answer)
        await cl.Message(content=answer).send()
        return  # 结束处理，不再执行后续工具调用

    mcp_tools = cl.user_session.get("mcp_tools")
    session = cl.user_session.get("mcp_session")
    # === 调用 planner_agent 生成 plan ===
    # 提取原始tools
    tool_list = []
    if mcp_tools:
        for v in mcp_tools.values():
            tool_list.extend(v)

    # 生成多步 plan
    plan_json = await generate_plan(history, msg.content, tool_list,client)
    plan = plan_json.get("plan", [])
    explanation = plan_json.get("explanation", "")

    process_steps = [f"多步执行计划：{json.dumps(plan, ensure_ascii=False, indent=2)}", f"计划说明：{explanation}"]

    answer_texts = []

    temp_values = cl.user_session.get("temp_values")

    for idx, step in enumerate(plan):
        action = step.get("action")
        tool = step.get("tool")
        input_data = step.get("input")
        result_var = step.get("result_var")
        step_desc = step.get("description", "")
        process_steps.append(f"Step {idx+1}: {step_desc}")
        #cl.logger.info(input_data)
        # 如果input_data是Dict类型，则表明其中包含了上下文中保存的临时变量，需从temp_values中提取对应的临时变量替换其值
        if isinstance(input_data, dict):
            for key in input_data:
                    if isinstance(input_data[key], str) and input_data[key] in temp_values:
                        input_data[key] = temp_values[input_data[key]]
        #cl.logger.info(temp_values)
        #cl.logger.info(input_data)
        if action == "call_tool" and tool and session:
            # 调用 MCP 工具
            try:
                raw_result = await session.call_tool(tool, input_data)
                parsed_contents = parse_mcp_result(raw_result)
                #cl.logger.info(parsed_contents)
                # 如果result_var中标识出了临时变量的名称，则将parsed_contents作为临时变量值赋给这个临时变量
                if result_var:
                    temp_values[result_var]=get_result_value(parsed_contents)
                for content_type, content_value in parsed_contents:
                    if content_type == "text":
                        answer_texts.append(content_value)
                        #打印工具返回结果
                        async with cl.Step(name="工具返回结果") as step:
                            step.output = content_value
                        #await cl.Message(content=f"[{tool}] {content_value}").send()
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
            #cl.logger.info(temp_values)
            llm_prompt = input_data if isinstance(input_data, str) else str(input_data)
            #cl.logger.info(input_data)
            completion = await client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {"role": "system", "content": assistant_prompt},
                    {"role": "user", "content": llm_prompt},
                ],
                temperature=0.01
            )
            llm_reply = completion.choices[0].message.content
            answer_texts.append(llm_reply)
            await cl.Message(content=llm_reply).send()
        else:
            # 计划格式不对
            process_steps.append(f"无法识别的计划类型：{step}")

    # 汇总本轮对话内容并保存在IRIS中
    answer = "\n".join(answer_texts)
    if answer:
        save_assistant_message(ctx, session_id, answer)
    counter = cl.user_session.get("counter")
    counter += 1
    cl.user_session.set("counter", counter)
    process_steps.append(f"你已经发送了 {counter} 条消息！")
    #await cl.Message(content="📝 本轮多步推理/执行过程：\n" + "\n".join(process_steps)).send()


