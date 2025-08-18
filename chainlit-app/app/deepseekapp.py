from dotenv import load_dotenv
import os
import uuid
from openai import AsyncOpenAI
from mcp import ClientSession
import chainlit as cl
import json
from utils import parse_mcp_result,get_result_value,get_practitioner,get_official_name
from context_manager import IRISContextManager
from planner_agent import generate_plan
from context_aware_agent import can_answer_from_context, generate_context_answer
from data_visualization_agent import generate_interactive_plotly_chart
import audioop
import numpy as np
import io
import wave
import dashscope
from dashscope import MultiModalConversation
import base64

load_dotenv()

prac_id = os.getenv("Practioner_ID")
practioner = get_practitioner(prac_id)
prac_name = get_official_name(practioner)
assistant_name = os.getenv("Assistant_NAME")
llm_model = os.getenv("LLM_MODEL")

#临床助手Prompt
assistant_prompt = f"""
你是一个临床医生的门诊助手。你将用医生容易阅读的自然语言和医生用中文交流。不要向医生返回对FHIR、SQL表等数据的技术信息，你将会将这些信息用自然语言描述后再向医生反馈。
你非常了解HL7 FHIR协议，知道id或资源id指的是id参数。
你还知道如下临床知识：
血压的字典码是85354-9。
当接收到一批同一患者的数据时，除了向医生描述患者数据，还应从临床角度进行总结。
当医生试图为患者开药或询问药物是否适用时，你应当先检查上下文，查看是否已通过FHIR Patient资源的$everything操作获得了患者的所有信息。如果有，则结合患者信息和药物信息回答问题；如果没有，则先通过$everything操作获取患者的完整档案，再结合药品信息回答问题。
"""

#药物医保报销风险检测助手 Prompt
insurance_expert_prompt = f"""
基于上下文信息中的以下信息回答医保拒付风险相关的问题。上下文中包含：
1. 第一个元素的信息通常为患者信息，应该为通过FHIR的$everything操作获得的完整档案，但你在描述信息时不要提到FHIR，只说是患者档案中的信息即可。
2. 之后的信息为通过知识库查询获得的，与问题中提到的药物最相关的医保规则信息

遵循原则:
1. 要依据获取到的医保规则仔细分析，确认报销约束是不是都得到了满足。
    例如盐酸右美托咪定的报销条件为：成人术前镇静/抗焦虑，则只有患者为成人且有手术医嘱且术前有类似焦虑的诊断或并病程记录才算满足条件。
    又如溴芬酸钠的报销条件为：限眼部手术后炎症。如果患者信息中没有眼部手术的记录或没有眼部手术后炎症的信息，则未满足报销条件。
2. 对于一种被问到的药物，如果知识库中没有明确的医保报销约束，则在规则中说明知识库中没有报销规则。但你需要根据医保的普通原则：
——————————————————
医保支付政策通常包括：
类型	说明
无医保限制	可报销，但必须符合临床适应症
限适应症报销	若无明确感染证据，即使无限制，也可能被认定为“不合理用药”而拒付
限二线用药	必须在一线抗生素无效或禁忌时使用，否则可能拒付
限特定人群	如仅限儿童、孕妇、HIV患者等
——————————————————
判断有没有拒付风险。

结合患者的实际病情判断用药的适应症是否存在。如果患者病情中存在适应症，则没有医保拒付风险；如果患者病情中未发现适应症，则必须判定为有医保拒付风险。
3. 对于二线用药，如果患者病情中没有一线用药记录或没有一线用药无效的记录，也应判断为没有适应症。在解释部分应该说明如果要用这种二线药物，对应的一线药物是什么（举例）。
4. 只要有一个药物存在拒付风险，则整体来看就有拒付风险。

回答时要简洁，按顺序包括三个方面内容：
1. 结论：根据当前上下文中的信息明确回答是否有医保拒付风险，只回答有没有风险即可。
2. 规则：针对问题中被询问的药物，而不是医保规则信息中的药物，逐条解释医保规则是什么样的。在说明医保规则时应严格引用上下文中记录的规则文本，不要创造内容。如果上下文中没有返回对应的药物的信息，可以当作没有特定的医保规则，不要列出其它药物的医保规则。禁止列出没被问到的药品的信息。l
3. 解释：针对问题中被问到的每一种药物，如果有拒付风险，说明原因。如果没有拒付风险，说明支撑条件是什么。不要列出没有被问到的药物的信息。
"""

# 定义一个用于检测静音的阈值和一个用于结束一轮（对话或交互）的超时时间。
SILENCE_THRESHOLD = (
    2000  # Adjust based on your audio level (e.g., lower for quieter audio)
)
SILENCE_TIMEOUT = 2000.0  # Seconds of silence to consider the turn finished

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
    api_key=os.getenv("DEEPSEEK_API_KEY"), 
    base_url="https://api.deepseek.com/v1"
)

# 音频理解（Qwen-Audio）客户端（由于Qwen不兼容OpenAI音频接口，需额外构建客户端）
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')
#.AsyncClient()

# 各类工具函数
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


# 语音转文字
async def speech_to_text(audio_buffer):
    base64_audio = base64.b64encode(audio_buffer).decode('utf-8')
    messages = [
            {
                "role": "user",
                "content": [
                    # 音频内容，使用base64编码
                    {"audio": f"data:audio/wav;base64,{base64_audio}"},
                    {"text": "请将这段语音转换为中文文字，仅输出转换后的内容，不添加任何前缀、后缀或解释说明，直接返回原始文本。"}
                ]
            }
        ]
    try:
            # 调用通意千问的音频模型
            response = MultiModalConversation.call(
                model='qwen2-audio-instruct',  # 使用音频专用模型
                messages=messages
            )
            # 处理API响应
            if response.status_code == 200 and response.output:
                content = response.output.choices[0].message.content
                print(f"语音转文字结果: {content}")
            if content:  # 先判断结果不为空
            # 提取列表中第一个字典的 'text' 字段值
                extracted_text = content[0]['text']
                print(f"提取后的文本: {extracted_text}")
                return extracted_text
            else:
                print(f"API调用失败: {response.message}")
                return None
    except Exception as e:
            print(f"处理音频时发生错误: {str(e)}")
            return None


# chainlit 事件钩子
@cl.on_chat_start
async def on_chat_start():
    # 每次新会话分配一个session_id并创建global
    #session_id = str(uuid.uuid4())
    session_id = get_or_create_session(cl, ctx)
    cl.user_session.set("session_id", session_id)
    cl.user_session.set("counter", 0)
    cl.user_session.set("temp_values",{})
    cl.logger.info(f"医生{prac_id}的名字是{prac_name}")
    #cl.user_session.set("practitioner",)
    cl.logger.info(f"新会话分配 session_id: {session_id}")
    initMsg = f"欢迎您，{prac_name}医生，我是您的门诊助手{assistant_name}。我会协助您完成门诊，欢迎您向我提出任何问题。"
    await cl.Message(
        content=initMsg
    ).send()
    save_assistant_message(ctx, session_id, initMsg)

@cl.on_mcp_connect
async def on_mcp_connect(connection, session: ClientSession):
    cl.logger.info(f"Chainlit 正在尝试连接 MCP Server: {connection.name}")
    try:
        result = await session.list_tools()
        tools = [{"name": t.name, "description": t.description, "input_schema": t.inputSchema} for t in result.tools]
        cl.user_session.set("mcp_tools", {connection.name: tools})
        cl.user_session.set("mcp_session", session)
        cl.logger.info(f"MCP 连接成功，已获取 {len(tools)} 个工具")
        
        cl.logger.info(json.dumps(tools,indent=2,ensure_ascii=False))
    except Exception as e:
        cl.logger.error(f"获取 MCP 工具列表时出错: {e}")


# ===================
# 主 on_message 方法
# ===================

@cl.on_message
async def on_message(msg: cl.Message):
    session_id = get_or_create_session(cl, ctx)
    #save_user_message(ctx, session_id, msg)
    history_str, history = get_history_str(ctx, session_id)
    original_quest = msg.content

    # === 上下文优先判断 ===
    # 先判断上下文中的数据是否足够回答问题
    can_answer, reasoning = await can_answer_from_context(history, msg.content, client,llm_model)
    #cl.logger.info(f"历史信息1： {history_str}")
    #cl.logger.info(f"历史信息2： {history}")
    #cl.logger.info(f"判断状态： {can_answer}, 原因是: {reasoning}")
    save_user_message(ctx, session_id, msg)
    # 用Step显示判断结果    
    async with cl.Step("判断结果显示") as step:
        step.output = f"🧠 上下文判断: {'可以' if can_answer else '不可以'}直接回答\n"+ f"📝 判断理由: {reasoning}"

    if can_answer:
        # 直接基于上下文生成回答
        answer = await generate_context_answer(history, msg.content, client, llm_model)
        visual_tag = '需要图表'
        need_visual = False
        # 保存并返回回答
        if visual_tag in answer:
            answer = answer.replace(visual_tag, "")
            need_visual = True
        save_assistant_message(ctx, session_id, answer)
        await cl.Message(content=answer).send()
        # 调用Agent绘图
        if need_visual:
            print("准备画图")
            await generate_interactive_plotly_chart(answer,original_quest,client, llm_model)

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
    plan_json = await generate_plan(history, msg.content, tool_list,client,llm_model)
    raw_content = plan_json.get("raw", [])
    json_str = raw_content.strip().replace('```json', '').replace('```', '').strip()
    json_data = json.loads(json_str)
    plan = json_data.get("plan", [])
    explanation = json_data.get("explanation", "")

    print("????????")
    print(plan_json)
    process_steps = [f"多步执行计划：{json.dumps(plan, ensure_ascii=False, indent=2)}", f"计划说明：{explanation}"]
    print(process_steps)
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
            message = cl.Message(content="")
            stream = await client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": assistant_prompt},
                    {"role": "user", "content": llm_prompt},
                ],
                stream=True,
                temperature=0.01
            )
            async for part in stream:
                delta = part.choices[0].delta
                if delta.content:
                    # Stream the output of the step
                    await message.stream_token(delta.content)
            answer_texts.append(message.content)
        # 医保风险分析
        elif action == "risk_analyst":
            context_data = input_data if isinstance(input_data, str) else str(input_data)
            user_question = step_desc if isinstance(step_desc, str) else str(step_desc)
            question_prompt = f"""
            问题：{user_question}
            上下文信息:{context_data}
            """
            message = cl.Message(content="")
            stream = await client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": insurance_expert_prompt},
                    {"role": "user", "content": question_prompt},
                ],
                stream=True,
                temperature=0.01
            )
            async for part in stream:
                delta = part.choices[0].delta
                if delta.content:
                    # Stream the output of the step
                    await message.stream_token(delta.content)
            answer_texts.append(message.content)
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

@cl.on_audio_start
async def on_audio_start():
    cl.user_session.set("silent_duration_ms", 0)
    cl.user_session.set("is_speaking", False)
    cl.user_session.set("audio_chunks", [])
    return True

@cl.on_audio_end
async def on_audio_end():
    await process_audio()
    return True

@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk):
    audio_chunks = cl.user_session.get("audio_chunks")

    if audio_chunks is not None:
        audio_chunk = np.frombuffer(chunk.data, dtype=np.int16)
        audio_chunks.append(audio_chunk)

    # If this is the first chunk, initialize timers and state
    if chunk.isStart:
        cl.user_session.set("last_elapsed_time", chunk.elapsedTime)
        cl.user_session.set("is_speaking", True)
        return

    audio_chunks = cl.user_session.get("audio_chunks")
    last_elapsed_time = cl.user_session.get("last_elapsed_time")
    silent_duration_ms = cl.user_session.get("silent_duration_ms")
    is_speaking = cl.user_session.get("is_speaking")

    # Calculate the time difference between this chunk and the previous one
    time_diff_ms = chunk.elapsedTime - last_elapsed_time
    cl.user_session.set("last_elapsed_time", chunk.elapsedTime)

    # Compute the RMS (root mean square) energy of the audio chunk
    audio_energy = audioop.rms(
        chunk.data, 2
    )  # Assumes 16-bit audio (2 bytes per sample)

    if audio_energy < SILENCE_THRESHOLD:
        # Audio is considered silent
        silent_duration_ms += time_diff_ms
        cl.user_session.set("silent_duration_ms", silent_duration_ms)
        if silent_duration_ms >= SILENCE_TIMEOUT and is_speaking:
            cl.user_session.set("is_speaking", False)
            #await process_audio() 
    else:
        # Audio is not silent, reset silence timer and mark as speaking
        cl.user_session.set("silent_duration_ms", 0)
        if not is_speaking:
            cl.user_session.set("is_speaking", True)

async def process_audio():
    # Get the audio buffer from the session
    if audio_chunks := cl.user_session.get("audio_chunks"):
        # Concatenate all chunks
        concatenated = np.concatenate(list(audio_chunks))

        # Create an in-memory binary stream
        wav_buffer = io.BytesIO()

        # Create WAV file with proper parameters
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)  # mono
            wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
            wav_file.setframerate(24000)  # sample rate (24kHz PCM)
            wav_file.writeframes(concatenated.tobytes())

        # Reset buffer position
        wav_buffer.seek(0)

        cl.user_session.set("audio_chunks", [])

    frames = wav_file.getnframes()
    rate = wav_file.getframerate()

    duration = frames / float(rate)
    if duration <= 1.50:
        print("The audio is too short, please try again.")
        return

    audio_buffer = wav_buffer.getvalue()

    input_audio_el = cl.Audio(content=audio_buffer, mime="audio/wav")
    #sound_input = ("audio.wav", audio_buffer, "audio/wav") 
       
    transcription = await speech_to_text(audio_buffer)

    async with cl.Step("将语音转为文字") as step:
        step.output = f"🧠 用户的问题是: {transcription}"
        step.elements = [input_audio_el]

    msg = cl.Message(content=f"🧠 用户的问题是: {transcription}")
    await msg.send()
    await on_message(msg)
