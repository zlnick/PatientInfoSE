import os
import time
from openai import AsyncOpenAI
from mcp import ClientSession
import chainlit as cl
import requests

client = AsyncOpenAI(
    api_key=os.getenv("DEEP_SEEK_API_KEY"), base_url="https://api.deepseek.com"
)

@cl.on_chat_start
def on_chat_start():
    cl.user_session.set("counter", 0)

@cl.on_mcp_connect
async def on_mcp_connect(connection, session: ClientSession):
    cl.logger.info(f"Chainlit 正在尝试连接 MCP Server: {connection.name}")
    try:
        result = await session.list_tools()
        tools = [ {"name": t.name, "description": t.description, "input_schema": t.inputSchema} for t in result.tools ]
        cl.user_session.set("mcp_tools", {connection.name: tools})
        cl.logger.info(f"MCP 连接成功，已获取 {len(tools)} 个工具")
        cl.logger.info(tools)
    except Exception as e:
        cl.logger.error(f"获取 MCP 工具列表时出错: {e}")


@cl.on_message
async def on_message(msg: cl.Message):
    start = time.time()
    stream = await client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system", "content": "You are an helpful assistant"},
            *cl.chat_context.to_openai(),
        ],
        stream=True,
    )

    # Flag to track if we've exited the thinking step
    thinking_completed = False

    # Streaming the thinking
    async with cl.Step(name="Thinking") as thinking_step:
        async for chunk in stream:
            delta = chunk.choices[0].delta
            reasoning_content = getattr(delta, "reasoning_content", None)
            if reasoning_content is not None and not thinking_completed:
                await thinking_step.stream_token(reasoning_content)
            elif not thinking_completed:
                # Exit the thinking step
                thought_for = round(time.time() - start)
                thinking_step.name = f"{thought_for}秒进行思考"
                await thinking_step.update()
                thinking_completed = True
                break

    final_answer = cl.Message(content="")

    # Streaming the final answer
    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            await final_answer.stream_token(delta.content)

    await final_answer.send()
    counter = cl.user_session.get("counter")
    counter += 1
    cl.user_session.set("counter", counter)
    await cl.Message(content=f"你已经发送了 {counter} 条消息！").send()