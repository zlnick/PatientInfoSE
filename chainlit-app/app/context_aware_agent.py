import json
import chainlit as cl

async def can_answer_from_context(history, current_query, client):
    """
    判断是否可以使用上下文回答当前问题
    返回: (can_answer, reasoning)
    """
    # 空历史直接返回False
    if not history:
        return False, "无历史上下文"
    
    # 构造判断提示
    prompt = f"""
你是一个非常了解HL7 FHIR的医疗信息助手。
请分析以下对话历史和当前问题，判断是否可以直接基于上下文回答问题。输出JSON格式：

{{
  "can_answer": true/false,
  "reasoning": "判断理由"
}}

### 分析指南：
1. 检查历史中是否有直接相关数据
2. 检查历史中的工具调用结果是否包含所需信息
3. 评估是否有足够上下文推导答案
4. 考虑话题是否一致
5. 注意，目前程序已经集成了绘制图表的功能。如果上下文中的数据足够用于绘制图表，你应当判断为可以直接基于上下文回答问题。

### 重点：
如果没有直接的FHIR资源支撑，就判断为不能回答问题。从SQL获取的数据只能用来进行财务收费方面的业务，不能当作临床信息。
例如Appointment资源中引用了患者（Patient），有患者的id和姓名，但没有其它详细信息，则必须判断为不能直接基于上下文回答问题。
你做出这样的判断后将由其它Agent决定如何获取这些数据。

### 对话历史：
{json.dumps(history, ensure_ascii=False, indent=2)}

### 当前问题：
"{current_query}"
"""

    async with cl.Step(name="上下文检测Agent", type="llm") as step:
        step.input = prompt

        stream = await client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": "你是一个上下文判断助手，只输出JSON格式"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.01,
            stream=True,
            max_tokens=150,
            response_format={"type": "json_object"}
        )

        async for part in stream:
            delta = part.choices[0].delta
            if delta.content:
                # Stream the output of the step
                await step.stream_token(delta.content)

        #cl.logger.info(step.output)
        response_data = json.loads(step.output)
        
        return response_data.get("can_answer", False), response_data.get("reasoning", "")

async def generate_context_answer(history, current_query, client):
    """
    基于上下文生成回答
    """
    context_prompt = """
    你是一个智能助手，请严格基于对话历史回答问题。
    如果用户的问题有数据可视化的意味，且你能够获得回答用户的问题所需的数据，则按照用户要求将所需数据整理为列表形态，再在回答之后换行并加上一句固定的语句"需要图表"。
    如果用户要求绘制图表，你先不要拒绝生成图表，先检查上下文中是否有所需的数据，如果有，则将用户要求的图表所需的数据先整理成列表形态，再在回答之后换行并加上一句固定的语句"需要图表"。
    如果用户没有查看图表的需要，就不要追加"需要图表"字样。
    如果上下文中缺少数据，就回答上下文中数据不足，绝不允许编造数据。
    """
    async with cl.Step(name="处理上下文Agent：", type="llm") as step:
        try:
            message = cl.Message(content="")
            stream = await client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {"role": "system", "content": context_prompt},
                    *history,
                    {"role": "user", "content": current_query}
                ],
                temperature=0.01,
                stream=True
            )
            async for part in stream:
                delta = part.choices[0].delta
                if delta.content:
                    # Stream the output of the step
                    await step.stream_token(delta.content)
            response = step.output
            return response
        except Exception as e:
            return "无法基于上下文生成回答"