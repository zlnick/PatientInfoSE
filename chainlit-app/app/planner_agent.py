import chainlit as cl
import json

async def generate_plan(history, user_input, tools, client):
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
     "input": 输入参数 (dict。如果要使用工具，则字段名与工具定义严格一致。）
       应根据计划上下文分析。如本步依赖于之前子任务产生的临时变量，必须用$var名引用。但input本身是一个json对象，可以使用多个属性或者再嵌入json对象。
       例如之前的子任务生成的result_var中保存的变量名为$var：
          "result_var": "$patient_info"
       那么在本步的任务中，如果需要用到之前的子任务产生的临时变量，在input字段中必须以同样的变量名引用，如：
            "others":"其它需要使用的变量"
            "$patient_info": "$patient_info"
     "result_var": 本步产生的结果所在的变量名（变量名必须为只包括一个'$'符号的$var。多步plan间如有依赖，用result_var实现数据流，即前一步任务生成的result_var变量将被下一步子任务通过input引用实现依赖传递。禁止凭空捏造上下文中不存在的变量作为某一步的输入。）
        例如："result_var": "$patient_info" 
     "description": "本步意图说明"
   }}
3. 多步plan间如有依赖，用result_var实现数据流，即上一步的result_var变量将被下一步通过input引用实现依赖传递。禁止凭空捏造上下文中不存在的变量作为某一步的输入。
4. 在返回之前再做一遍判断，如果最后的结果是tool返回的数据，则你还需要加一步llm_answer，用大模型将返回的数据转换为医生容易阅读的形态再返回给用户。注意，医生通常关注和患者本身相关的病情等数据，对于数据是不是FHIR资源数据，格式是不是符合FHIR标准，以及数据中的审计、版本、是不是由工具生成的这一类信息不关注，应当在生成意图描述时详细要求大模型不要描述FHIR协议相关的数据特征，以医生的需要来总结和描述数据。
5. 最终输出格式：
{{
  "plan": [step1, step2, ...],
  "explanation": "简要说明你的计划拆解思路"
}}
仅输出严格JSON格式plan和explanation，不要输出其他内容，不要输出"```json"这样的标签。
"""

    async with cl.Step(name="执行计划生成Agent", type="llm") as step:
        #step.input = prompt
        stream = await client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": "你是一个计划生成Agent，只负责输出JSON结构计划"},
                {"role": "user", "content": prompt},
            ],
            stream=True,
            temperature=0.01
        )
        async for part in stream:
            delta = part.choices[0].delta
            if delta.content:
                # Stream the output of the step
                await step.stream_token(delta.content)
        #cl.logger.info(step.output)
        response = step.output
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


