import json

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

### 对话历史：
{json.dumps(history, ensure_ascii=False, indent=2)}

### 当前问题：
"{current_query}"
"""
    
    try:
        # 使用轻量模型快速判断
        completion = await client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": "你是一个上下文判断助手，只输出JSON格式"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.01,
            max_tokens=150,
            response_format={"type": "json_object"}
        )
        
        # 解析JSON响应
        response = completion.choices[0].message.content
        response_data = json.loads(response)
        
        # 返回判断结果和推理过程
        return response_data.get("can_answer", False), response_data.get("reasoning", "")
    except Exception as e:
        return False, f"判断错误: {str(e)}"

async def generate_context_answer(history, current_query, client):
    """
    基于上下文生成回答（使用更强大的模型）
    """
    messages = [
        {"role": "system", "content": "你是一个智能助手，请严格基于对话历史回答问题"},
        *history,
        {"role": "user", "content": current_query}
    ]
    
    try:
        # 使用更强大的模型生成回答
        completion = await client.chat.completions.create(
            model="qwen-plus",
            messages=messages,
            temperature=0.01
        )
        return completion.choices[0].message.content
    except Exception as e:
        return "无法基于上下文生成回答"