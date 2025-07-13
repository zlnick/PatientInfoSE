from dotenv import load_dotenv
from fastapi import FastAPI, Request
import uvicorn
import os
from openai import AsyncOpenAI

# 加载环境变量
load_dotenv()

# 初始化 FastAPI
app = FastAPI()

# 配置 Qwen Plus 的 OpenAI 兼容接口
Qwen_API_KEY = os.getenv("Qwen_API_KEY")
client = AsyncOpenAI(
    api_key=Qwen_API_KEY,
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)

@app.post("/generate-fhir-query")
async def generate_fhir_query(request: Request):
    data = await request.json()
    natural_language_query = data.get("query", "")
    context = data.get("context", {})

    if not natural_language_query:
        return {"error": "Missing query"}

    system_prompt = """
你是一个FHIR查询助手。请根据用户自然语言需求，生成符合FHIR R4标准的查询参数，使用FHIR Search Parameter和Reverse Chaining。
- 仅使用官方定义的参数，例如：Encounter查询时间范围用date，Patient年龄用birthdate。
- 不要直接用FHIR资源字段名（如period）。
- 避免非法参数。
返回一个JSON对象，包含：
{
  "resource_type": "资源类型",
  "filters": {"参数名":"值",...}
}
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": natural_language_query}
    ]

    try:
        response = await client.chat.completions.create(
            model="qwen-plus",
            messages=messages,
            temperature=0.2,
            timeout=20
        )
        content = response.choices[0].message.content
        return {"fhir_query": content.strip()}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
