from dotenv import load_dotenv
import os
import chainlit as cl
import dashscope
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from dashscope import Generation
from openai import AsyncOpenAI

load_dotenv()

# LLM客户端
client = AsyncOpenAI(
    api_key=os.getenv("Qwen_API_KEY"), 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

async def generate_interactive_plotly_chart(data_description, chart_request):
    """
    生成交互式Plotly图表并显示在Chainlit界面
    
    参数:
        data_description: 数据的描述（表格数据、CSV等）
        chart_request: 用户对图表的请求描述
    """
    # 步骤1: 调用大模型生成Plotly可视化代码
    prompt = f"""
    请根据以下数据生成Plotly交互式图表代码：
    
    {data_description}
    
    用户要求：{chart_request}
    
    要求：
    1. 使用plotly.express或plotly.graph_objects库
    2. 代码必须包含必要的导入语句
    3. 生成交互式图表，包含悬停提示、缩放、平移等交互功能
    4. 根据输入的数据的特征决定用哪一种图表进行展示
    5. 设置合适的图表标题、坐标轴标签和图例
    6. 使用专业配色方案
    7. 代码最后必须创建并返回一个Figure对象：fig
    8. 仅返回Python代码，不要包含任何解释
    9. 图表标题、坐标轴标签应当使用中文描述
    
    示例代码框架：
    import plotly.express as px
    import pandas as pd
    
    # 数据定义
    df = pd.DataFrame({{
        '月份': ['1月', '2月', '3月', '4月', '5月', '6月'],
        '销售额(万元)': [150, 180, 220, 200, 240, 280],
        '成本(万元)': [120, 140, 180, 170, 200, 230]
    }})
    
    # 创建交互式折线图
    fig = px.line(df, x='月份', y=['销售额(万元)', '成本(万元)'], 
                 title='2023年销售与成本趋势',
                 labels={{'value': '金额(万元)', 'variable': '指标'}},
                 markers=True)
    
    # 设置布局
    fig.update_layout(
        hovermode='x unified',
        template='plotly_white',
        legend_title_text='',
        xaxis_title='月份',
        yaxis_title='金额(万元)',
        hoverlabel=dict(bgcolor="white", font_size=12)
    )
    
    # 确保最后一行是fig，这样exec后可以获取到Figure对象
    fig
    """
    
    messages = [
        {"role": "user", "content": prompt}
    ]

    # 调用通义千问生成代码
    completion = await client.chat.completions.create(
            model="qwen-plus",
            messages=messages,
            temperature=0.01
        )
        
    generated_code = completion.choices[0].message.content
    

    async with cl.Step("生成的可视化代码:") as step:
            step.output = "**生成的可视化代码**:\n```python\n" + generated_code + "\n```"
    
    # 清理可能的Markdown代码块标记
    if generated_code.startswith("```python"):
        generated_code = generated_code[10:]
    if generated_code.endswith("```"):
        generated_code = generated_code[:-3]
    
    # 步骤2: 执行生成的代码
    try:
        # 创建执行环境
        local_vars = {}
        global_vars = {
            'px': px,
            'go': go,
            'pd': pd,
            'np': np,
            'fig': None  # 确保fig在全局变量中
        }
        
        # 执行生成的代码
        exec(generated_code, global_vars, local_vars)

        # 获取生成的Figure对象
        fig = global_vars.get('fig') or local_vars.get('fig')
        if not fig:
            # 尝试从local_vars中获取
            for var in local_vars.values():
                if isinstance(var, go.Figure):
                    fig = var
                    break
        
        if not fig:
            raise Exception("无法获取图表Figure对象")
        
        # 步骤3: 在Chainlit中展示交互式图表
        # 直接使用cl.Plotly展示Figure对象（这才是正确的方式！）
        plotly_element = cl.Plotly(
            name="interactive_chart",
            figure=fig,
            display="inline"
        )

        await cl.Message(
            content="这是根据您的要求生成的交互式图表:",
            elements=[plotly_element]
        ).send()

        return fig
        
    except Exception as e:
        error_msg = f"执行可视化代码时出错: {str(e)}"
        await cl.Message(content=error_msg).send()
        # 显示有问题的代码以便调试
        await cl.Message(
            content="**生成的代码**:\n```python\n" + generated_code + "\n```",
            author="Code"
        ).send()
        return None

@cl.on_chat_start
async def on_chat_start():
    """初始化聊天界面"""
    await cl.Message(
        content="欢迎使用数据可视化助手！我可以帮您将数据转换为交互式图表。请提供您的数据和可视化需求。"
    ).send()
    
    # 提供示例数据
    sample_data = """
    月份: 1月, 2月, 3月, 4月, 5月, 6月
    销售额(万元): 150, 180, 220, 200, 240, 280
    成本(万元): 120, 140, 180, 170, 200, 230
    利润率(%): 20, 22.2, 18.2, 15, 16.7, 17.9
    """
    
    await cl.Message(
        content="示例数据:\n```\n" + sample_data + "\n```\n您可以尝试输入: '请用折线图展示销售额和成本的趋势'",
        author="System"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    # 检查是否是图表请求
    if "图表" in message.content or "画图" in message.content or "可视化" in message.content or "plot" in message.content or "chart" in message.content:
        await cl.Message(
            content="正在分析您的请求并生成交互式图表..."
        ).send()
        
        # 从上下文获取数据或使用示例数据
        sample_data = """
        {
        '月份': ['1月', '2月', '3月', '4月', '5月', '6月'],
        '销售额(万元)': [150, 180, 220, 200, 240, 280],
        '成本(万元)': [120, 140, 180, 170, 200, 230]
        }
        """
        
        # 生成并显示图表
        await generate_interactive_plotly_chart(sample_data, message.content)
    else:
        # 处理普通消息
        await cl.Message(
            content=f"您说的是: {message.content}\n\n如果您需要数据可视化，请描述您的数据和可视化需求。"
        ).send()

