import chainlit as cl
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


async def generate_interactive_plotly_chart(data_description, chart_request,client,llm_model):
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
    async with cl.Step("生成图表Agent:") as step:
        try:
            stream = await client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.01,
                stream=True
            )
            async for part in stream:
                delta = part.choices[0].delta
                if delta.content:
                    # Stream the output of the step
                    await step.stream_token(delta.content)
            generated_code = step.output
        except Exception as e:
            return "无法生成图表"
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



