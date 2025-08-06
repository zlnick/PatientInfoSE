import asyncio
from openai import AsyncOpenAI

async def generate_sql_query(question: str, schema_info: str, client: AsyncOpenAI) -> str:
    """
    将自然语言问题转换为SQL查询
    :param question: 自然语言问题
    :param schema_info: 数据库表结构信息
    :param client: OpenAI客户端
    :return: 生成的SQL查询
    """
    prompt = f"""
    你是一个text2sql专家，能将自然语言问题转换为正确的SQL查询。
    数据库表结构如下:
    {schema_info}
    
    请根据以下问题生成SQL查询:
    {question}
    
    注意:
    1. 只返回SQL查询的结果，不要添加任何解释
    2. 确保SQL语法正确
    3. 使用正确的表名和字段名
    """
    
    completion = await client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "你是一个专业的SQL生成器"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.01
    )
    return completion.choices[0].message.content

async def execute_sql_query(sql: str, db_connection) -> list:
    """执行SQL查询并返回结果"""
    try:
        # 这里根据实际数据库类型实现查询执行逻辑
        # 例如使用sqlite3、psycopg2等
        cursor = db_connection.cursor()
        cursor.execute(sql)
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        return results
    except Exception as e:
        return [f"SQL执行错误: {str(e)}"]

async def text2sql_agent(question: str, schema_info: str, client: AsyncOpenAI, db_connection) -> list:
    """
    text2sql agent主函数
    :param question: 自然语言问题
    :param schema_info: 数据库结构信息
    :param client: OpenAI客户端
    :param db_connection: 数据库连接
    :return: 查询结果
    """
    sql = await generate_sql_query(question, schema_info, client)
    results = await execute_sql_query(sql, db_connection)
    return {
        "sql": sql,
        "results": results
    }