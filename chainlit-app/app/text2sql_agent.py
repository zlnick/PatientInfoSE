import os
from dotenv import load_dotenv
import asyncio
from openai import AsyncOpenAI

async def generate_sql_query(question: str, schema_info: str, client: AsyncOpenAI,llm_model) -> str:
    """
    将自然语言问题转换为SQL查询语句
    :param question: 自然语言问题
    :param schema_info: 数据库表结构信息
    :param client: OpenAI客户端
    :return: 生成的SQL查询语句
    """
    prompt = f"""
    你是一个text2sql专家，能将自然语言问题转换为正确的SQL查询语句。
    数据库表结构如下:
    {schema_info}
    
    请根据以下问题生成SQL查询语句:
    {question}
    
    注意:
    1. 只返回SQL查询语句，不要添加任何解释
    2. 确保SQL语法正确
    3. 使用正确的表名和字段名
    """
    
    completion = await client.chat.completions.create(
        model=llm_model,
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

async def text2sql_agent(question: str, schema_info: str, client: AsyncOpenAI, db_connection,llm_model) -> list:
    """
    text2sql agent主函数
    :param question: 自然语言问题
    :param schema_info: 数据库结构信息
    :param client: OpenAI客户端
    :param db_connection: 数据库连接
    :return: 查询结果
    """
    sql = await generate_sql_query(question, schema_info, client,llm_model)
    results = await execute_sql_query(sql, db_connection)
    return {
        "sql": sql,
        "results": results
    }

if __name__ == "__main__":
    load_dotenv()
    # LLM客户端
    client = AsyncOpenAI(
        api_key=os.getenv("Qwen_API_KEY"), 
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    problem = '列举资源ID为Patient/794的患者的总花费是多少？'
    scheme = """
    [{"TableName": "Data.Order", "Description": "就诊订单表，每次就诊过程会产生一到多条订单。", "Columns": [{"ColumnName": "ID", "Type": "bigint", "Description": ""}, {"ColumnName": "Encounter", "Type": "varchar", "Description": "就诊ID，与FHIR资源的Key保持 一致，如'Encounter/456'"}, {"ColumnName": "OrderCategory", "Type": "varchar", "Description": " 订单类型"}, {"ColumnName": "Patient", "Type": "varchar", "Description": "患者ID，与FHIR资源的Key保持一致，如'Patient/123'"}]}, {"TableName": "Data.OrderItem", "Description": "收费明细项，每一个就诊订单有零到多个收费明细项。", "Columns": [{"ColumnName": "ID", "Type": "bigint", "Description": ""}, {"ColumnName": "Currency", "Type": "varchar", "Description": "收费货币单位：CNY-人民币,USD-美元"}, {"ColumnName": "ItemName", "Type": "varchar", "Description": "收费项名称"}, {"ColumnName": "OrderID", "Type": "varchar", "Description": "订单编 号，关联Data.Order表的ID属性"}, {"ColumnName": "Price", "Type": "numeric", "Description": "收费项价格"}]}]
    """
    str = asyncio.run(generate_sql_query(problem,scheme,client))

    print(str)