import pandas as pd
import logging
from datetime import datetime
import iris as irisnative
from dotenv import load_dotenv
import os
import dashscope
#from dashscope import TextEmbedding

load_dotenv()

connection = irisnative.connect(
    os.getenv("IRIS_HOSTNAME"), 
    int(os.getenv("IRIS_PORT")), 
    os.getenv("IRIS_NAMESPACE"), 
    os.getenv("IRIS_USERNAME"), 
    os.getenv("IRIS_PASSWORD")
)

# 配置日志
logging.basicConfig(
    filename='excel_data_printer.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def prepare_table():
    try:
        cursor = connection.cursor()
        clear_sql = """Drop table Demo.DrugInfo"""
        cursor.execute(clear_sql)
        table_sql = """
            CREATE TABLE IF NOT EXISTS Demo.DrugInfo (
                DrugEmbedding VECTOR(FLOAT,1536),
                RuleInsurance VARCHAR(10000)
            )
        """
        cursor.execute(table_sql)
        cursor.close()
    except Exception as e:
        return [f"SQL执行错误: {str(e)}"]



def read_excel_data(file_path):
    """读取Excel文件数据"""
    try:
        # 读取Excel文件，保留所有原始数据类型
        df = pd.read_excel(file_path, dtype=str)
        logging.info(f"成功读取Excel文件，共{len(df)}条记录，{len(df.columns)}列")
        return df
    except FileNotFoundError:
        error_msg = f"Excel文件未找到: {file_path}"
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)
    except Exception as e:
        error_msg = f"读取Excel失败: {str(e)}"
        logging.error(error_msg)
        raise

def insert_data(df, batch_size=100):
    """按行打印数据，每行显示列名和对应的值"""
    if df.empty:
        print("Excel文件中没有数据")
        return
    
    # 打印表头信息
    print(f"\n共有 {len(df)} 条记录，列名如下：")
    print(", ".join(df.columns))
    print("\n" + "="*80 + "\n")
    
    sql_insert = """
        INSERT INTO Demo.DrugInfo (DrugEmbedding, RuleInsurance)
        VALUES (?, ?)
    """
    cursor = connection.cursor()
    
    # 逐行处理数据
    for index, row in df.iterrows():
        row_number = index + 1  # 行号从1开始
        print(f"处理第 {row_number} 行数据")

        # 检查备注是否为空（包括None、空字符串、仅空白字符的情况）
        remark = row['备注']
        if pd.isna(remark) or str(remark).strip() == '':
            print(f"第 {row_number} 行备注为空，跳过插入")
            continue

        # 提取药品名称和备注，组合成字符串
        drug_item = f"{row['药品名称']}的报销约束是:{row['备注']}"
        ruleEmbedding = get_embedding(drug_item)
        embedding_str = ",".join(map(str, ruleEmbedding))
        # 执行插入（这里仅传入组合后的列表，根据实际表结构调整参数）
        cursor.execute(sql_insert, [embedding_str,drug_item])
    cursor.close()

def get_embedding(texts):
        """获取文本的嵌入向量"""
        # DashScope Embedding API调用
        response = dashscope.TextEmbedding.call(
            model="text-embedding-v2",  # Qwen3 Embedding专用模型
            input=texts
        )
        if response.status_code == 200:
            # 提取嵌入向量
            embeddings = [item["embedding"] for item in response.output["embeddings"]]
            return embeddings
        else:
            print(texts)
            raise Exception(f"Embedding API error: {response.code}, {response.message}")

def main():
    
    print("--准备药品医保规则表DrugInfo.Insurance--")
    prepare_table()
    print("--药品医保规则表DrugInfo.Insurance处理完毕--")
    # 配置Excel文件路径（请根据实际情况修改）
    excel_file = "insurance_drug.xlsx"  # 替换为你的Excel文件路径
    dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
    try:
        # 读取Excel数据
        df = read_excel_data(excel_file)
        # 按行转换并插入数据
        insert_data(df,50)
        print(f"数据处理完成，共 {len(df)} 条记录")
    except Exception as e:
        print(f"操作失败：{str(e)}")

if __name__ == "__main__":
    main()
    