import fastparquet
import pandas as pd
import json

# 打开Parquet文件
parquet_file = fastparquet.ParquetFile('hhh.parquet')

# 读取Parquet文件内容到Pandas DataFrame
df = parquet_file.to_pandas()

# 打印DataFrame的前几行
print(df.head())

# 将DataFrame保存为CSV文件
df.to_csv('output.csv', index=False)

# 读取CSV文件
file_path = 'output.csv'  # 替换为你的CSV文件路径
df = pd.read_csv(file_path)

# 提取question和answer列
questions = df['INSTRUCTION']
answers = df['RESPONSE']

# 构建结果列表
result = []

# 打印每个问题及其对应的答案
for question, answer in zip(questions, answers):
    result_item = {
        "Instruction": question,
        "Input": "",  # 假设没有额外的输入
        "output": answer
    }
    result.append(result_item)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print("-" * 40)

# 将结果转换为JSON格式
json_result = json.dumps(result, ensure_ascii=False, indent=4)
print(json_result)

# 保存结果到文件
with open('output.json', 'w', encoding='utf-8') as f:
    f.write(json_result)