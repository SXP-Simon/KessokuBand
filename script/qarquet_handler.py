import fastparquet

# 打开Parquet文件
parquet_file = fastparquet.ParquetFile('')

# 读取Parquet文件内容到Pandas DataFrame
df = parquet_file.to_pandas()

# 打印DataFrame的前几行
print(df.head())

# 将DataFrame保存为CSV文件
df.to_csv('output.csv', index=False)

import pandas as pd
import json

# 读取CSV文件
file_path = 'output.csv'  # 替换为你的CSV文件路径
df = pd.read_csv(file_path)

# 提取question和choices列
questions = df['question']
choices = df['choices']
answers = df['answer']

# 构建结果列表
result = []

for question, choice, answer in zip(questions, choices, answers):
    try:
        # 将choices字符串转换为列表
        choice_list = eval(choice)
        # 获取对应的answer
        answer_index = int(answer)  # answer列中的值是索引
        selected_answer = choice_list[answer_index]

        # 构建结果项
        result_item = {
            "Instruction": question,
            "Input": "",  # 假设没有额外的输入
            "output": selected_answer
        }
        result.append(result_item)
    except Exception as e:
        print(f"Error processing question: {question}")
        print(f"Choices: {choice}")
        print(f"Answer: {answer}")
        print(f"Error: {e}")
        print("-" * 40)

# 将结果转换为JSON格式
json_result = json.dumps(result, ensure_ascii=False, indent=4)
print(json_result)

# 如果需要保存到文件
with open('output.json', 'w', encoding='utf-8') as f:
    f.write(json_result)