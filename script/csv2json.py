# import pandas as pd
# import json
#
# # 读取CSV文件
# file_path = 'validation.csv'  # 替换为你的CSV文件路径
# df = pd.read_csv(file_path)
#
# # 查看原始数据
# print("原始数据:")
# print(df.head())
#
# # 1. 过滤有效样本：确保每个样本都包含 instruction, hypothesis, output(例子而已)，实际修改
#
# valid_samples = df[['context', 'question', 'answer']].dropna()
#
# # 2. 提取前 10000 条有效样本
# valid_samples = valid_samples.head(10000)
#
# # 3. 转换为 JSON 格式
# json_data = valid_samples.to_dict(orient='records')
#
# # 4. 控制 JSON 文件大小：将数据分批保存为多个 JSON 文件
# batch_size = 1000  # 每个 JSON 文件包含的样本数量
# num_batches = len(json_data) // batch_size + 1
#
# for i in range(num_batches):
#     start_idx = i * batch_size
#     end_idx = start_idx + batch_size
#     batch_data = json_data[start_idx:end_idx]
#
#     # 保存为 JSON 文件
#     output_file_path = f'valid_samples_batch_{i + 1}.json'
#     with open(output_file_path, 'w', encoding='utf-8') as f:
#         json.dump(batch_data, f, ensure_ascii=False, indent=4)
#
#     print(f"Batch {i + 1} saved to {output_file_path}")
#
# # 打印提取的有效样本数量
# print(f"Total valid samples extracted: {len(valid_samples)}")


import pandas as pd
import json
from datetime import datetime
import os

# 读取CSV文件
file_path = 'Train.csv'  # 替换为你的CSV文件路径
df = pd.read_csv(file_path,encoding='utf-8', on_bad_lines='skip')

# 查看原始数据
print("原始数据:")
print(df.head())

valid_samples = df[['Context', 'Input', 'Answers']].dropna()

valid_samples = valid_samples.head(66000)

# 3. 转换为 JSON 格式
json_data = valid_samples.to_dict(orient='records')

# 4. 控制 JSON 文件大小：将数据分批保存为多个 JSON 文件
batch_size = 66000  # 每个 JSON 文件包含的样本数量
num_batches = len(json_data) // batch_size + (1 if len(json_data) % batch_size else 0)

# 获取当前时间并格式化为字符串
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

# 创建以当前时间为名称的文件夹
folder_path = f'story_question_answer_json_output_{current_time}'
os.makedirs(folder_path, exist_ok=True)

for i in range(num_batches):
    start_idx = i * batch_size
    end_idx = start_idx + batch_size
    batch_data = json_data[start_idx:end_idx]

    # 保存为 JSON 文件到新文件夹
    output_file_path = os.path.join(folder_path, f'story_question_answer_samples_batch_{i + 1}.json')
    with open(output_file_path, 'w', encoding='utf-8',errors='ignore') as f:
        json.dump(batch_data, f, ensure_ascii=False, indent=4)

    print(f"Batch {i + 1} saved to {output_file_path}")

# 打印提取的有效样本数量
print(f"Total valid samples extracted: {len(valid_samples)}")