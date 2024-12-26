import pandas as pd
import json
from datetime import datetime
import os

# 读取CSV文件
file_path = 'train.csv'  # 替换为你的CSV文件路径
df = pd.read_csv(file_path)

# 查看原始数据
print("原始数据:")
print(df.head())

# 获取当前时间并格式化为字符串
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

# 创建以当前时间为名称的文件夹
folder_path = f'logic_group_json_output_{current_time}'
os.makedirs(folder_path, exist_ok=True)

# 根据 label 列的值对数据进行分组
grouped = df.groupby('label')

# 遍历每个分组，并对每个分组生成 JSON 文件
for label, group in grouped:
    valid_samples = group[['premise', 'hypothesis', 'explanation_1']].dropna()
    valid_samples = valid_samples.head(150000)  # 限制每个分组的样本数量

    # 转换为 JSON 格式
    json_data = valid_samples.to_dict(orient='records')

    # 控制 JSON 文件大小：将数据分批保存为多个 JSON 文件
    batch_size = 50000  # 每个 JSON 文件包含的样本数量
    num_batches = len(json_data) // batch_size + (1 if len(json_data) % batch_size else 0)

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_data = json_data[start_idx:end_idx]

        # 保存为 JSON 文件到新文件夹
        output_file_path = os.path.join(folder_path, f'label_{label}_train_samples_batch_{i + 1}.json')
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(batch_data, f, ensure_ascii=False, indent=4)

        print(f"Batch {i + 1} for label {label} saved to {output_file_path}")

    # 打印提取的有效样本数量
    print(f"Total valid samples extracted for label {label}: {len(valid_samples)}")