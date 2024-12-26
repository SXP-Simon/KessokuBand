import os

# 定义文件夹路径
folder_path = '../BBBBB比赛/logic_group_json_output_20241213_001058'  # 替换为你的文件夹路径

# 定义新的标签名称映射
label_mapping = {
    'label_0': 'entail',
    'label_1': 'contradict',
    'label_2': 'neutral',
}

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 检查文件是否为 JSON 文件
    if filename.endswith('.json'):
        # 构建完整的文件路径
        old_file_path = os.path.join(folder_path, filename)

        # 检查文件名中是否包含需要替换的标签
        for old_label, new_label in label_mapping.items():
            if old_label in filename:
                # 构建新的文件名
                new_filename = filename.replace(old_label, new_label)
                new_file_path = os.path.join(folder_path, new_filename)

                # 重命名文件
                os.rename(old_file_path, new_file_path)
                print(f"Renamed '{old_file_path}' to '{new_file_path}'")