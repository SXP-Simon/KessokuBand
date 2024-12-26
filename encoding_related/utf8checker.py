import pandas as pd
import csv

# 读取 CSV 文件的列名
file_path = 'merged_output.json'  # 替换为您的 CSV 文件路径
columns = pd.read_json(file_path)
import chardet

with open(file_path, 'rb') as file:
    raw_data = file.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding']
print(encoding)