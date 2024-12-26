import pandas as pd
import csv

# 读取 CSV 文件的列名
file_path = 'Train.csv'  # 替换为您的 CSV 文件路径
columns = pd.read_csv(file_path, nrows=0).columns.tolist()
print(columns)

# import chardet
#
# with open(file_path, 'rb') as file:
#     raw_data = file.read()
#     result = chardet.detect(raw_data)
#     encoding = result['encoding']
#
# print(encoding)
#
# with open(file_path, 'r', encoding=encoding) as file:
#     reader = csv.reader(file)
#     number_of_rows = sum(1 for row in reader)
with open(file_path, 'r', encoding='gbk', errors='ignore') as file:
    reader = csv.reader(file)
    number_of_rows = sum(1 for row in reader)
print(number_of_rows)