import pandas as pd

# 假设你的数据集已经加载到一个名为 df 的 DataFrame 中
# 如果数据集是 CSV 文件，你可以使用以下代码加载它
df = pd.read_csv('validation.csv')

# 查看数据集的前几行
print(df.head())

# 根据 label 列的值对数据进行分组
grouped = df.groupby('label')

# 打印每个分类的统计信息
for name, group in grouped:
    print(f"\nGroup: {name}")
    print(group.describe())

# 如果你想要将每个分组的数据保存到单独的文件中
for name, group in grouped:
    group.to_csv(f'group_{name}.csv', index=False)