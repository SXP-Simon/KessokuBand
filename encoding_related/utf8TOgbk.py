import json

def convert_json_encoding(input_file, output_file, input_encoding='MacRoman', output_encoding='utf-8'):
    # 读取 UTF-8 编码的 JSON 文件
    with open(input_file, 'r', encoding=input_encoding) as f:
        data = json.load(f)

    # 将数据写入 GBK 编码的 JSON 文件
    with open(output_file, 'w', encoding=output_encoding) as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"JSON 文件已从 {input_encoding} 转换为 {output_encoding}，并保存到 {output_file}")

# 示例用法
input_file = 'merging_pot.json'  # 替换为你的输入 JSON 文件路径
output_file = 'format2.json'  # 替换为你希望生成的 GBK 编码 JSON 文件路径

convert_json_encoding(input_file, output_file)