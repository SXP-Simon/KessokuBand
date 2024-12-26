import pandas as pd
import json

csv_file_path = 'train.csv'  # Replace with your CSV file path
json_file_path = '12102230.json'  # Replace with your JSON file path

df = pd.read_csv(csv_file_path)

# Extract question and answer columns
questions = df['instruction']
answers = df['output']

# Build result list
result = []

# Iterate through each question and its corresponding answer
for question, answer in zip(questions, answers):
    result_item = {
        "Instruction": question,
        "Input": "",  # Assuming no additional input
        "Output": answer
    }
    result.append(result_item)
    # print(f"Question: {question}")
    # print(f"Answer: {answer}")
    # print("-" * 40)

# Convert result to JSON format
json_result = json.dumps(result, ensure_ascii=False, indent=4)
print(json_result)

# Save result to file
with open(json_file_path, 'w', encoding='utf-8') as f:
    f.write(json_result)