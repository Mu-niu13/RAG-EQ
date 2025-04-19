import csv
import json

csv_file = 'EmpatheticDialogues/train.csv'
json_file = 'empathetic_dialogues.json'

data = []
with open(csv_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append({
            'utterance': row['utterance'],
            'emotion': row['context']
        })

with open(json_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4)

print(f"[DONE] {json_file} 生成完成！")
