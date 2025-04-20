import json
import os
import random


SOURCE_PATH = "knowledge.json"
OUTPUT_PATH = "eval_queries.json"

#load
with open(SOURCE_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

valid = [entry for entry in data if "query" in entry]

sampled = random.sample(valid, 20)

eval_queries = [
    {
        "query": item["query"],
        "conv_id": i
    }
    for i, item in enumerate(sampled)
]


with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(eval_queries, f, indent=2, ensure_ascii=False)

print(f"successful")
