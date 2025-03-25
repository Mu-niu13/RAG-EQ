# import torch
#
# print(torch.cuda.is_available())  # True
# print(torch.cuda.get_device_name(0))  # 显卡名称

from datasets import load_dataset

dataset = load_dataset("empathetic_dialogues", trust_remote_code=True)
dataset['train'].to_csv("train.csv", index=False)
dataset['test'].to_csv("test.csv", index=False)
dataset['validation'].to_csv("valid.csv", index=False)
