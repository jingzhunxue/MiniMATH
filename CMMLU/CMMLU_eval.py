import sys
from pathlib import Path
import pandas as pd

# 添加 frontend 目录的父目录到 sys.path
project_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(project_dir))

import os
import torch
from tokenizer.qwen72B.tokenization_qwen import QWenTokenizer

from model.modelv0 import ModelArgs, Transformer

import json

import re

from tqdm import tqdm

def get_csv_files(dir):
    return [f for f in os.listdir(dir) if f.endswith('.csv')]

print("当前工作目录:", os.getcwd())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

print(f"Using device: {device}")

tokenizer = QWenTokenizer.from_pretrained("tokenizer/qwen72B", local_files_only=True)

print(f"Vocab size: {tokenizer.vocab_size}")

model_args = dict(
    dim=0,
    n_layers=0,
    n_heads=0,
    n_kv_heads=0,
    vocab_size=0,
    multiple_of=0,
    max_seq_len=0,
    dropout=0,
)

out_dir = "out"
ckpt = "" ## your ckpt name here
print(f"inference from {out_dir}")

ckpt_path = os.path.join(out_dir, ckpt+".pt")
checkpoint = torch.load(ckpt_path, map_location="cpu")
checkpoint_model_args = checkpoint["model_args"]


for k in ["dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size", "multiple_of", "max_seq_len"]:
    model_args[k] = checkpoint_model_args[k]
    print(f"model_args[{k}] = {model_args[k]}")

# create the model
gptconf = ModelArgs(**model_args)
model = Transformer(gptconf)
state_dict = checkpoint["model"]

unwanted_prefix = "module._orig_mod."
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model = model.half()
model.eval()
model = model.to(device)

csv_files = get_csv_files('CMMLU/test')

top_p = 1.0
top_k = 1

total_count = 0
correct_count = 0

bar = tqdm()

with open('CMMLU/result.json', 'w', encoding="UTF-8") as f:
    for csv_file in csv_files:
        if "mathematics" in csv_file or "physics" in csv_file:
            df = pd.read_csv(f'CMMLU/test/{csv_file}')
            for index, row in df.iterrows():
                # 获取每个字段的值
                problem = row['Question'] + "\n" + "选项：A. " + row['A'] + "B. " + row['B'] + "C. " + row['C'] + "D. " + row['D']
                answer = row['Answer']
                
                messages=[
                    {
                        "role": "user",
                        "content": problem
                    }
                ]

                output_text = model.generate_messages(tokenizer, messages, max_new_tokens=512, temperature=0.01, top_p=top_p, device=device)
                matches = re.findall(r"[A-D]", output_text)
                if len(matches) == 0:
                    print(f"Error: {output_text}")
                    continue
                total_count += 1
                if matches[-1] == answer:
                    correct_count += 1
                    f.write(json.dumps({"结果":"正确", "question": problem, "answer": output_text, "standard":answer}, ensure_ascii=False) + "\n")
                else:
                    f.write(json.dumps({"结果":"错误", "question": problem, "answer": output_text, "standard":answer}, ensure_ascii=False) + "\n")
                bar.update(1)
                bar.set_postfix_str(f"Accuracy: {correct_count / total_count:.4f}")


                f.flush()