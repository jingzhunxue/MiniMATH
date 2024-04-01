import sys
from pathlib import Path

# 添加 frontend 目录的父目录到 sys.path
project_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(project_dir))

import os
import torch
from tokenizer.qwen72B.tokenization_qwen import QWenTokenizer

from model.modelv0 import ModelArgs, Transformer

import json
import random
from tqdm import tqdm
import jsonlines

if __name__ == "__main__":

    print("当前工作目录:", os.getcwd())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
    

    model = model.to(device)
    model = model.half()
    model.eval()

    train_json = {}
    
    import re

    pattern = re.compile(r"#### (\d+)")

    with jsonlines.open("GSM8K/gsm_test.jsonl") as reader:
        for obj in reader:
            match = re.search(pattern, obj["answer"])
            if match:
                train_json[obj["question"]] = match.group(1)
            else:
                train_json[obj["question"]] = obj["answer"]

    problem_nums = 1000
    bar = tqdm(total=problem_nums)
    #打乱df顺序
    train_json_list = list(train_json.items())
    sampled_train_json_list = random.sample(train_json_list, problem_nums)
    df_topn = dict(sampled_train_json_list)


    total_score = 0
    total_correct = 0
    total_count = 0
    top_k = 1
    top_p = 1.0
    with open("GSM8K/result.json", "w", encoding="UTF-8") as f:
        for q, a in df_topn.items():
            messages = [
                {
                    "role": "user",
                    "content": q
                }
            ]
            output_text = model.generate_messages(tokenizer, messages, max_new_tokens=512, temperature=0.1, top_p=top_p, device=device)

            matches = re.findall(r'-?\d+(?:\.\d+)?', output_text)
            
            if len(matches) > 0 :
                ans = matches[-1]
                if ans == a:
                    result = {
                        "result": "正确",
                        "score": "100"
                    }
                else:
                    result = {
                        "result": "错误",
                        "score": "0"
                    }
    
            total_count += 1
            result["question"] = q
            result["answer"] = output_text
            result["traing_answer"] = a
            json.dump(result, f, indent=4, ensure_ascii=False)
            f.write("\n")
            
            total_score += int(result["score"])
            
            if result["result"] == "正确":
                total_correct += 1
            
            f.flush()
            
            bar.update(1)
            bar.set_postfix_str(f"total score: {total_score}, evarage score: {total_score/total_count}, correct rate: {total_correct/total_count}")

        f.write(f"total score: {total_score}, evarage score: {total_score/total_count}, correct rate: {total_correct/total_count}")
        print(f"total score: {total_score}, evarage score: {total_score/total_count}")
    total_score = 0
    total_correct = 0