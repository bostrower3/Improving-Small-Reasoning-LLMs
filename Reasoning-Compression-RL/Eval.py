import json
import yaml
import re
import torch
import wandb

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse


# ------------------------------------------------
# WandB
# ------------------------------------------------

def init_wandb(cfg):

    if "wandb" not in cfg:
        return

    api_file = cfg["wandb"]["api_key_file"]

    with open(api_file) as f:
        key = f.read().strip()

    wandb.login(key=key)

    wandb.init(
        project=cfg["wandb"]["project"],
        entity=cfg["wandb"]["entity"],
        name=cfg["run_name"],
        config=cfg
    )


# ------------------------------------------------
# Dataset
# ------------------------------------------------

def load_eval_dataset(name):

    if name.lower() == "gsm8k":

        ds = load_dataset("gsm8k", "main")["test"]

        return list(zip(ds["question"], ds["answer"]))

    raise ValueError("Dataset not supported")


# ------------------------------------------------
# Prompt
# ------------------------------------------------

# def build_prompt(q):

# #     return f"""
# # Solve the following math problem step by step.

# # Question:
# # {q}

# # At the end write:

# # FINAL ANSWER: <number>
# # """

#     return f"Please reason step by step, and put your final answer within \\boxed{{}}.\n\nQuestion: {q}"

def build_prompt(tokenizer, q):
    messages = [
        {"role": "user", "content": f"Please reason step by step, and put your final answer within \\boxed{{}}.\n\nQuestion: {q}"}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)



# ------------------------------------------------
# Answer extraction
# ------------------------------------------------

def extract_answer(text):

    # match = re.search(r"FINAL ANSWER:\s*(-?\d+\.?\d*)", text)

    # if match:
    #     return match.group(1)

    # match = re.search(r"####\s*(-?\d+\.?\d*)", text)

    # if match:
    #     return match.group(1)

    # nums = re.findall(r"-?\d+\.?\d*", text)

    # if nums:
    #     return nums[-1]

    # return None

    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match.group(1).strip()
    match = re.search(r"####\s*(-?\d+\.?\d*)", text)
    if match:
        return match.group(1)
    nums = re.findall(r"-?\d+\.?\d*", text)
    return nums[-1] if nums else None


def extract_gold(answer):

    if "####" in answer:
        return answer.split("####")[-1].strip()

    return extract_answer(answer)


def normalize(x):

    if x is None:
        return None

    x = x.strip().replace(",", "")

    try:
        x = str(float(x))
    except:
        pass

    return x


# ------------------------------------------------
# Batched generation
# ------------------------------------------------

def generate_batch(model, tokenizer, prompts, max_tokens):

    inputs = tokenizer(
        prompts,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():

        # outputs = model.generate(
        #     **inputs,
        #     max_new_tokens=max_tokens,
        #     do_sample=False
        # )

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
        )

    # decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    prompt_len = inputs["input_ids"].shape[1]
    decoded = tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)

    return decoded


# ------------------------------------------------
# Evaluation
# ------------------------------------------------

def evaluate(cfg):

    batch_size = cfg["experiment"]["batch_size"]
    max_tokens = cfg["experiment"]["max_new_tokens"]
    output_file = cfg["experiment"]["output_file"]

    dataset = load_eval_dataset(cfg["dataset"])

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"],
        torch_dtype=torch.float16,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []

    correct = 0
    total = 0
    total_len = 0 

    for i in tqdm(range(0, len(dataset), batch_size)):

        batch = dataset[i:i+batch_size]

        questions = [q for q, _ in batch]
        answers = [a for _, a in batch]

        prompts = [build_prompt(tokenizer, q) for q in questions]

        outputs = generate_batch(model, tokenizer, prompts, max_tokens)

        for q, out, gold in zip(questions, outputs, answers):

            pred = extract_answer(out)
            gold_ans = extract_gold(gold)

            is_correct = normalize(pred) == normalize(gold_ans)

            if is_correct:
                correct += 1

            total += 1
            out_len = len(tokenizer(out, add_special_tokens=False)["input_ids"])
            total_len += out_len

            entry = {
                "question": q,
                "model_output": out,
                "prediction": pred,
                "gold": gold_ans,
                "correct": is_correct
            }

            results.append(entry)

        acc = correct / total

        avg_len = total_len / total

        if wandb.run:
            wandb.log(
                {
                    "running_accuracy": acc,
                    "running_avg_length": avg_len
                }
            )

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

    final_acc = correct / total
    final_avg_len = total_len / total

    print(f"\nFinal Accuracy: {final_acc:.4f}")
    print(f"\nFinal Avg Len: {final_avg_len:.4f}")

    if wandb.run:

        wandb.log(
            {
                "final_accuracy": final_acc,
                "final_avg_length": final_avg_len
            }
        )

        artifact = wandb.Artifact("eval_results", type="dataset")
        artifact.add_file(output_file)

        wandb.log_artifact(artifact)


# ------------------------------------------------
# Main
# ------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")

    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)


    cfg = cfg['Eval']
    init_wandb(cfg)
    evaluate(cfg)