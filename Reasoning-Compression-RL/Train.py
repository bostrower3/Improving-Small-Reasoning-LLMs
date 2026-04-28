import os
import re
import math
import json
import copy
import yaml
import torch
import wandb
import argparse

from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from accelerate import Accelerator

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

def load_train_dataset(name, data_path=None):
    if name.lower() == "gsm8k":
        ds = load_dataset("gsm8k", "main")["train"]
        ds = ds.select(range(5))
        return list(zip(ds["question"], ds["answer"]))
    
    if name.lower() == "compression_dataset":
        from datasets import load_from_disk
        ds = load_from_disk(data_path)
        ds = ds.select(range(10))

        # Filter out rows where extracted answer is None
        ds = ds.filter(lambda x: x["extracted"] is not None and x["extracted"] != "")
        print(f"Loaded {len(ds)} problems with parseable answers")
        return list(zip(ds["problem"], ds["extracted"]))
    
    raise ValueError("Dataset not supported")


# ------------------------------------------------
# Prompt / answer parsing
# ------------------------------------------------

def build_prompt(tokenizer, q):
    # messages = [
    #     {"role": "user", "content": f"Question: {q}\n\nAnswer briefly. End with: Final answer: <number>"}
    # ]
    # return tokenizer.apply_chat_template(
    #     messages,
    #     tokenize=False,
    #     add_generation_prompt=True
    # )

    messages = [
        {"role": "user", "content": f"Please reason step by step, and put your final answer within \\boxed{{}}.\n\nQuestion: {q}"}
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

def extract_answer(text):
    # m = re.search(
    #     r"final answer:\s*(?:[^\d-]*)(-?\d+(?:\.\d+)?)",
    #     text,
    #     re.IGNORECASE
    # )
    # if m:
    #     return m.group(1)

    # m = re.search(r"####\s*\$?\s*(-?\d+(?:\.\d+)?)", text)
    # if m:
    #     return m.group(1)

    # return None

    # Match \boxed{...}
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match.group(1).strip()
    # Fallback to original patterns
    match = re.search(r"final answer:\s*(-?\d+\.?\d*)", text, re.IGNORECASE)
    if match:
        return match.group(1)
    nums = re.findall(r"-?\d+\.?\d*", text)
    if nums:
        return nums[-1]
    return None


def extract_gold(answer):
    # if "####" in answer:
    #     return answer.split("####")[-1].strip()
    # return extract_answer(answer)

    # For compression_dataset, answer is already extracted
    if "####" in answer:
        return answer.split("####")[-1].strip()
    return answer.strip()


def normalize(x):
    if x is None:
        return None

    x = x.strip().replace(",", "")
    try:
        x = str(float(x))
    except Exception:
        pass
    return x


# ------------------------------------------------
# Utilities
# ------------------------------------------------

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def token_length(tokenizer, text):
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def compute_rewards_for_prompt(responses, gold_answer, tokenizer, alpha=0.1, eps=1e-6):
    """
    Paper-inspired reward:
      reward = correct * (1 - alpha * sigmoid((len - mean_correct_len)/std_correct_len))

    Mean/std are computed per prompt over correct rollouts only.
    """
    preds = [extract_answer(r) for r in responses]
    correct = [1.0 if normalize(p) == normalize(gold_answer) else 0.0 for p in preds]
    lengths = [token_length(tokenizer, r) for r in responses]

    correct_lengths = [L for L, c in zip(lengths, correct) if c == 1.0]

    if len(correct_lengths) == 0:
        return [0.0 for _ in responses], correct, lengths

    mu = sum(correct_lengths) / len(correct_lengths)
    var = sum((L - mu) ** 2 for L in correct_lengths) / max(len(correct_lengths), 1)
    std = math.sqrt(var) + eps

    rewards = []
    for L, c in zip(lengths, correct):
        if c == 0.0:
            rewards.append(0.0)
        else:
            penalty = sigmoid((L - mu) / std)
            rewards.append(1.0 - alpha * penalty)

    return rewards, correct, lengths


def rloo_advantages(rewards):
    """
    Leave-one-out baseline.
    IMPORTANT: no advantage normalization.
    """
    K = len(rewards)
    adv = []
    for i in range(K):
        if K == 1:
            baseline = 0.0
        else:
            baseline = (sum(rewards) - rewards[i]) / (K - 1)
        adv.append(rewards[i] - baseline)
    return adv


# ------------------------------------------------
# Model helpers
# ------------------------------------------------

def maybe_wrap_lora(model, cfg):
    if not cfg["experiment"].get("use_lora", False):
        return model

    lora_config = LoraConfig(
        r=cfg["experiment"].get("lora_r", 8),
        lora_alpha=cfg["experiment"].get("lora_alpha", 16),
        lora_dropout=cfg["experiment"].get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "up_proj", "down_proj", "gate_proj"
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def get_generated_token_logprobs(model, input_ids, attention_mask, generated_ids):
    """
    Returns per-token logprobs of generated_ids under model.
    input_ids: [1, prompt_len]
    generated_ids: [1, gen_len]
    """
    full_ids = torch.cat([input_ids, generated_ids], dim=1)
    full_mask = torch.cat(
        [attention_mask, torch.ones_like(generated_ids, device=attention_mask.device)],
        dim=1
    )

    outputs = model(input_ids=full_ids, attention_mask=full_mask)
    logits = outputs.logits[:, :-1, :]  # predict next token

    target_ids = full_ids[:, 1:]
    log_probs = torch.log_softmax(logits, dim=-1)
    token_logprobs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

    prompt_len = input_ids.shape[1]
    gen_logprobs = token_logprobs[:, prompt_len - 1:]  # generated token positions
    return gen_logprobs


def sample_response(model, tokenizer, accelerator, prompt, max_new_tokens, temperature, top_p, max_prompt_length):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_length
    )
    inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}

    gen_model = accelerator.unwrap_model(model)
    was_training = gen_model.training
    gen_model.eval()

    with torch.no_grad():
        outputs = gen_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            remove_invalid_values=True,
        )

    if was_training:
        gen_model.train()

    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = outputs[:, prompt_len:]
    gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

    return {
        "prompt_ids": inputs["input_ids"],
        "prompt_mask": inputs["attention_mask"],
        "gen_ids": gen_ids,
        "gen_text": gen_text
    }


# ------------------------------------------------
# PPO loss
# ------------------------------------------------

def ppo_loss_from_rollouts(model, rollouts, clip_eps):
    """
    Each rollout contains:
      old_logprobs: [1, T]
      advantage: scalar
      prompt_ids, prompt_mask, gen_ids
    """
    losses = []

    for r in rollouts:
        new_logprobs = get_generated_token_logprobs(
            model,
            r["prompt_ids"],
            r["prompt_mask"],
            r["gen_ids"]
        )

        old_logprobs = r["old_logprobs"]
        adv = torch.tensor(r["advantage"], device=new_logprobs.device, dtype=new_logprobs.dtype)

        ratio = torch.exp(new_logprobs - old_logprobs)
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv

        token_obj = torch.min(unclipped, clipped)
        loss = -token_obj.mean()
        losses.append(loss)

    return torch.stack(losses).mean()


# ------------------------------------------------
# Training
# ------------------------------------------------

def train(cfg):
    exp = cfg["experiment"]

    accelerator = Accelerator(
        gradient_accumulation_steps=exp["grad_accum_steps"],
        mixed_precision="fp16" if torch.cuda.is_available() else "no"
    )

    batch_size = exp["batch_size"]
    grad_accum_steps = exp["grad_accum_steps"]
    epochs = exp["epochs"]
    lr = exp["lr"]
    clip_eps = exp["clip_eps"]
    max_prompt_length = exp["max_prompt_length"]
    max_new_tokens = exp["max_new_tokens"]
    num_rollouts = exp["num_rollouts"]
    alpha = exp["alpha"]
    temperature = exp["temperature"]
    top_p = exp["top_p"]
    output_dir = exp["output_dir"]
    save_every = exp["save_every"]

    os.makedirs(output_dir, exist_ok=True)

    #dataset = load_train_dataset(cfg["dataset"])
    dataset = load_train_dataset(cfg["dataset"], cfg.get("data_path"))

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"],
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = maybe_wrap_lora(model, cfg)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    global_step = 0

    for epoch in range(exp["epochs"]):
        iterator = range(0, len(dataset), exp["batch_size"])

        if accelerator.is_main_process:
            iterator = tqdm(iterator, desc=f"Epoch {epoch+1}/{exp['epochs']}")

        epoch_reward_sum = 0.0
        epoch_acc_sum = 0.0
        epoch_len_sum = 0.0
        epoch_loss_sum = 0.0
        epoch_batches = 0

        for start in iterator:
            batch = dataset[start:start + exp["batch_size"]]

            with accelerator.accumulate(model):
                all_rollouts = []
                batch_reward = 0.0
                batch_acc = 0.0
                batch_len = 0.0
                batch_count = 0

                for question, gold in batch:
                    prompt = build_prompt(tokenizer, question)
                    gold_answer = extract_gold(gold)

                    prompt_rollouts = []
                    response_texts = []

                    for _ in range(exp["num_rollouts"]):
                        s = sample_response(
                            model=model,
                            tokenizer=tokenizer,
                            accelerator=accelerator,
                            prompt=prompt,
                            max_new_tokens=exp["max_new_tokens"],
                            temperature=exp["temperature"],
                            top_p=exp["top_p"],
                            max_prompt_length=exp["max_prompt_length"]
                        )
                        response_texts.append(s["gen_text"])
                        prompt_rollouts.append(s)

                    rewards, correct_flags, lengths = compute_rewards_for_prompt(
                        response_texts,
                        gold_answer,
                        tokenizer,
                        alpha=exp["alpha"]
                    )

                    advantages = rloo_advantages(rewards)

                    for i, s in enumerate(prompt_rollouts):
                        with torch.no_grad():
                            old_lp = get_generated_token_logprobs(
                                model,
                                s["prompt_ids"],
                                s["prompt_mask"],
                                s["gen_ids"]
                            )

                        all_rollouts.append({
                            "prompt_ids": s["prompt_ids"],
                            "prompt_mask": s["prompt_mask"],
                            "gen_ids": s["gen_ids"],
                            "old_logprobs": old_lp.detach(),
                            "advantage": advantages[i]
                        })

                    batch_reward += sum(rewards) / len(rewards)
                    batch_acc += sum(correct_flags) / len(correct_flags)
                    batch_len += sum(lengths) / len(lengths)
                    batch_count += 1

                loss = ppo_loss_from_rollouts(model, all_rollouts, exp["clip_eps"])
                accelerator.backward(loss)

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                optimizer.zero_grad()

                global_step += 1

                avg_reward = batch_reward / max(batch_count, 1)
                avg_acc = batch_acc / max(batch_count, 1)
                avg_len = batch_len / max(batch_count, 1)

                epoch_reward_sum += avg_reward
                epoch_acc_sum += avg_acc
                epoch_len_sum += avg_len
                epoch_loss_sum += loss.item()
                epoch_batches += 1

                running_reward = epoch_reward_sum / epoch_batches
                running_acc = epoch_acc_sum / epoch_batches
                running_len = epoch_len_sum / epoch_batches
                running_loss = epoch_loss_sum / epoch_batches

                if accelerator.is_main_process:
                    iterator.set_postfix({
                        "loss": f"{running_loss:.4f}",
                        "acc": f"{running_acc:.3f}",
                        "len": f"{running_len:.1f}",
                        "reward": f"{running_reward:.3f}"
                    })

                if accelerator.is_main_process and wandb.run and global_step % cfg["wandb"].get("log_every", 1) == 0:
                    wandb.log({
                        "train/batch_loss": loss.item(),
                        "train/batch_reward": avg_reward,
                        "train/batch_accuracy": avg_acc,
                        "train/batch_length": avg_len,
                        "train/running_loss": running_loss,
                        "train/running_reward": running_reward,
                        "train/running_accuracy": running_acc,
                        "train/running_length": running_len,
                        "train/global_step": global_step,
                        "train/epoch": epoch + 1,
                        "train/alpha": exp["alpha"],
                    })

                if accelerator.is_main_process and global_step % exp["save_every"] == 0:
                    save_path = os.path.join(output_dir, f"step_{global_step}")
                    unwrapped = accelerator.unwrap_model(model)
                    unwrapped.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)

        epoch_loss = epoch_loss_sum / max(epoch_batches, 1)
        epoch_reward = epoch_reward_sum / max(epoch_batches, 1)
        epoch_acc = epoch_acc_sum / max(epoch_batches, 1)
        epoch_len = epoch_len_sum / max(epoch_batches, 1)

        if accelerator.is_main_process:
            print(
                f"\nEpoch {epoch+1} summary | "
                f"loss={epoch_loss:.4f} | "
                f"acc={epoch_acc:.3f} | "
                f"len={epoch_len:.1f} | "
                f"reward={epoch_reward:.3f}"
            )

            if wandb.run:
                wandb.log({
                    "epoch/loss": epoch_loss,
                    "epoch/accuracy": epoch_acc,
                    "epoch/length": epoch_len,
                    "epoch/reward": epoch_reward,
                    "epoch_number": epoch + 1,
                })

            if global_step % save_every == 0:
                save_path = os.path.join(output_dir, f"step_{global_step}")
                os.makedirs(save_path, exist_ok=True)
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)

    final_path = os.path.join(output_dir, "final")
    os.makedirs(final_path, exist_ok=True)
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    print(f"Training complete. Final model saved to: {final_path}")


# ------------------------------------------------
# Main
# ------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="train_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cfg = cfg["Train"]
    init_wandb(cfg)
    train(cfg)