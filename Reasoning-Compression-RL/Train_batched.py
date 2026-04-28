import os
import re
import math
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
        config=cfg,
    )


# ------------------------------------------------
# Dataset
# ------------------------------------------------

def load_train_dataset(name, data_path=None, max_train_samples=None):
    if name.lower() == "gsm8k":
        ds = load_dataset("gsm8k", "main")["train"]
        if max_train_samples is not None:
            ds = ds.select(range(min(max_train_samples, len(ds))))
        return list(zip(ds["question"], ds["answer"]))

    if name.lower() == "compression_dataset":
        from datasets import load_from_disk
        ds = load_from_disk(data_path)
        ds = ds.filter(lambda x: x["extracted"] is not None and x["extracted"] != "")
        if max_train_samples is not None:
            ds = ds.select(range(min(max_train_samples, len(ds))))
        print(f"Loaded {len(ds)} problems with parseable answers")
        return list(zip(ds["problem"], ds["extracted"]))

    raise ValueError("Dataset not supported")


# ------------------------------------------------
# Prompt / answer parsing
# ------------------------------------------------

def build_prompt(tokenizer, q):
    messages = [
        {"role": "user", "content": f"Please reason step by step, and put your final answer within \\boxed{{}}.\n\nQuestion: {q}"}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_answer(text):
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match.group(1).strip()
    match = re.search(r"final answer:\s*(-?\d+\.?\d*)", text, re.IGNORECASE)
    if match:
        return match.group(1)
    nums = re.findall(r"-?\d+\.?\d*", text)
    if nums:
        return nums[-1]
    return None


def extract_gold(answer):
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
# Reward / advantage
# ------------------------------------------------

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def token_length(tokenizer, text):
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def compute_rewards_for_prompt(responses, gold_answer, tokenizer, alpha=0.1, eps=1e-6):
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
    k = len(rewards)
    adv = []
    for i in range(k):
        baseline = 0.0 if k == 1 else (sum(rewards) - rewards[i]) / (k - 1)
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
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


@torch.no_grad()
def sample_batch_rollouts(model, tokenizer, accelerator, prompts, num_rollouts, max_new_tokens, temperature, top_p, max_prompt_length):
    """Generate B * num_rollouts sampled responses in one model.generate call."""
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
    )
    inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}

    gen_model = accelerator.unwrap_model(model)
    was_training = gen_model.training
    gen_model.eval()

    outputs = gen_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=num_rollouts,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        remove_invalid_values=True,
    )

    if was_training:
        gen_model.train()

    prompt_width = inputs["input_ids"].shape[1]
    gen_ids = outputs[:, prompt_width:]

    # If pad_token_id == eos_token_id, this masks generated EOS too. That is okay for the PPO loss.
    gen_mask = (gen_ids != tokenizer.pad_token_id).long()

    prompt_ids = inputs["input_ids"].repeat_interleave(num_rollouts, dim=0)
    prompt_mask = inputs["attention_mask"].repeat_interleave(num_rollouts, dim=0)
    gen_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

    return {
        "prompt_ids": prompt_ids,
        "prompt_mask": prompt_mask,
        "gen_ids": gen_ids,
        "gen_mask": gen_mask,
        "gen_texts": gen_texts,
    }


def get_generated_token_logprobs_batch(model, prompt_ids, prompt_mask, gen_ids, gen_mask):
    """
    Batched generated-token logprob extraction.

    Important numerical fixes:
    1. Compute log_softmax in fp32.
    2. Do not gather pad/eos-masked generated token IDs directly.
    3. Do not multiply -inf by 0; use masked_fill instead.
    """
    full_ids = torch.cat([prompt_ids, gen_ids], dim=1)
    full_mask = torch.cat([prompt_mask, gen_mask], dim=1)

    outputs = model(input_ids=full_ids, attention_mask=full_mask)

    # fp32 log-softmax is much safer than fp16 here
    logits = outputs.logits[:, :-1, :].float()
    target_ids = full_ids[:, 1:].clone()

    prompt_width = prompt_ids.shape[1]
    gen_width = gen_ids.shape[1]

    # Positions corresponding to generated tokens in token_logprobs
    gen_start = prompt_width - 1
    gen_end = gen_start + gen_width

    # Make a full target mask aligned with target_ids/logits
    target_mask = full_mask[:, 1:].clone()
    gen_target_mask = target_mask[:, gen_start:gen_end]

    # Avoid gathering invalid/padded token IDs at masked positions
    target_ids = target_ids.masked_fill(target_mask == 0, 0)

    log_probs = torch.log_softmax(logits, dim=-1)
    token_logprobs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

    gen_logprobs = token_logprobs[:, gen_start:gen_end]

    # Never do gen_logprobs * mask if gen_logprobs may contain -inf/nan.
    # Replace inactive positions with 0 instead.
    gen_logprobs = gen_logprobs.masked_fill(gen_target_mask == 0, 0.0)

    if not torch.isfinite(gen_logprobs).all():
        raise RuntimeError(
            "Non-finite generated logprobs detected inside get_generated_token_logprobs_batch. "
            f"logits finite={torch.isfinite(logits).all().item()}, "
            f"log_probs finite={torch.isfinite(log_probs).all().item()}, "
            f"token_logprobs finite={torch.isfinite(token_logprobs).all().item()}"
        )

    return gen_logprobs, gen_target_mask


def ppo_loss_batched(model, rollout_batch, advantages, clip_eps):
    new_logprobs, mask = get_generated_token_logprobs_batch(
        model,
        rollout_batch["prompt_ids"],
        rollout_batch["prompt_mask"],
        rollout_batch["gen_ids"],
        rollout_batch["gen_mask"],
    )

    old_logprobs = rollout_batch["old_logprobs"]

    advantages = torch.as_tensor(
        advantages,
        device=new_logprobs.device,
        dtype=new_logprobs.dtype
    ).view(-1, 1)

    # Safety: inactive tokens should not participate in ratio at all.
    new_logprobs = new_logprobs.masked_fill(mask == 0, 0.0)
    old_logprobs = old_logprobs.masked_fill(mask == 0, 0.0)

    log_ratio = new_logprobs - old_logprobs
    log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)

    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages

    token_obj = torch.min(unclipped, clipped) * mask

    denom = mask.sum(dim=1).clamp_min(1)
    per_seq_loss = -(token_obj.sum(dim=1) / denom)
    loss = per_seq_loss.mean()

    if not torch.isfinite(loss):
        raise RuntimeError(
            "Non-finite PPO loss detected. "
            f"new_logprobs finite={torch.isfinite(new_logprobs).all().item()}, "
            f"old_logprobs finite={torch.isfinite(old_logprobs).all().item()}, "
            f"ratio finite={torch.isfinite(ratio).all().item()}, "
            f"advantages finite={torch.isfinite(advantages).all().item()}, "
            f"mask min/max={mask.min().item()}/{mask.max().item()}, "
            f"tokens per seq min/max={mask.sum(dim=1).min().item()}/{mask.sum(dim=1).max().item()}"
        )

    return loss


# ------------------------------------------------
# Training
# ------------------------------------------------

def train(cfg):
    exp = cfg["experiment"]

    accelerator = Accelerator(
        gradient_accumulation_steps=exp["grad_accum_steps"],
        mixed_precision="bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "fp16",
    )

    output_dir = exp["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    dataset = load_train_dataset(
        cfg["dataset"],
        cfg.get("data_path"),
        max_train_samples=exp.get("max_train_samples"),
    )

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"],
        torch_dtype=dtype,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = maybe_wrap_lora(model, cfg)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=exp["lr"])
    global_step = 0

    for epoch in range(exp["epochs"]):
        iterator = range(0, len(dataset), exp["batch_size"])
        if accelerator.is_main_process:
            iterator = tqdm(iterator, desc=f"Epoch {epoch + 1}/{exp['epochs']}")

        epoch_reward_sum = 0.0
        epoch_acc_sum = 0.0
        epoch_len_sum = 0.0
        epoch_loss_sum = 0.0
        epoch_batches = 0

        for start in iterator:
            batch = dataset[start : start + exp["batch_size"]]
            questions = [q for q, _ in batch]
            gold_answers = [extract_gold(gold) for _, gold in batch]
            prompts = [build_prompt(tokenizer, q) for q in questions]

            with accelerator.accumulate(model):
                rollout_batch = sample_batch_rollouts(
                    model=model,
                    tokenizer=tokenizer,
                    accelerator=accelerator,
                    prompts=prompts,
                    num_rollouts=exp["num_rollouts"],
                    max_new_tokens=exp["max_new_tokens"],
                    temperature=exp["temperature"],
                    top_p=exp["top_p"],
                    max_prompt_length=exp["max_prompt_length"],
                )

                with torch.no_grad():
                    old_logprobs, _ = get_generated_token_logprobs_batch(
                        model,
                        rollout_batch["prompt_ids"],
                        rollout_batch["prompt_mask"],
                        rollout_batch["gen_ids"],
                        rollout_batch["gen_mask"],
                    )
                rollout_batch["old_logprobs"] = old_logprobs.detach()

                all_advantages = []
                batch_reward = 0.0
                batch_acc = 0.0
                batch_len = 0.0

                R = exp["num_rollouts"]
                for i, gold_answer in enumerate(gold_answers):
                    lo, hi = i * R, (i + 1) * R
                    responses = rollout_batch["gen_texts"][lo:hi]

                    rewards, correct_flags, lengths = compute_rewards_for_prompt(
                        responses,
                        gold_answer,
                        tokenizer,
                        alpha=exp["alpha"],
                    )
                    all_advantages.extend(rloo_advantages(rewards))

                    batch_reward += sum(rewards) / len(rewards)
                    batch_acc += sum(correct_flags) / len(correct_flags)
                    batch_len += sum(lengths) / len(lengths)

                loss = ppo_loss_batched(model, rollout_batch, all_advantages, exp["clip_eps"])
                accelerator.backward(loss)

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

                global_step += 1

                n_prompts = max(len(batch), 1)
                avg_reward = batch_reward / n_prompts
                avg_acc = batch_acc / n_prompts
                avg_len = batch_len / n_prompts

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
                        "reward": f"{running_reward:.3f}",
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
                f"\nEpoch {epoch + 1} summary | "
                f"loss={epoch_loss:.4f} | acc={epoch_acc:.3f} | "
                f"len={epoch_len:.1f} | reward={epoch_reward:.3f}"
            )
            if wandb.run:
                wandb.log({
                    "epoch/loss": epoch_loss,
                    "epoch/accuracy": epoch_acc,
                    "epoch/length": epoch_len,
                    "epoch/reward": epoch_reward,
                    "epoch_number": epoch + 1,
                })

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
