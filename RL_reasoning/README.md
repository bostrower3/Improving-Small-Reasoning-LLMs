## Open-RS Reproduction

Paper: [Reinforcement Learning for Reasoning in Small LLMs: What Works and What Doesn't](https://arxiv.org/abs/2503.16219) (Open-RS).
Upstream code: [knoveleng/open-rs](https://github.com/knoveleng/open-rs).

We reproduce the Open-RS GRPO recipe (DeepSeek-R1-Distill-Qwen-1.5B + format/cosine rewards on `knoveleng/open-rs`) at **reduced scale** on a single A100-40GB:
- **1000-sample subset** of the 7K training set
- **100 training steps** (paper: 500)
- **100 problems each** from GSM8K + MATH-500 for eval

We then evaluate on **GSM8K** and **MATH-500** (the benchmarks in our class proposal — the paper itself uses AMC23 / AIME24, so this is a benchmark transfer worth reporting).

## Files

| Path | Purpose |
|---|---|
| `configs/grpo_colab_a100.yaml` | Training config: 1.5B model, 100 steps, single-A100-tuned. |
| `scripts/train_min.py` | Thin wrapper: subsamples dataset via `OMSCS_MAX_TRAIN_SAMPLES` env var, then calls upstream open-r1's `main()`. |
| `scripts/eval_benchmarks.py` | vLLM eval on GSM8K + MATH-500 with configurable subset size and dtype. |
| `scripts/setup_colab.sh` | Clones open-rs, installs deps (lets open-rs resolve its own pins), sanity-checks imports. |


### Setup + baseline + launch training 

**mount Drive, copy project to fast local disk**

```python
from google.colab import drive
drive.mount('/content/drive')

# Tarball expected at /content/drive/MyDrive/OMSCS.tar.gz (re-tar on Cirro after edits)
!rm -rf /content/omscs && mkdir -p /content/omscs
!tar xzf /content/drive/MyDrive/OMSCS.tar.gz -C /content/omscs --strip-components=1
!mkdir -p /content/drive/MyDrive/OMSCS_runs
!ls /content/omscs
```

**install deps**

```python
!bash /content/omscs/scripts/setup_colab.sh
```

```python
!sed -i 's|attn_implementation: flash_attention_2|attn_implementation: sdpa|' \
    /content/omscs/configs/grpo_colab_a100.yaml
```

**point output_dir at Drive (so checkpoints survive disconnects)**

```python
!sed -i 's|^output_dir:.*|output_dir: /content/drive/MyDrive/OMSCS_runs/min_run1|' \
    /content/omscs/configs/grpo_colab_a100.yaml
!grep output_dir /content/omscs/configs/grpo_colab_a100.yaml
```

**baseline eval**

```python
!python /content/omscs/scripts/eval_benchmarks.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --tag baseline \
    --gsm8k_n 100 --math500_n 100 \
    --out /content/drive/MyDrive/OMSCS_runs/results.csv
```

**smoke test**

```python
import os
os.environ["OMSCS_MAX_TRAIN_SAMPLES"] = "1000"

!cd /content/work/open-rs && \
    OMSCS_MAX_TRAIN_SAMPLES=1000 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    accelerate launch --num_processes=1 --mixed_precision=bf16 \
        /content/omscs/scripts/train_min.py \
        --config /content/omscs/configs/grpo_colab_a100.yaml \
        --max_steps=5 \
        --output_dir=/content/drive/MyDrive/OMSCS_runs/smoke
```

**Full 100-step training (~4–5h)**
```python
!cd /content/work/open-rs && \
    OMSCS_MAX_TRAIN_SAMPLES=1000 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    accelerate launch --num_processes=1 --mixed_precision=bf16 \
        /content/omscs/scripts/train_min.py \
        --config /content/omscs/configs/grpo_colab_a100.yaml
```

**final eval on the trained checkpoint**

```python
import glob
ckpts = sorted(glob.glob("/content/drive/MyDrive/OMSCS_runs/min_run1/checkpoint-*"),
               key=lambda p: int(p.rsplit("-", 1)[1]))
final = ckpts[-1]
print("Evaluating", final)

!python /content/omscs/scripts/eval_benchmarks.py \
    --model {final} \
    --tag step{final.rsplit('-', 1)[1]} \
    --gsm8k_n 100 --math500_n 100 \
    --out /content/drive/MyDrive/OMSCS_runs/results.csv
```

**pull TensorBoard for plots**

```python
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/OMSCS_runs/min_run1/runs
```

