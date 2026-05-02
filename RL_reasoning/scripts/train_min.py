""" GRPO training 

1) Subsample the training split via OMSCS_MAX_TRAIN_SAMPLES env var. We
   monkey-patch datasets.load_dataset so the subsample is applied before the
   trainer ever sees the data.

2) Inject fields that newer TRL doesn't define as GRPOConfig attributes
   but that open-rs reads via direct attribute access (system_prompt,
   chat_template, callbacks, benchmarks, hub revision flags). 

Usage:
    OMSCS_MAX_TRAIN_SAMPLES=1000 \\
    accelerate launch --num_processes=1 --mixed_precision=bf16 \\
        OMSCS/scripts/train_min.py \\
        --config OMSCS/configs/grpo_colab_a100.yaml
"""
import os
import sys

import datasets
import yaml


_orig_load_dataset = datasets.load_dataset


def _patched_load_dataset(*args, **kwargs):
    ds = _orig_load_dataset(*args, **kwargs)
    n = int(os.environ.get("OMSCS_MAX_TRAIN_SAMPLES", "0"))
    if n <= 0:
        return ds
    if isinstance(ds, datasets.DatasetDict):
        for split in ds:
            sz = min(n, len(ds[split]))
            ds[split] = ds[split].shuffle(seed=42).select(range(sz))
            print(f"[train_min] subsampled split={split} -> {sz} rows", file=sys.stderr)
    elif isinstance(ds, datasets.Dataset):
        sz = min(n, len(ds))
        ds = ds.shuffle(seed=42).select(range(sz))
        print(f"[train_min] subsampled -> {sz} rows", file=sys.stderr)
    return ds


datasets.load_dataset = _patched_load_dataset


# Fields open-rs reads off training_args that newer TRL doesn't declare on
# GRPOConfig. (field, default).
OPENRS_EXTRA_FIELDS = [
    ("system_prompt", None),
    ("chat_template", None),
    ("callbacks", []),
    ("benchmarks", []),
    ("hub_model_revision", "main"),
    ("overwrite_hub_revision", False),
    ("push_to_hub_revision", False),
]


if __name__ == "__main__":
    from trl import TrlParser
    from trl import GRPOConfig, ModelConfig
    from open_r1.grpo import GRPOScriptArguments, main

    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    raw = {}
    if "--config" in sys.argv:
        cfg_path = sys.argv[sys.argv.index("--config") + 1]
        with open(cfg_path) as f:
            raw = yaml.safe_load(f) or {}

    for fld, default in OPENRS_EXTRA_FIELDS:
        if not hasattr(training_args, fld):
            value = raw.get(fld, default)
            setattr(training_args, fld, value)
            print(f"[train_min] injected training_args.{fld} = {value!r}", file=sys.stderr)

    main(script_args, training_args, model_args)
