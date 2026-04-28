from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class LatentTokenConfig:
    """
    Minimal config for the latent-token model implementation.
    Only includes settings that are directly used by model.py.
    """
    #Base Model:

    #Latent tokens paper uses Llama-3.2-1B and Llama-3.1-8B.
    model_name: str = "meta-llama/Llama-3.2-1B"         # HuggingFace model id for pretrained model
    # Precision used when loading the backbone:
    # - "auto" lets HF choose based on environment/checkpoint
    # - fp16/bf16 are common for GPU memory savings
    torch_dtype: Literal["auto", "float16", "bfloat16", "float32"] = "auto"

    #Optional device placement strategy forwarded to HF (e.g., "auto" for sharded placement).
    device_map: Optional[str] = None

    # Latent token design
    # Number of latent tokens per insertion group (paper notation: m in k_m / Start_m / End_m).
    num_latent_tokens: int = 2

    # Where latent groups are inserted:
    # "start": Start_m baseline (prompt-tuning style)
    # "end": End_m baseline (pause/filler-like placement)
    # "periodic": k_m style (every k verbal tokens)
    # "comma": task-specific delimiter insertion (e.g., Comma_m in synthetic tasks)
    insertion_strategy: Literal["start", "end", "periodic", "comma"] = "comma"

    # k for periodic insertion (used only when insertion_strategy == "periodic").
    insertion_period: int = 8

    # When True, latent tokens reuse the position id of their CORRESPONDING verbal token. (default=True in the paper)
    freeze_position_ids: bool = True

    # False: latent tokens are PREPENDED before anchor tokens (paper default).
    # True: latent tokens are APPENDED after anchor tokens (tested in their ablation studies)
    append_mode: bool = False

    # Label value ignored by CrossEntropy loss; used for latent positions and masked verbal positions.
    ignore_index: int = -100 #need to double check based on model.

    # Upper bound on autoregressive decoding length during model.generate(). 
    max_new_tokens: int = 128 #this is varied in some of their synthetic tasks

    # Stop token id for generation; if None, generation stops only at max_new_tokens.
    eos_token_id: Optional[int] = None

