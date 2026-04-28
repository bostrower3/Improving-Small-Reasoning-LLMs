from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import wandb
import torch


@dataclass
class LoggingConfig:
    output_root: str = "outputs/summation"
    use_wandb: bool = False
    wandb_project: str = "latent-tokens"
    wandb_entity: Optional[str] = None
    checkpoint_every: int = 1
    save_best: bool = True
    save_last: bool = True


class ExperimentLogger:
    """
    Handles:
    - local metric history
    - checkpointing
    - saving final trained parameters
    - optional W&B logging
    """

    def __init__(
        self,
        cfg: LoggingConfig,
        exp_name: str,
        seed: int,
        model_config: Any,
        train_hparams: Dict[str, Any],
    ) -> None:
        self.cfg = cfg
        self.exp_name = exp_name
        self.seed = seed
        self.best_val_loss = float("inf")

        self.run_dir = Path(cfg.output_root) / exp_name / f"seed_{seed}"
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "lr": [],
            "test_metrics": {},
        }

        model_cfg_dict = asdict(model_config) if hasattr(model_config, "__dataclass_fields__") else dict(model_config)

        metadata = {
            "exp_name": exp_name,
            "seed": seed,
            "model_config": model_cfg_dict,
            "train_hparams": train_hparams,
            "logging_config": asdict(cfg),
        }
        with open(self.run_dir / "run_config.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        self.wandb_run = None
        if cfg.use_wandb:
            try:
                self.wandb_run = wandb.init(
                    project=cfg.wandb_project,
                    entity=cfg.wandb_entity,
                    name=f"{exp_name}_seed{seed}",
                    config=metadata,
                    reinit=True,
                )
            except Exception as e:
                print(f"[WARN] W&B init failed, continuing without W&B: {e}")
                self.wandb_run = None

    @staticmethod
    def _latent_state_dict(model) -> Dict[str, torch.Tensor]:
        # Only trainable parameters for this project.
        return model.latent_embeddings.state_dict()

    def _save_checkpoint_file(
        self,
        path: Path,
        model,
        optimizer,
        scheduler,
        epoch: int,
        val_loss: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = {
            "epoch": epoch,
            "val_loss": val_loss,
            "latent_state_dict": self._latent_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "model_name": model.config.model_name,
            "model_config": asdict(model.config) if hasattr(model.config, "__dataclass_fields__") else None,
            "extra": extra or {},
        }
        torch.save(payload, path)

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        lr: float,
        model,
        optimizer,
        scheduler,
    ) -> None:
        self.history["train_loss"].append(float(train_loss))
        self.history["val_loss"].append(float(val_loss))
        self.history["lr"].append(float(lr))

        if self.wandb_run is not None:
            self.wandb_run.log(
                {
                    "epoch": epoch,
                    "train/loss": float(train_loss),
                    "val/loss": float(val_loss),
                    "train/lr": float(lr),
                },
                step=epoch,
            )

        if self.cfg.checkpoint_every > 0 and (epoch % self.cfg.checkpoint_every == 0):
            self._save_checkpoint_file(
                path=self.ckpt_dir / f"epoch_{epoch:03d}.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                val_loss=val_loss,
            )

        if self.cfg.save_best and val_loss < self.best_val_loss:
            self.best_val_loss = float(val_loss)
            self._save_checkpoint_file(
                path=self.ckpt_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                val_loss=val_loss,
                extra={"is_best": True},
            )

    def log_test_metrics(self, metrics: Dict[str, float]) -> None:
        self.history["test_metrics"] = {k: float(v) for k, v in metrics.items()}

        if self.wandb_run is not None:
            self.wandb_run.log({f"test/{k}": float(v) for k, v in metrics.items()})

    def save_final(self, model, optimizer, scheduler, final_epoch: int) -> None:
        # Final lightweight "trained params" artifact.
        torch.save(
            {
                "latent_state_dict": self._latent_state_dict(model),
                "model_name": model.config.model_name,
                "model_config": asdict(model.config) if hasattr(model.config, "__dataclass_fields__") else None,
                "final_epoch": final_epoch,
            },
            self.run_dir / "latent_params_final.pt",
        )

        if self.cfg.save_last:
            self._save_checkpoint_file(
                path=self.ckpt_dir / "last.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=final_epoch,
                val_loss=self.history["val_loss"][-1] if self.history["val_loss"] else float("nan"),
            )

        with open(self.run_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

    def close(self) -> None:
        if self.wandb_run is not None:
            self.wandb_run.finish()


def load_training_checkpoint(path: str, model, optimizer=None, scheduler=None, map_location: str = "cpu") -> Dict[str, Any]:
    """
    Restore latent params + optimizer/scheduler state to resume training.
    """
    ckpt = torch.load(path, map_location=map_location)

    model.latent_embeddings.load_state_dict(ckpt["latent_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    return ckpt