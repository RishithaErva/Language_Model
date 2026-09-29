"""
src/trainer.py

Pretraining engine for decoder-only Transformer LMs with mandatory checkpoint resumption.
Supports:
- AdamW optimizer (learning rate, betas, weight decay).
- Cosine Decay Learning Rate Scheduler with linear warmup.
- Gradient norm clipping.
- Mandatory checkpoint saving and full state restoration (model, optimizer, scheduler, step, config).
- Mixed precision training (AMP) when CUDA is available.
- Validation evaluation loop.
"""

import os
import math
import time
import json
from pathlib import Path
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm


def get_cosine_schedule_with_warmup(optimizer, warmup_steps: int, max_steps: int, min_lr_ratio: float = 0.1):
    """
    Cosine decay learning rate schedule with linear warmup.
    """
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


class Trainer:
    """
    Pretraining trainer supporting state saving and checkpoint resumption.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        config: dict,
        output_dir: Path,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        # Hyperparameters
        lr = config.get("learning_rate", 6e-4)
        weight_decay = config.get("weight_decay", 0.1)
        betas = tuple(config.get("betas", (0.9, 0.95)))
        
        # Configure weight decay (apply decay only to 2D parameters like weights, not biases or LayerNorm)
        decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
        nodecay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0}
        ]
        
        self.optimizer = AdamW(optim_groups, lr=lr, betas=betas)
        
        self.max_steps = config.get("max_steps", 10000)
        self.warmup_steps = config.get("warmup_steps", 500)
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, self.warmup_steps, self.max_steps)
        
        self.grad_clip = config.get("grad_clip", 1.0)
        self.eval_interval = config.get("eval_interval", 500)
        self.save_interval = config.get("save_interval", 1000)
        # A uniquely-named checkpoint_step_X.pt "milestone" snapshot is written only every
        # milestone_interval steps (not every save_interval) -- checkpoint_latest.pt alone is
        # what resume actually reads, so writing a brand-new large file every save_interval is
        # pure storage growth for no functional benefit. Old milestones beyond
        # keep_last_n_checkpoints are deleted automatically, so on-disk usage stays bounded no
        # matter how long training runs or how many times it's resumed.
        self.milestone_interval = config.get("milestone_interval", 5000)
        self.keep_last_n_checkpoints = config.get("keep_last_n_checkpoints", 2)

        self.global_step = 0
        self.epoch = 0
        self.history = []

        # AMP Scaler for GPU acceleration
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    def save_checkpoint(self, filename: str = None, prune: bool = False):
        """
        Always (over)writes checkpoint_latest.pt -- the sole file `load_checkpoint`/resume
        reads, so this alone is enough to survive an interrupted session. If `filename` is
        given, an additional uniquely-named copy is written under that name (used for
        `checkpoint_final.pt` and for periodic milestone snapshots). When `prune=True`, older
        checkpoint_step_*.pt milestones beyond `keep_last_n_checkpoints` are deleted right
        after, so milestone storage never grows unbounded.
        """
        state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
            "history": self.history
        }

        latest_path = self.output_dir / "checkpoint_latest.pt"
        torch.save(state, latest_path)
        print(f"[Trainer] Saved checkpoint -> {latest_path}")

        if filename:
            named_path = self.output_dir / filename
            torch.save(state, named_path)
            print(f"[Trainer] Saved checkpoint -> {named_path}")

        if prune:
            self._prune_old_checkpoints()

        # Keep training_history.json in sync with self.history on every save, not just at the
        # end of train() -- so a session that gets cut off mid-run (e.g. Kaggle's 12h cap)
        # still leaves behind real loss/PPL data to plot, instead of nothing at all. self.history
        # is also saved inside the checkpoint itself, but this plain-JSON copy is what
        # generate_report_visuals.py reads, and it's what the notebook's background sync loop
        # actually pushes to Drive every 5 minutes.
        history_path = self.output_dir / "training_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

    def _prune_old_checkpoints(self):
        """
        Keeps only the `keep_last_n_checkpoints` most recent checkpoint_step_*.pt milestone
        snapshots, deleting older ones. checkpoint_latest.pt and checkpoint_final.pt are never
        touched here -- they aren't named checkpoint_step_*.pt.
        """
        if self.keep_last_n_checkpoints <= 0:
            return
        milestones = sorted(
            self.output_dir.glob("checkpoint_step_*.pt"),
            key=lambda p: int(p.stem.rsplit("_", 1)[-1])
        )
        for old_ckpt in milestones[:-self.keep_last_n_checkpoints]:
            old_ckpt.unlink()
            print(f"[Trainer] Pruned old checkpoint -> {old_ckpt.name}")

    def load_checkpoint(self, ckpt_path: Path):
        """
        Resumes training from a saved checkpoint.
        """
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            
        print(f"[Trainer] Resuming from checkpoint -> {ckpt_path}")
        state = torch.load(ckpt_path, map_location=self.device)
        
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.scheduler.load_state_dict(state["scheduler_state_dict"])
        self.global_step = state.get("global_step", 0)
        self.epoch = state.get("epoch", 0)
        self.history = state.get("history", [])
        print(f"[Trainer] Successfully resumed at step {self.global_step}")

    @torch.no_grad()
    def evaluate(self) -> float:
        """
        Computes validation loss over val_dataloader.
        """
        self.model.eval()
        total_loss = 0.0
        total_batches = 0

        autocast_cm = torch.amp.autocast('cuda', enabled=(self.device == "cuda")) if hasattr(torch, "amp") else torch.cuda.amp.autocast(enabled=(self.device == "cuda"))

        for x, y in self.val_dataloader:
            x, y = x.to(self.device), y.to(self.device)
            with autocast_cm:
                _, loss = self.model(x, targets=y)
            total_loss += loss.item()
            total_batches += 1

        self.model.train()
        avg_loss = total_loss / max(1, total_batches)
        return avg_loss

    def train(self, resume_path: Path = None):
        """
        Main pretraining execution loop.
        """
        if resume_path and Path(resume_path).exists():
            self.load_checkpoint(resume_path)

        print(f"[Trainer] Starting training on {self.device}. Target steps: {self.max_steps}")
        self.model.train()
        start_time = time.time()
        autocast_cm = torch.amp.autocast('cuda', enabled=(self.device == "cuda")) if hasattr(torch, "amp") else torch.cuda.amp.autocast(enabled=(self.device == "cuda"))

        while self.global_step < self.max_steps:
            self.epoch += 1
            for x, y in self.train_dataloader:
                if self.global_step >= self.max_steps:
                    break

                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)

                with autocast_cm:
                    _, loss = self.model(x, targets=y)

                self.scaler.scale(loss).backward()
                
                # Unscale gradients before clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

                self.global_step += 1

                # Periodic evaluation & logging
                if self.global_step % self.eval_interval == 0 or self.global_step == 1:
                    val_loss = self.evaluate()
                    ppl = math.exp(val_loss)
                    lr_curr = self.scheduler.get_last_lr()[0]
                    elapsed = time.time() - start_time
                    
                    log_entry = {
                        "step": self.global_step,
                        "train_loss": loss.item(),
                        "val_loss": val_loss,
                        "perplexity": ppl,
                        "lr": lr_curr,
                        "elapsed_sec": round(elapsed, 2)
                    }
                    self.history.append(log_entry)
                    
                    print(
                        f"Step {self.global_step:6d}/{self.max_steps} | "
                        f"Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | "
                        f"PPL: {ppl:.2f} | LR: {lr_curr:.6f}"
                    )

                # Periodic checkpointing: checkpoint_latest.pt every save_interval (cheap resume
                # safety net), a named+pruned milestone snapshot only every milestone_interval
                # (keeps on-disk storage bounded -- see save_checkpoint docstring).
                if self.global_step % self.save_interval == 0:
                    is_milestone = self.global_step % self.milestone_interval == 0
                    self.save_checkpoint(
                        filename=f"checkpoint_step_{self.global_step}.pt" if is_milestone else None,
                        prune=is_milestone
                    )

        # Final checkpoint save (not pruned -- checkpoint_final.pt is kept permanently).
        # save_checkpoint() already writes training_history.json as part of every save.
        self.save_checkpoint("checkpoint_final.pt")
        print(f"[Trainer] Training complete. Log saved -> {self.output_dir / 'training_history.json'}")
