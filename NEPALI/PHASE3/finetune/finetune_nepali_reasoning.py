"""
NEPALI/PHASE3/finetune/finetune_nepali_reasoning.py

Finetunes Model L (Nepali) on the reasoning dataset, starting from the Phase 2
pretrained checkpoint. Stops early once val accuracy stops improving.
"""

import sys
import json
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from src.model import GPTLanguageModel
from src.dataset import SentencePieceTokenizer
from src.trainer import get_cosine_schedule_with_warmup

IGNORE_INDEX = -100


class ReasoningDataset(torch.utils.data.Dataset):
    """Loads prompt/answer pairs, tokenizes as [BOS]+prompt+answer+[EOS], pads, and
    masks the loss so only the answer tokens count."""

    def __init__(self, jsonl_path, tokenizer, max_seq_len=128):
        self.tokenizer, self.max_seq_len = tokenizer, max_seq_len
        self.examples = self._load(jsonl_path)

    def _load(self, jsonl_path):
        records = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                prompt_ids = self.tokenizer.encode(rec["prompt"], add_bos=True, add_eos=False)
                answer_ids = self.tokenizer.encode(rec["answer"], add_bos=False, add_eos=True)
                if len(prompt_ids) >= self.max_seq_len:
                    continue
                full_ids = (prompt_ids + answer_ids)[: self.max_seq_len]
                answer_start = len(prompt_ids)
                labels = list(full_ids)
                for i in range(min(answer_start, len(labels))):
                    labels[i] = IGNORE_INDEX
                pad_len = self.max_seq_len - len(full_ids)
                if pad_len > 0:
                    full_ids += [self.tokenizer.pad_id] * pad_len
                    labels += [IGNORE_INDEX] * pad_len
                records.append({"input_ids": full_ids, "labels": labels, "template": rec.get("template", "unknown")})
        return records

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        x = torch.tensor(ex["input_ids"][:-1], dtype=torch.long)
        y = torch.tensor(ex["labels"][1:], dtype=torch.long)
        return x, y


def create_reasoning_dataloader(jsonl_path, tokenizer, batch_size=16, max_seq_len=128, shuffle=True):
    dataset = ReasoningDataset(jsonl_path, tokenizer, max_seq_len=max_seq_len)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class ReasoningFinetuner:
    """Same checkpoint format as the Phase 2 Trainer, loads weights only from the
    pretrained checkpoint (fresh optimizer/scheduler)."""

    def __init__(self, model, train_dataloader, val_dataloader, config, output_dir, device="cpu"):
        self.model = model.to(device)
        self.train_dataloader, self.val_dataloader = train_dataloader, val_dataloader
        self.train_dataset = train_dataloader.dataset
        self.train_batch_size = train_dataloader.batch_size
        self.config, self.device = config, device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # curriculum template rebalancing: shift the training mix from easy templates
        # (transitive, already near-saturated fast) toward hard ones (pairwise/threeway,
        # numeric comparison) as training progresses, instead of a flat 1:1:1 mix
        self.use_template_rebalancing = config.get("use_template_rebalancing", False)
        self.easy_templates = set(config.get("easy_templates", []))
        self.hard_templates = set(config.get("hard_templates", []))
        self.curriculum_easy_start = config.get("curriculum_easy_weight_start", 1.0)
        self.curriculum_easy_end = config.get("curriculum_easy_weight_end", 1.0)
        self.curriculum_hard_start = config.get("curriculum_hard_weight_start", 1.0)
        self.curriculum_hard_end = config.get("curriculum_hard_weight_end", 1.0)
        self.train_templates = [ex["template"] for ex in self.train_dataset.examples] if self.use_template_rebalancing else None

        decay = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
        nodecay = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
        self.optimizer = AdamW(
            [{"params": decay, "weight_decay": config.get("weight_decay", 0.01)},
             {"params": nodecay, "weight_decay": 0.0}],
            lr=config.get("learning_rate", 3e-5), betas=tuple(config.get("betas", (0.9, 0.95))),
        )
        self.max_steps = config.get("max_steps", 2000)
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, config.get("warmup_steps", 100), self.max_steps)
        self.grad_clip = config.get("grad_clip", 1.0)
        self.eval_interval = config.get("eval_interval", 50)
        self.save_interval = config.get("save_interval", 100)
        # Named checkpoint_step_N.pt snapshots every milestone_interval steps, on top of
        # checkpoint_latest.pt -- lets sweep_forgetting_vs_steps_*.py evaluate multiple
        # points along one run instead of needing a separate finetune per step count.
        self.milestone_interval = config.get("milestone_interval", 0)
        # Early stopping on the real task metric (answer-token accuracy), not raw val
        # loss -- val loss rises from overconfidence long before accuracy actually
        # degrades on this task, so stopping on loss alone would pick an undertrained
        # checkpoint. checkpoint_final.pt is kept in sync with the best-accuracy state
        # seen so far throughout training, so it's always the checkpoint to use.
        self.patience = config.get("early_stopping_patience", 6)
        self.best_accuracy = -1.0
        self.patience_counter = 0
        self.global_step, self.epoch, self.history = 0, 0, []

    def _rebuild_train_dataloader(self):
        """Recomputes per-example sample weights from the current curriculum position
        (global_step / max_steps) and rebuilds the train dataloader with a weighted
        sampler. Called once per epoch, not per step -- cheap enough and the curriculum
        only needs to shift gradually."""
        if not self.use_template_rebalancing:
            return
        progress = min(1.0, self.global_step / max(1, self.max_steps))
        easy_w = self.curriculum_easy_start + progress * (self.curriculum_easy_end - self.curriculum_easy_start)
        hard_w = self.curriculum_hard_start + progress * (self.curriculum_hard_end - self.curriculum_hard_start)
        weights = [
            easy_w if t in self.easy_templates else (hard_w if t in self.hard_templates else 1.0)
            for t in self.train_templates
        ]
        sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.train_batch_size, sampler=sampler)
        print(f"[Finetuner] Curriculum refreshed at step {self.global_step}: easy_weight={easy_w:.2f} hard_weight={hard_w:.2f}")

    def load_pretrained(self, ckpt_path):
        state = torch.load(Path(ckpt_path), map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
        print(f"[Finetuner] Initialized from pretrained checkpoint -> {ckpt_path}")

    def _compute_loss(self, x, labels):
        logits, _ = self.model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=IGNORE_INDEX)
        return logits, loss

    def save_checkpoint(self, filename=None):
        state = {
            "global_step": self.global_step, "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config, "history": self.history,
        }
        torch.save(state, self.output_dir / "checkpoint_latest.pt")
        if filename:
            torch.save(state, self.output_dir / filename)
        with open(self.output_dir / "finetune_history.json", "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

    def load_checkpoint(self, ckpt_path):
        state = torch.load(Path(ckpt_path), map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.scheduler.load_state_dict(state["scheduler_state_dict"])
        self.global_step = state.get("global_step", 0)
        self.epoch = state.get("epoch", 0)
        self.history = state.get("history", [])
        print(f"[Finetuner] Resumed at step {self.global_step}")

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss, total_batches, correct, total = 0.0, 0, 0, 0
        for x, y in self.val_dataloader:
            x, y = x.to(self.device), y.to(self.device)
            logits, loss = self._compute_loss(x, y)
            total_loss += loss.item()
            total_batches += 1
            preds = logits.argmax(dim=-1)
            mask = y != IGNORE_INDEX
            correct += (preds[mask] == y[mask]).sum().item()
            total += mask.sum().item()
        self.model.train()
        return {"val_loss": total_loss / max(1, total_batches), "answer_token_accuracy": correct / max(1, total)}

    def train(self, resume_path=None):
        if resume_path and Path(resume_path).exists():
            self.load_checkpoint(resume_path)
        print(f"[Finetuner] Starting on {self.device}. Target steps: {self.max_steps}")
        self.model.train()
        start_time = time.time()
        while self.global_step < self.max_steps:
            self.epoch += 1
            self._rebuild_train_dataloader()
            for x, y in self.train_dataloader:
                if self.global_step >= self.max_steps:
                    break
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                _, loss = self._compute_loss(x, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                self.scheduler.step()
                self.global_step += 1

                if self.global_step % self.eval_interval == 0 or self.global_step == 1:
                    m = self.evaluate()
                    lr_curr = self.scheduler.get_last_lr()[0]
                    self.history.append({
                        "step": self.global_step, "train_loss": loss.item(), "val_loss": m["val_loss"],
                        "answer_token_accuracy": m["answer_token_accuracy"], "lr": lr_curr,
                        "elapsed_sec": round(time.time() - start_time, 2),
                    })
                    print(f"Step {self.global_step:5d}/{self.max_steps} | Train Loss: {loss.item():.4f} | "
                          f"Val Loss: {m['val_loss']:.4f} | Answer-token Acc: {m['answer_token_accuracy']:.4f} | LR: {lr_curr:.6f}")

                    if m["answer_token_accuracy"] > self.best_accuracy:
                        self.best_accuracy = m["answer_token_accuracy"]
                        self.patience_counter = 0
                        self.save_checkpoint(filename="checkpoint_final.pt")
                        print(f"  -> New best answer-token accuracy ({self.best_accuracy:.4f}); saved as checkpoint_final.pt")
                    else:
                        self.patience_counter += 1
                        if self.patience_counter >= self.patience:
                            print(f"[Finetuner] Early stopping: no improvement in answer-token accuracy for "
                                  f"{self.patience} eval checks (best={self.best_accuracy:.4f}).")
                            self.save_checkpoint()
                            print(f"[Finetuner] Done (early stop). Log -> {self.output_dir / 'finetune_history.json'}")
                            return

                if self.global_step % self.save_interval == 0:
                    self.save_checkpoint()
                if self.milestone_interval and self.global_step % self.milestone_interval == 0:
                    self.save_checkpoint(filename=f"checkpoint_step_{self.global_step}.pt")

        self.save_checkpoint()
        print(f"[Finetuner] Done (reached max_steps). Best answer-token accuracy: {self.best_accuracy:.4f}. "
              f"Log -> {self.output_dir / 'finetune_history.json'}")


def main():
    phase3_dir = Path(__file__).resolve().parents[1]
    language_dir = phase3_dir.parent
    config_path = phase3_dir / "configs" / "model_l_finetune_config.json"

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    tokenizer_path = (language_dir / config["tokenizer_path"]).resolve()
    pretrained_ckpt = (language_dir / config["pretrained_checkpoint"]).resolve()
    data_dir = phase3_dir / "reasoning_data"
    output_dir = phase3_dir / "finetune" / "checkpoints"

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"=== Finetuning Model L (Nepali) on reasoning data | device={device} ===")

    tokenizer = SentencePieceTokenizer(str(tokenizer_path))
    finetune_seq_len = config.get("finetune_max_seq_len", 128)
    train_loader = create_reasoning_dataloader(
        data_dir / "train.jsonl", tokenizer,
        batch_size=config.get("batch_size", 16), max_seq_len=finetune_seq_len, shuffle=True,
    )
    val_loader = create_reasoning_dataloader(
        data_dir / "val.jsonl", tokenizer,
        batch_size=config.get("batch_size", 16), max_seq_len=finetune_seq_len, shuffle=False,
    )
    print(f"Train examples: {len(train_loader.dataset)} | Val examples: {len(val_loader.dataset)}")

    model = GPTLanguageModel(
        vocab_size=config.get("vocab_size", 12000), d_model=config.get("d_model", 512),
        n_layer=config.get("n_layer", 6), n_head=config.get("n_head", 8),
        d_ff=config.get("d_ff", 2048), max_seq_len=config.get("max_seq_len", 512),
        dropout=config.get("dropout", 0.1), tie_weights=config.get("tie_weights", True),
    )

    finetuner = ReasoningFinetuner(model, train_loader, val_loader, config, output_dir, device)
    finetuner.load_pretrained(pretrained_ckpt)
    resume_ckpt = output_dir / "checkpoint_latest.pt"
    finetuner.train(resume_path=resume_ckpt if resume_ckpt.exists() else None)


if __name__ == "__main__":
    main()
