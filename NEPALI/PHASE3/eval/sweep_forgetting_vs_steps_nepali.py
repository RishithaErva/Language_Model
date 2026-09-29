"""
NEPALI/PHASE3/eval/sweep_forgetting_vs_steps_nepali.py

Checks general Nepali perplexity (Phase 1 val split) at every saved checkpoint
step, alongside reasoning accuracy, to see where the model starts forgetting.
Run after finetuning has produced checkpoint_step_N.pt milestones.
"""

import sys
import json
import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

import torch
import matplotlib.pyplot as plt
from src.model import GPTLanguageModel
from src.dataset import SentencePieceTokenizer, create_dataloader
from src.metrics import compute_perplexity_and_bpb


def load_model(config, checkpoint_path, device):
    model = GPTLanguageModel(
        vocab_size=config.get("vocab_size", 12000), d_model=config.get("d_model", 512),
        n_layer=config.get("n_layer", 6), n_head=config.get("n_head", 8),
        d_ff=config.get("d_ff", 2048), max_seq_len=config.get("max_seq_len", 512),
        dropout=config.get("dropout", 0.1), tie_weights=config.get("tie_weights", True),
    )
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    return model, state.get("global_step", 0)


def main():
    phase3_dir = Path(__file__).resolve().parents[1]
    language_dir = phase3_dir.parent
    config_path = phase3_dir / "configs" / "model_l_finetune_config.json"
    checkpoints_dir = phase3_dir / "finetune" / "checkpoints"

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"=== Forgetting-vs-steps sweep: Model L (Nepali) | device={device} ===")

    tokenizer_path = (language_dir / config["tokenizer_path"]).resolve()
    tokenizer = SentencePieceTokenizer(str(tokenizer_path))

    general_val_jsonl = language_dir / "PHASE1" / "outputs" / "data_splits" / "val.jsonl"
    cache_dir = Path(__file__).resolve().parent / "cache"
    val_loader = create_dataloader(
        general_val_jsonl, tokenizer, batch_size=32, max_seq_len=config.get("max_seq_len", 512),
        shuffle=False, cache_dir=cache_dir,
    )

    history_path = checkpoints_dir / "finetune_history.json"
    history_by_step = {}
    if history_path.exists():
        history = json.load(open(history_path, "r", encoding="utf-8"))
        # BUG FIX: guard against KeyError if any history entry lacks 'answer_token_accuracy'
        history_by_step = {h["step"]: h["answer_token_accuracy"] for h in history if "answer_token_accuracy" in h}

    def nearest_logged_accuracy(step):
        if not history_by_step:
            return None
        return history_by_step.get(step, history_by_step[min(history_by_step, key=lambda s: abs(s - step))])

    pretrained_ckpt = (language_dir / config["pretrained_checkpoint"]).resolve()
    pretrained_model, _ = load_model(config, pretrained_ckpt, device)
    pretrained_metrics = compute_perplexity_and_bpb(pretrained_model, val_loader, tokenizer, device=device)
    pretrained_ppl = pretrained_metrics["perplexity"]
    print(f"[step 0 / pretrained] general-LM PPL: {pretrained_ppl:.2f}")
    del pretrained_model

    rows = [{"step": 0, "general_lm_ppl": pretrained_ppl, "ppl_pct_change": 0.0, "reasoning_answer_token_acc": None}]

    milestone_paths = sorted(
        checkpoints_dir.glob("checkpoint_step_*.pt"),
        key=lambda p: int(re.search(r"checkpoint_step_(\d+)\.pt", p.name).group(1))
    )
    final_path = checkpoints_dir / "checkpoint_final.pt"
    if final_path.exists():
        milestone_paths.append(final_path)

    for ckpt_path in milestone_paths:
        model, step = load_model(config, ckpt_path, device)
        metrics = compute_perplexity_and_bpb(model, val_loader, tokenizer, device=device)
        ppl = metrics["perplexity"]
        pct = 100.0 * (ppl - pretrained_ppl) / pretrained_ppl
        acc = nearest_logged_accuracy(step)
        rows.append({"step": step, "general_lm_ppl": ppl, "ppl_pct_change": pct, "reasoning_answer_token_acc": acc})
        acc_str = f"{acc:.4f}" if acc is not None else "n/a"
        print(f"[step {step:5d} / {ckpt_path.name}] general-LM PPL: {ppl:8.2f} ({pct:+7.1f}%) | reasoning answer-token acc: {acc_str}")
        del model

    rows = sorted({r["step"]: r for r in rows}.values(), key=lambda r: r["step"])

    results_path = Path(__file__).resolve().parent / "results" / "forgetting_vs_steps_sweep.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\n[sweep] Table saved -> {results_path}")

    safe_rows = [r for r in rows if r["ppl_pct_change"] <= 15.0]
    if safe_rows:
        best = max(safe_rows, key=lambda r: r["step"])
        print(f"[sweep] Recommended stopping point: step {best['step']} "
              f"(general-LM PPL {best['general_lm_ppl']:.2f}, {best['ppl_pct_change']:+.1f}% vs. pretrained; "
              f"reasoning answer-token acc {best['reasoning_answer_token_acc']}).")
    else:
        print("[sweep] WARNING: every checkpoint exceeds the 15% general-LM PPL threshold -- "
              "even the earliest milestone shows forgetting. Consider a lower learning rate "
              "or evaluating even earlier steps.")

    steps = [r["step"] for r in rows]
    ppls = [r["general_lm_ppl"] for r in rows]
    accs = [r["reasoning_answer_token_acc"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax1.plot(steps, ppls, marker="o", color="#d62728", label="General-LM PPL (Phase 1 val split)")
    ax1.axhline(pretrained_ppl, color="#888888", linestyle="--", label="Pretrained baseline PPL")
    ax1.set_xlabel("Finetuning step")
    ax1.set_ylabel("General-LM Perplexity", color="#d62728")
    ax1.tick_params(axis="y", labelcolor="#d62728")

    ax2 = ax1.twinx()
    steps_with_acc = [s for s, a in zip(steps, accs) if a is not None]
    accs_only = [a for a in accs if a is not None]
    ax2.plot(steps_with_acc, accs_only, marker="s", color="#2c7fb8", label="Reasoning answer-token accuracy")
    ax2.set_ylabel("Reasoning answer-token accuracy", color="#2c7fb8")
    ax2.tick_params(axis="y", labelcolor="#2c7fb8")
    ax2.set_ylim(0, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    ax1.set_title("Nepali (Model L): General-LM PPL vs. Reasoning Accuracy over Finetuning Steps")
    plt.tight_layout()

    plot_path = Path(__file__).resolve().parent / "plots" / "nepali_forgetting_vs_steps.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[sweep] Plot saved -> {plot_path}")


if __name__ == "__main__":
    main()
