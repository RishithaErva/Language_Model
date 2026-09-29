"""
TELUGU/PHASE3/eval/eval_reasoning_telugu.py

Evaluates Model H (Telugu) on the reasoning test set, pretrained vs finetuned --
exact match accuracy per template, some example outputs, and a comparison plot.
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

import torch
import matplotlib.pyplot as plt
from src.model import GPTLanguageModel
from src.dataset import SentencePieceTokenizer


def load_model(config, checkpoint_path, device):
    model = GPTLanguageModel(
        vocab_size=config.get("vocab_size", 12000), d_model=config.get("d_model", 512),
        n_layer=config.get("n_layer", 6), n_head=config.get("n_head", 8),
        d_ff=config.get("d_ff", 2048), max_seq_len=config.get("max_seq_len", 512),
        dropout=0.0,  # BUG FIX: always 0.0 for eval — don't rely on model.eval() alone
        tie_weights=config.get("tie_weights", True),
    )
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def generate_answer(model, tokenizer, prompt, device, max_new_tokens=12):
    model.eval()
    prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    prompt_len = idx.size(1)
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -model.max_seq_len:])
        next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        if next_id.item() == tokenizer.eos_id:
            break
        idx = torch.cat((idx, next_id), dim=1)
    return tokenizer.decode(idx[0, prompt_len:].cpu().tolist())


def exact_match(prediction, reference):
    return prediction.strip() == reference.strip()


@torch.no_grad()  # BUG FIX: prevents gradient tracking in the outer loop even though generate_answer is also decorated
def evaluate_model(model, tokenizer, test_records, device):
    predictions, references, templates, qualitative = [], [], [], []
    for rec in test_records:
        pred = generate_answer(model, tokenizer, rec["prompt"], device=device, max_new_tokens=12)
        predictions.append(pred)
        references.append(rec["answer"])
        templates.append(rec["template"])
        qualitative.append({
            "prompt": rec["prompt"], "reference": rec["answer"], "prediction": pred,
            "template": rec["template"], "correct": exact_match(pred, rec["answer"]),
        })

    overall_acc = sum(1 for p, r in zip(predictions, references) if exact_match(p, r)) / max(1, len(predictions))
    per_template = defaultdict(lambda: {"correct": 0, "total": 0})
    for p, r, t in zip(predictions, references, templates):
        per_template[t]["total"] += 1
        if exact_match(p, r):
            per_template[t]["correct"] += 1
    per_template_acc = {t: v["correct"] / v["total"] for t, v in per_template.items()}
    return {"overall_accuracy": overall_acc, "per_template_accuracy": per_template_acc, "qualitative": qualitative}


def main():
    phase3_dir = Path(__file__).resolve().parents[1]
    language_dir = phase3_dir.parent
    config_path = phase3_dir / "configs" / "model_h_finetune_config.json"
    results_dir = Path(__file__).resolve().parent / "results"
    plots_dir = Path(__file__).resolve().parent / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"=== Evaluating Model H (Telugu) reasoning finetune | device={device} ===")

    tokenizer_path = (language_dir / config["tokenizer_path"]).resolve()
    tokenizer = SentencePieceTokenizer(str(tokenizer_path))

    pretrained_ckpt = (language_dir / config["pretrained_checkpoint"]).resolve()
    finetuned_ckpt = phase3_dir / "finetune" / "checkpoints" / "checkpoint_final.pt"

    test_path = phase3_dir / "reasoning_data" / "test.jsonl"
    test_records = [json.loads(line) for line in open(test_path, "r", encoding="utf-8") if line.strip()]
    print(f"Test examples: {len(test_records)}")

    pretrained_model = load_model(config, pretrained_ckpt, device)
    pretrained_results = evaluate_model(pretrained_model, tokenizer, test_records, device)
    print(f"\n[Pretrained] Overall exact-match accuracy: {pretrained_results['overall_accuracy']:.4f}")
    for t, acc in pretrained_results["per_template_accuracy"].items():
        print(f"  {t}: {acc:.4f}")
    del pretrained_model

    finetuned_model = load_model(config, finetuned_ckpt, device)
    finetuned_results = evaluate_model(finetuned_model, tokenizer, test_records, device)
    print(f"\n[Finetuned] Overall exact-match accuracy: {finetuned_results['overall_accuracy']:.4f}")
    for t, acc in finetuned_results["per_template_accuracy"].items():
        print(f"  {t}: {acc:.4f}")

    results = {
        "language": "telugu",
        "pretrained_checkpoint": str(pretrained_ckpt),
        "finetuned_checkpoint": str(finetuned_ckpt),
        "pretrained": {
            "overall_accuracy": pretrained_results["overall_accuracy"],
            "per_template_accuracy": pretrained_results["per_template_accuracy"],
        },
        "finetuned": {
            "overall_accuracy": finetuned_results["overall_accuracy"],
            "per_template_accuracy": finetuned_results["per_template_accuracy"],
        },
        "qualitative_examples_pretrained": pretrained_results["qualitative"][:15],
        "qualitative_examples_finetuned": finetuned_results["qualitative"][:15],
    }

    results_path = results_dir / "eval_reasoning_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[eval_reasoning_telugu] Full results saved -> {results_path}")

    templates = sorted(set(list(pretrained_results["per_template_accuracy"].keys()) + list(finetuned_results["per_template_accuracy"].keys())))
    pre_accs = [pretrained_results["per_template_accuracy"].get(t, 0.0) for t in templates]
    fin_accs = [finetuned_results["per_template_accuracy"].get(t, 0.0) for t in templates]

    x = range(len(templates))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - width / 2 for i in x], pre_accs, width, label="Pretrained", color="#888888")
    ax.bar([i + width / 2 for i in x], fin_accs, width, label="Finetuned", color="#2c7fb8")
    ax.set_xticks(list(x))
    ax.set_xticklabels(templates, rotation=15)
    ax.set_ylabel("Exact-match accuracy")
    ax.set_xlabel("Reasoning template")
    ax.set_title("Telugu (Model H): Pretrained vs. Finetuned Reasoning Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()
    plot_path = plots_dir / "telugu_reasoning_accuracy_comparison.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[eval_reasoning_telugu] Plot saved -> {plot_path}")


if __name__ == "__main__":
    main()
