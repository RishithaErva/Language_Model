"""
TELUGU/PHASE3/attention/attention_reasoning_telugu.py

Compares Model H's (Telugu) attention pretrained vs. finetuned on reasoning prompts,
early layer vs. late layer, via the Phase 2 AttentionAnalyzer toolkit.
"""

import sys
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

import torch
from src.model import GPTLanguageModel
from src.dataset import SentencePieceTokenizer
from src.attention_analyzer import AttentionAnalyzer


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


def main():
    phase3_dir = Path(__file__).resolve().parents[1]
    language_dir = phase3_dir.parent
    config_path = phase3_dir / "configs" / "model_h_finetune_config.json"
    plots_dir = Path(__file__).resolve().parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"=== Post-Finetune Attention Analysis: Model H (Telugu) | device={device} ===")

    tokenizer = SentencePieceTokenizer(str((language_dir / config["tokenizer_path"]).resolve()))
    pretrained_ckpt = (language_dir / config["pretrained_checkpoint"]).resolve()
    finetuned_ckpt = phase3_dir / "finetune" / "checkpoints" / "checkpoint_final.pt"

    pretrained_model = load_model(config, pretrained_ckpt, device)
    finetuned_model = load_model(config, finetuned_ckpt, device)

    early_layer, late_layer = 0, config.get("n_layer", 6) - 1

    sample_texts = [
        "రాము వయస్సు 40 సంవత్సరాలు. సీత వయస్సు 25 సంవత్సరాలు. వీరిలో ఎవరు వయసులో పెద్దవారు?",
        "పుస్తకం కంటే సైకిల్ ధరలో ఎక్కువ. సైకిల్ కంటే కారు ధరలో ఎక్కువ. పుస్తకం మరియు కారు లో ఎవరికి ఎక్కువ ధర ఉంది?",
    ]

    summary = {"language": "telugu", "early_layer": early_layer, "late_layer": late_layer, "samples": []}

    # BUG FIX: create one AttentionAnalyzer per model and reuse it — previously a new
    # instance (and a redundant .to(device) call) was created for every method call.
    pre_analyzer = AttentionAnalyzer(pretrained_model, tokenizer, device=device)
    fin_analyzer = AttentionAnalyzer(finetuned_model, tokenizer, device=device)

    for i, text in enumerate(sample_texts):
        pre_analysis = pre_analyzer.extract_attentions(text)
        fin_analysis = fin_analyzer.extract_attentions(text)

        pre_entropy = pre_analyzer.compute_attention_entropy(pre_analysis["attentions"])
        fin_entropy = fin_analyzer.compute_attention_entropy(fin_analysis["attentions"])
        pre_distance = pre_analyzer.compute_mean_attention_distance(pre_analysis["attentions"])
        fin_distance = fin_analyzer.compute_mean_attention_distance(fin_analysis["attentions"])

        pre_analyzer.plot_heatmap(pre_analysis, layer_idx=early_layer, head_idx=0,
                               save_path=plots_dir / f"telugu_sample{i+1}_pretrained_layer1_head1.png",
                               title=f"Telugu Sample {i+1} — Pretrained — Layer 1 Head 1")
        pre_analyzer.plot_heatmap(pre_analysis, layer_idx=late_layer, head_idx=0,
                               save_path=plots_dir / f"telugu_sample{i+1}_pretrained_layer{late_layer+1}_head1.png",
                               title=f"Telugu Sample {i+1} — Pretrained — Layer {late_layer+1} Head 1")

        fin_analyzer.plot_heatmap(fin_analysis, layer_idx=early_layer, head_idx=0,
                                   save_path=plots_dir / f"telugu_sample{i+1}_finetuned_layer1_head1.png",
                                   title=f"Telugu Sample {i+1} — Finetuned — Layer 1 Head 1")
        fin_analyzer.plot_heatmap(fin_analysis, layer_idx=late_layer, head_idx=0,
                                   save_path=plots_dir / f"telugu_sample{i+1}_finetuned_layer{late_layer+1}_head1.png",
                                   title=f"Telugu Sample {i+1} — Finetuned — Layer {late_layer+1} Head 1")

        summary["samples"].append({
            "text": text,
            "tokens": pre_analysis["tokens"],
            "pretrained_entropy_early_mean": float(pre_entropy[early_layer].mean()),
            "pretrained_entropy_late_mean": float(pre_entropy[late_layer].mean()),
            "finetuned_entropy_early_mean": float(fin_entropy[early_layer].mean()),
            "finetuned_entropy_late_mean": float(fin_entropy[late_layer].mean()),
            "pretrained_distance_early_mean": float(pre_distance[early_layer].mean()),
            "pretrained_distance_late_mean": float(pre_distance[late_layer].mean()),
            "finetuned_distance_early_mean": float(fin_distance[early_layer].mean()),
            "finetuned_distance_late_mean": float(fin_distance[late_layer].mean()),
        })

        print(f"\nSample {i+1}: {text}")
        print(f"  Entropy  (early) pretrained={summary['samples'][-1]['pretrained_entropy_early_mean']:.4f} finetuned={summary['samples'][-1]['finetuned_entropy_early_mean']:.4f}")
        print(f"  Entropy  (late)  pretrained={summary['samples'][-1]['pretrained_entropy_late_mean']:.4f} finetuned={summary['samples'][-1]['finetuned_entropy_late_mean']:.4f}")
        print(f"  Distance (early) pretrained={summary['samples'][-1]['pretrained_distance_early_mean']:.4f} finetuned={summary['samples'][-1]['finetuned_distance_early_mean']:.4f}")
        print(f"  Distance (late)  pretrained={summary['samples'][-1]['pretrained_distance_late_mean']:.4f} finetuned={summary['samples'][-1]['finetuned_distance_late_mean']:.4f}")

    results_path = Path(__file__).resolve().parent / "attention_reasoning_summary.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[attention_reasoning_telugu] Summary saved -> {results_path}")


if __name__ == "__main__":
    main()
