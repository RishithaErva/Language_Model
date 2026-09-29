"""
ablation/eval_ablation.py

Evaluates the no-positional-embeddings ablation of Model L (Nepali) using the same
evaluation suite as NEPALI/PHASE2/eval_nepali.py (PPL/BPB on val + test, greedy/
temperature generation, diversity diagnostics, BLEU/chrF++/ROUGE-L, attention
heatmaps/entropy/mean-attention-distance) -- and, whenever the standard
(non-ablated) Model L checkpoint can be found, evaluates that too and writes a
side-by-side comparison table, as the assignment's Bonus section requires.
"""

import os
import sys
import json
import argparse
import importlib.util
from pathlib import Path

ABLATION_DIR = Path(__file__).resolve().parent
# dataset.py/metrics.py/attention_analyzer.py are unmodified, so they are not
# duplicated here -- imported from the main project's src/ package (this folder
# lives inside the full repo, as a sibling of NEPALI/, TELUGU/, and src/).
ROOT_DIR = ABLATION_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

import torch
from src.dataset import SentencePieceTokenizer, create_dataloader
from src.metrics import (
    compute_perplexity_and_bpb,
    generate_text,
    compute_diversity_metrics,
    compute_generation_metrics,
    extract_test_prefix_reference_pairs
)
from src.attention_analyzer import AttentionAnalyzer

# model.py is not shared -- ablation/src/model.py carries the use_positional_embeddings
# flag the main src/model.py doesn't have, so it is loaded directly from its own file
# instead of through the "src" package above (which would resolve to the main model.py).
_model_spec = importlib.util.spec_from_file_location("ablation_model", ABLATION_DIR / "src" / "model.py")
_ablation_model = importlib.util.module_from_spec(_model_spec)
_model_spec.loader.exec_module(_ablation_model)
GPTLanguageModel = _ablation_model.GPTLanguageModel

VARIANT_CONFIGS = {
    "no_pos": "model_l_no_pos_config.json",
}

# Same held-out fallback prompts as NEPALI/PHASE2/eval_nepali.py, used only if
# test.jsonl isn't available to extract real held-out prefix/reference pairs from.
DEFAULT_PROMPTS = [
    "नेपाल एक बहुसांस्कृतिक र",
    "विज्ञान तथा प्रविधिको विकासले",
    "नेपाली भाषा र साहित्यको"
]
DEFAULT_REFERENCES = [
    "नेपाल एक बहुसांस्कृतिक र प्राकृतिक सुन्दरताले भरिपूर्ण देश हो।",
    "विज्ञान तथा प्रविधिको विकासले आधुनिक समाजमा ठूलो परिवर्तन ल्याएको छ।",
    "नेपाली भाषा र साहित्यको इतिहास ज्यादै प्राचीन र धनी छ।"
]
ATTENTION_SAMPLE_TEXT = "नेपाल एक सुन्दर देश हो।"


def find_data_root() -> Path:
    """Same lookup order as train_ablation.py's find_data_root()."""
    bundled = ABLATION_DIR / "NEPALI" / "PHASE1" / "outputs"
    if (bundled / "data_splits" / "val.jsonl").exists():
        return bundled

    local = ABLATION_DIR.parent / "NEPALI" / "PHASE1" / "outputs"
    if (local / "data_splits" / "val.jsonl").exists():
        return local

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for root, dirs, _ in os.walk(kaggle_input):
            if "NEPALI" in dirs:
                candidate = Path(root) / "NEPALI" / "PHASE1" / "outputs"
                if (candidate / "data_splits" / "val.jsonl").exists():
                    return candidate

    raise FileNotFoundError(
        "Could not find NEPALI/PHASE1/outputs/data_splits/ locally or under /kaggle/input."
    )


def find_baseline_checkpoint():
    """
    Locates the standard (non-ablated) Model L checkpoint to compare the ablation
    against: a copy under ablation/baseline_checkpoint/checkpoint_final.pt, or the
    graded checkpoint at ../NEPALI/PHASE2/checkpoints/checkpoint_final.pt. Returns
    None if neither exists -- the ablated model is still evaluated on its own, just
    without a side-by-side comparison table.
    """
    candidates = [
        ABLATION_DIR / "baseline_checkpoint" / "checkpoint_final.pt",
        ABLATION_DIR.parent / "NEPALI" / "PHASE2" / "checkpoints" / "checkpoint_final.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def build_model(config: dict) -> GPTLanguageModel:
    return GPTLanguageModel(
        vocab_size=config.get("vocab_size", 12000),
        d_model=config.get("d_model", 512),
        n_layer=config.get("n_layer", 6),
        n_head=config.get("n_head", 8),
        d_ff=config.get("d_ff", 2048),
        max_seq_len=config.get("max_seq_len", 512),
        dropout=config.get("dropout", 0.1),
        tie_weights=config.get("tie_weights", True),
        use_positional_embeddings=config.get("use_positional_embeddings", True),
        norm_type=config.get("norm_type", "layernorm")
    )


def evaluate_model(tag: str, model, tokenizer, device: str, data_root: Path, cache_dir: Path,
                    plots_dir: Path, analyzer_tag: str) -> dict:
    """Runs the full Phase 2 evaluation suite on one model and returns a results dict."""
    results = {"tag": tag}
    model.to(device)
    model.eval()

    val_jsonl = data_root / "data_splits" / "val.jsonl"
    test_jsonl = data_root / "data_splits" / "test.jsonl"

    # 1. PPL & BPB -- validation split
    if val_jsonl.exists():
        val_loader = create_dataloader(val_jsonl, tokenizer, batch_size=32, shuffle=False, cache_dir=cache_dir)
        metrics = compute_perplexity_and_bpb(model, val_loader, tokenizer, device=device)
        results["intrinsic_metrics"] = metrics
        print(f"\n--- [{tag}] Intrinsic LM Metrics (Validation) ---", flush=True)
        print(f"Val Loss:   {metrics['val_loss']:.4f}", flush=True)
        print(f"Perplexity: {metrics['perplexity']:.4f}", flush=True)
        print(f"BPB:        {metrics['bpb']:.4f}", flush=True)

    # 1b. PPL & BPB -- held-out test split
    if test_jsonl.exists():
        test_loader = create_dataloader(test_jsonl, tokenizer, batch_size=32, shuffle=False, cache_dir=cache_dir)
        test_metrics = compute_perplexity_and_bpb(model, test_loader, tokenizer, device=device)
        results["test_intrinsic_metrics"] = test_metrics
        print(f"\n--- [{tag}] Intrinsic LM Metrics (Test) ---", flush=True)
        print(f"Test Loss:  {test_metrics['val_loss']:.4f}", flush=True)
        print(f"Test PPL:   {test_metrics['perplexity']:.4f}", flush=True)
        print(f"Test BPB:   {test_metrics['bpb']:.4f}", flush=True)

    # 2. Text Generation Sampling & Held-Out Reference Extraction
    test_pairs = (
        extract_test_prefix_reference_pairs(test_jsonl, tokenizer, num_samples=50, prefix_len=15, continuation_len=30)
        if test_jsonl.exists() else []
    )
    if test_pairs:
        prompts = [p["prompt"] for p in test_pairs]
        references = [p["reference"] for p in test_pairs]
    else:
        prompts, references = DEFAULT_PROMPTS, DEFAULT_REFERENCES

    generated_samples, temperature_sweep = [], []
    for prompt in prompts[:3]:
        print(f"\n[{tag}] Prompt: {prompt}", flush=True)
        for temp in [0.5, 1.0, 1.5]:
            gen = generate_text(model, tokenizer, prompt, max_new_tokens=15, temperature=temp, device=device)
            print(f"  [Temp {temp:.1f}]: {gen}", flush=True)
            generated_samples.append(gen)
            temperature_sweep.append({"prompt": prompt, "temperature": temp, "generation": gen})
    results["temperature_sweep_samples"] = temperature_sweep

    # 3. Diversity Diagnostics
    div = compute_diversity_metrics(generated_samples)
    results["diversity_metrics"] = div
    print(f"\n--- [{tag}] Diversity Diagnostics ---", flush=True)
    print(f"Distinct-1:        {div['distinct_1']:.4f}", flush=True)
    print(f"Distinct-2:        {div['distinct_2']:.4f}", flush=True)
    print(f"4-gram Repetition: {div['repetition_rate_4gram']:.4f}", flush=True)

    # 4. Generation Quality Metrics (BLEU-4, chrF++, ROUGE-L)
    greedy_gens = [generate_text(model, tokenizer, p, max_new_tokens=30, temperature=0.0, device=device) for p in prompts]
    gen_metrics = compute_generation_metrics(references, greedy_gens)
    results["generation_quality_metrics"] = gen_metrics
    results["greedy_generations"] = [
        {"prompt": p, "reference": r, "generation": g}
        for p, r, g in zip(prompts, references, greedy_gens)
    ]
    print(f"\n--- [{tag}] Generation Quality Metrics ---", flush=True)
    print(f"BLEU-4:  {gen_metrics['bleu_4']:.2f}", flush=True)
    print(f"chrF++:  {gen_metrics['chrf_plus_plus']:.2f}", flush=True)
    print(f"ROUGE-L: {gen_metrics['rouge_l']:.2f}", flush=True)

    # 5. Attention Analysis (multiple heads, early + late layer)
    print(f"\n--- [{tag}] Attention Analysis ---", flush=True)
    analyzer = AttentionAnalyzer(model, tokenizer, device=device)
    analysis = analyzer.extract_attentions(ATTENTION_SAMPLE_TEXT)

    entropy = analyzer.compute_attention_entropy(analysis["attentions"])
    distance = analyzer.compute_mean_attention_distance(analysis["attentions"])

    print(f"Attention Entropy (Layer 1, Head 1): {entropy[0, 0]:.4f}", flush=True)
    print(f"Attention Distance (Layer 1, Head 1): {distance[0, 0]:.4f}", flush=True)
    results["attention_sample_text"] = ATTENTION_SAMPLE_TEXT
    results["attention_entropy_layer1_head1"] = float(entropy[0, 0])
    results["attention_distance_layer1_head1"] = float(distance[0, 0])
    results["attention_entropy_all_layers_heads"] = entropy.tolist()
    results["attention_distance_all_layers_heads"] = distance.tolist()

    n_layer = len(analysis["attentions"])
    last_layer = n_layer - 1

    analyzer.plot_heatmap(analysis, layer_idx=0, head_idx=0,
                           save_path=plots_dir / f"{analyzer_tag}_attn_layer1_head1.png",
                           title=f"Nepali ({tag}) Layer 1 Head 1 Attention")
    analyzer.plot_heatmap(analysis, layer_idx=last_layer, head_idx=0,
                           save_path=plots_dir / f"{analyzer_tag}_attn_layer{n_layer}_head1.png",
                           title=f"Nepali ({tag}) Layer {n_layer} Head 1 Attention")
    analyzer.plot_multi_head_grid(analysis, layer_idx=0, head_indices=[0, 1, 2, 3],
                                   save_path=plots_dir / f"{analyzer_tag}_attn_layer1_multihead_grid.png",
                                   title=f"Nepali ({tag}) Early Layer Multi-Head Attention Grid")
    analyzer.plot_multi_head_grid(analysis, layer_idx=last_layer, head_indices=[0, 1, 2, 3],
                                   save_path=plots_dir / f"{analyzer_tag}_attn_layer{n_layer}_multihead_grid.png",
                                   title=f"Nepali ({tag}) Late Layer Multi-Head Attention Grid")

    return results


def build_comparison_table(baseline_results: dict, ablated_results: dict, variant: str) -> str:
    """Markdown table: standard Model L vs. the ablated variant, metric by metric."""
    def g(d, *keys):
        cur = d
        for k in keys:
            if cur is None:
                return None
            cur = cur.get(k)
        return cur

    rows = [
        ("Val Loss", g(baseline_results, "intrinsic_metrics", "val_loss"), g(ablated_results, "intrinsic_metrics", "val_loss")),
        ("Val Perplexity", g(baseline_results, "intrinsic_metrics", "perplexity"), g(ablated_results, "intrinsic_metrics", "perplexity")),
        ("Val BPB", g(baseline_results, "intrinsic_metrics", "bpb"), g(ablated_results, "intrinsic_metrics", "bpb")),
        ("Test Loss", g(baseline_results, "test_intrinsic_metrics", "val_loss"), g(ablated_results, "test_intrinsic_metrics", "val_loss")),
        ("Test Perplexity", g(baseline_results, "test_intrinsic_metrics", "perplexity"), g(ablated_results, "test_intrinsic_metrics", "perplexity")),
        ("Test BPB", g(baseline_results, "test_intrinsic_metrics", "bpb"), g(ablated_results, "test_intrinsic_metrics", "bpb")),
        ("BLEU-4", g(baseline_results, "generation_quality_metrics", "bleu_4"), g(ablated_results, "generation_quality_metrics", "bleu_4")),
        ("chrF++", g(baseline_results, "generation_quality_metrics", "chrf_plus_plus"), g(ablated_results, "generation_quality_metrics", "chrf_plus_plus")),
        ("ROUGE-L", g(baseline_results, "generation_quality_metrics", "rouge_l"), g(ablated_results, "generation_quality_metrics", "rouge_l")),
        ("Distinct-1", g(baseline_results, "diversity_metrics", "distinct_1"), g(ablated_results, "diversity_metrics", "distinct_1")),
        ("Distinct-2", g(baseline_results, "diversity_metrics", "distinct_2"), g(ablated_results, "diversity_metrics", "distinct_2")),
        ("4-gram Repetition", g(baseline_results, "diversity_metrics", "repetition_rate_4gram"), g(ablated_results, "diversity_metrics", "repetition_rate_4gram")),
        ("Attention Entropy (L1H1)", g(baseline_results, "attention_entropy_layer1_head1"), g(ablated_results, "attention_entropy_layer1_head1")),
        ("Attention Distance (L1H1)", g(baseline_results, "attention_distance_layer1_head1"), g(ablated_results, "attention_distance_layer1_head1")),
    ]

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else ("--" if v is None else str(v))

    lines = [
        f"# Ablation Comparison: Standard Model L vs. `{variant}`",
        "",
        f"| Metric | Standard Model L | Ablated (`{variant}`) |",
        "|---|---|---|",
    ]
    for name, base_val, abl_val in rows:
        lines.append(f"| {name} | {fmt(base_val)} | {fmt(abl_val)} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the no-positional-embeddings ablation against the standard model.")
    parser.add_argument("--variant", choices=list(VARIANT_CONFIGS.keys()), default="no_pos")
    args = parser.parse_args()
    variant = args.variant

    config_path = ABLATION_DIR / "configs" / VARIANT_CONFIGS[variant]
    with open(config_path, "r", encoding="utf-8") as f:
        ablated_config = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Evaluating Ablation '{variant}' (+ standard Model L for comparison) on {device} ===", flush=True)

    data_root = find_data_root()
    # tokenizer/ is not duplicated in ablation/ -- it lives alongside the data splits,
    # at NEPALI/PHASE1/outputs/tokenizer/ (same outputs/ dir find_data_root() found).
    tokenizer = SentencePieceTokenizer(str(data_root / "tokenizer" / "nepali_bpe.model"))

    results_dir = ABLATION_DIR / "outputs" / variant / "results"
    plots_dir = ABLATION_DIR / "outputs" / variant / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = ABLATION_DIR / "outputs" / "_shared_cache"

    # --- Ablated model ---
    ablated_ckpt_dir = ABLATION_DIR / "outputs" / variant / "checkpoints"
    ablated_ckpt = ablated_ckpt_dir / "checkpoint_final.pt"
    if not ablated_ckpt.exists():
        ablated_ckpt = ablated_ckpt_dir / "checkpoint_latest.pt"
        print(f"[Info] No final checkpoint yet for '{variant}' -- evaluating latest checkpoint instead: {ablated_ckpt}", flush=True)
    if not ablated_ckpt.exists():
        raise FileNotFoundError(
            f"No checkpoint found under {ablated_ckpt_dir}. Run train_ablation.py --variant {variant} first."
        )

    ablated_model = build_model(ablated_config)
    state = torch.load(ablated_ckpt, map_location=device)
    ablated_model.load_state_dict(state["model_state_dict"])
    print(f"Loaded ablated checkpoint: {ablated_ckpt}", flush=True)

    ablated_results = evaluate_model(
        f"ablated_{variant}", ablated_model, tokenizer, device, data_root, cache_dir, plots_dir,
        analyzer_tag=f"nepali_{variant}"
    )
    ablated_results["checkpoint_used"] = str(ablated_ckpt)
    ablated_results["config"] = ablated_config

    results_path = results_dir / f"eval_results_{variant}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(ablated_results, f, indent=2, ensure_ascii=False)
    print(f"\n[eval_ablation] Ablated-model results saved -> {results_path}", flush=True)

    # --- Standard (baseline) model, for the required side-by-side comparison ---
    baseline_ckpt = find_baseline_checkpoint()
    if baseline_ckpt is None:
        print(
            "\n[Warning] No baseline (standard Model L) checkpoint found -- skipping comparison.",
            flush=True
        )
        return

    baseline_config = dict(ablated_config)
    baseline_config["use_positional_embeddings"] = True
    baseline_config["norm_type"] = "layernorm"
    baseline_model = build_model(baseline_config)
    state = torch.load(baseline_ckpt, map_location=device)
    baseline_model.load_state_dict(state["model_state_dict"])
    print(f"Loaded baseline checkpoint: {baseline_ckpt}", flush=True)

    baseline_results = evaluate_model(
        "standard", baseline_model, tokenizer, device, data_root, cache_dir, plots_dir,
        analyzer_tag="nepali_standard"
    )
    baseline_results["checkpoint_used"] = str(baseline_ckpt)

    baseline_path = results_dir / "eval_results_standard.json"
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(baseline_results, f, indent=2, ensure_ascii=False)
    print(f"[eval_ablation] Standard-model results saved -> {baseline_path}", flush=True)

    comparison_md = build_comparison_table(baseline_results, ablated_results, variant)
    comparison_path = results_dir / f"comparison_{variant}_vs_standard.md"
    with open(comparison_path, "w", encoding="utf-8") as f:
        f.write(comparison_md)
    print(f"[eval_ablation] Comparison table saved -> {comparison_path}", flush=True)
    print("\n" + comparison_md, flush=True)


if __name__ == "__main__":
    main()
