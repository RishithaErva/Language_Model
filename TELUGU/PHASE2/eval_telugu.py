"""
TELUGU/eval_telugu.py

Evaluation script for Model H (Telugu Transformer LM):
- Computes Validation Loss, Perplexity (PPL), and Bits-Per-Byte (BPB).
- Generates text continuations for held-out prefixes at T=0.5, 1.0, 1.5.
- Computes Diversity Metrics (Distinct-1, Distinct-2, 4-gram Repetition Rate).
- Computes Generation Quality Metrics (BLEU-4, chrF++, ROUGE-L) against held-out references.
- Extracts Attention Maps, Attention Entropy, and Mean Attention Distance.
- Plots single-head and multi-head attention heatmaps for early vs late layers.
"""

import sys
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

import torch
from src.model import GPTLanguageModel
from src.dataset import SentencePieceTokenizer, create_dataloader
from src.metrics import (
    compute_perplexity_and_bpb,
    generate_text,
    compute_diversity_metrics,
    compute_generation_metrics,
    extract_test_prefix_reference_pairs
)
from src.attention_analyzer import AttentionAnalyzer


def main():
    telugu_dir = Path(__file__).resolve().parent
    config_path = telugu_dir / "configs" / "model_h_config.json"
    tokenizer_path = telugu_dir / "tokenizer" / "telugu_bpe.model"
    # NOT committed to git -- fetch train/val/test.jsonl from Drive and place them here first.
    data_dir = telugu_dir / "data"
    # NOT committed to git -- fetch checkpoint_final.pt from Drive (see README) and place it here.
    checkpoint_dir = telugu_dir / "checkpoints"
    checkpoint_path = checkpoint_dir / "checkpoint_final.pt"
    if not checkpoint_path.exists():
        # Training may not have reached max_steps yet (e.g. a Kaggle session cut off mid-run) --
        # fall back to the latest saved step rather than silently evaluating an untrained model.
        checkpoint_path = checkpoint_dir / "checkpoint_latest.pt"
        print(f"[Info] No final checkpoint yet -- evaluating latest checkpoint instead: {checkpoint_path}", flush=True)
    # Tokenized-id cache must be writable even if data_dir is a read-only mount (e.g. Kaggle input)
    cache_dir = telugu_dir / "cache"

    # results/ (JSON) and plots/ (PNGs) are both committed to git -- everything eval writes
    # ends up in one of these two, matching TELUGU/results/ and TELUGU/plots/ already in the repo.
    results_dir = telugu_dir / "results"
    plots_dir = telugu_dir / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Evaluating Model H (Telugu) on {device} ===", flush=True)

    tokenizer = SentencePieceTokenizer(str(tokenizer_path))
    model = GPTLanguageModel(
        vocab_size=config.get("vocab_size", 12000),
        d_model=config.get("d_model", 512),
        n_layer=config.get("n_layer", 6),
        n_head=config.get("n_head", 8),
        d_ff=config.get("d_ff", 2048),
        max_seq_len=config.get("max_seq_len", 512),
        dropout=config.get("dropout", 0.1),
        tie_weights=config.get("tie_weights", True)
    )

    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}", flush=True)
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
    else:
        print(f"[Warning] Checkpoint {checkpoint_path} not found. Running evaluation on un-trained model for verification.", flush=True)

    model.to(device)

    # Accumulates every number/sample this script produces, so results survive past the
    # console/notebook output -- written to report_dir / "eval_results.json" at the end.
    results = {"language": "telugu", "checkpoint_used": str(checkpoint_path)}

    # 1. PPL & BPB Evaluation -- validation split
    val_jsonl = data_dir / "val.jsonl"
    if val_jsonl.exists():
        val_loader = create_dataloader(val_jsonl, tokenizer, batch_size=32, shuffle=False, cache_dir=cache_dir)
        metrics = compute_perplexity_and_bpb(model, val_loader, tokenizer, device=device)
        results["intrinsic_metrics"] = metrics
        print("\n--- Intrinsic LM Metrics (Validation) ---", flush=True)
        print(f"Val Loss:   {metrics['val_loss']:.4f}", flush=True)
        print(f"Perplexity: {metrics['perplexity']:.4f}", flush=True)
        print(f"BPB:        {metrics['bpb']:.4f}", flush=True)

    # 1b. PPL & BPB Evaluation -- held-out test split (reported separately from validation,
    # since validation loss was also used during training for scheduling/monitoring)
    test_jsonl_loss = data_dir / "test.jsonl"
    if test_jsonl_loss.exists():
        test_loader = create_dataloader(test_jsonl_loss, tokenizer, batch_size=32, shuffle=False, cache_dir=cache_dir)
        test_metrics = compute_perplexity_and_bpb(model, test_loader, tokenizer, device=device)
        results["test_intrinsic_metrics"] = test_metrics
        print("\n--- Intrinsic LM Metrics (Test) ---", flush=True)
        print(f"Test Loss:  {test_metrics['val_loss']:.4f}", flush=True)
        print(f"Test PPL:   {test_metrics['perplexity']:.4f}", flush=True)
        print(f"Test BPB:   {test_metrics['bpb']:.4f}", flush=True)

    # 2. Text Generation Sampling & Held-Out Reference Extraction
    test_jsonl = data_dir / "test.jsonl"
    test_pairs = extract_test_prefix_reference_pairs(test_jsonl, tokenizer, num_samples=50, prefix_len=15, continuation_len=30)

    if test_pairs:
        print(f"\n--- Extracted {len(test_pairs)} Prefix/Reference Pairs from {test_jsonl.name} ---", flush=True)
        prompts = [p["prompt"] for p in test_pairs]
        references = [p["reference"] for p in test_pairs]
    else:
        print("\n--- Using Default Held-Out Validation Prompts ---", flush=True)
        prompts = [
            "భారతదేశం విభిన్న సాంస్కృతిక",
            "విజ్ఞాన శాస్త్రము మరియు సాంకేతికత",
            "తెలుగు భాషా చరిత్ర మరియు"
        ]
        references = [
            "భారతదేశం విభిన్న సాంస్కృతిక విశేషాలతో నిండిన దేశం.",
            "విజ్ఞాన శాస్త్రము మరియు సాంకేతికత రంగాలు వేగంగా అభివృద్ధి చెందుతున్నాయి.",
            "తెలుగు భాషా చరిత్ర మరియు సంస్కృతి అతి పురాతనమైనవి."
        ]

    generated_samples = []
    temperature_sweep = []
    for prompt in prompts[:3]:
        print(f"\nPrompt: {prompt}", flush=True)
        for temp in [0.5, 1.0, 1.5]:
            gen = generate_text(model, tokenizer, prompt, max_new_tokens=15, temperature=temp, device=device)
            print(f"  [Temp {temp:.1f}]: {gen}", flush=True)
            generated_samples.append(gen)
            temperature_sweep.append({"prompt": prompt, "temperature": temp, "generation": gen})
    results["temperature_sweep_samples"] = temperature_sweep

    # 3. Diversity Diagnostics
    div = compute_diversity_metrics(generated_samples)
    results["diversity_metrics"] = div
    print("\n--- Diversity Diagnostics ---", flush=True)
    print(f"Distinct-1:           {div['distinct_1']:.4f}", flush=True)
    print(f"Distinct-2:           {div['distinct_2']:.4f}", flush=True)
    print(f"4-gram Repetition:    {div['repetition_rate_4gram']:.4f}", flush=True)

    # 4. Generation Quality Metrics (BLEU-4, chrF++, ROUGE-L)
    # max_new_tokens matches continuation_len (30) used above: a shorter hypothesis than its
    # reference triggers BLEU's brevity penalty regardless of content quality, which would
    # deflate the score for a reason that has nothing to do with the model.
    greedy_gens = [generate_text(model, tokenizer, p, max_new_tokens=30, temperature=0.0, device=device) for p in prompts]
    gen_metrics = compute_generation_metrics(references, greedy_gens)
    results["generation_quality_metrics"] = gen_metrics
    results["greedy_generations"] = [
        {"prompt": p, "reference": r, "generation": g}
        for p, r, g in zip(prompts, references, greedy_gens)
    ]
    print("\n--- Generation Quality Metrics ---", flush=True)
    print(f"BLEU-4:               {gen_metrics['bleu_4']:.2f}", flush=True)
    print(f"chrF++:               {gen_metrics['chrf_plus_plus']:.2f}", flush=True)
    print(f"ROUGE-L:              {gen_metrics['rouge_l']:.2f}", flush=True)

    # 5. Attention Analysis (Multiple Heads Across Early & Late Layers)
    print("\n--- Attention Analysis ---", flush=True)
    analyzer = AttentionAnalyzer(model, tokenizer, device=device)
    sample_text = "తెలుగు భారతదేశంలోని ప్రసిద్ధ భాష."
    analysis = analyzer.extract_attentions(sample_text)

    entropy = analyzer.compute_attention_entropy(analysis["attentions"])
    distance = analyzer.compute_mean_attention_distance(analysis["attentions"])

    print(f"Attention Entropy (Layer 1, Head 1): {entropy[0, 0]:.4f}", flush=True)
    print(f"Attention Distance (Layer 1, Head 1): {distance[0, 0]:.4f}", flush=True)
    results["attention_sample_text"] = sample_text
    results["attention_entropy_layer1_head1"] = float(entropy[0, 0])
    results["attention_distance_layer1_head1"] = float(distance[0, 0])
    results["attention_entropy_all_layers_heads"] = entropy.tolist()
    results["attention_distance_all_layers_heads"] = distance.tolist()

    # Plot single head heatmaps
    analyzer.plot_heatmap(analysis, layer_idx=0, head_idx=0, save_path=plots_dir / "telugu_attn_layer1_head1.png", title="Telugu Layer 1 Head 1 Attention")
    analyzer.plot_heatmap(analysis, layer_idx=5, head_idx=0, save_path=plots_dir / "telugu_attn_layer6_head1.png", title="Telugu Layer 6 Head 1 Attention")

    # Plot multi-head grid heatmaps for early vs. late layers
    analyzer.plot_multi_head_grid(analysis, layer_idx=0, head_indices=[0, 1, 2, 3], save_path=plots_dir / "telugu_attn_layer1_multihead_grid.png", title="Telugu Early Layer 1 Multi-Head Attention Grid")
    analyzer.plot_multi_head_grid(analysis, layer_idx=5, head_indices=[0, 1, 2, 3], save_path=plots_dir / "telugu_attn_layer6_multihead_grid.png", title="Telugu Late Layer 6 Multi-Head Attention Grid")

    # Persist everything computed above -- console/notebook output alone doesn't survive a
    # lost session, but this file does, and it's what gets synced to Drive alongside the heatmaps.
    results_path = results_dir / "eval_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[eval_telugu] Full results saved -> {results_path}", flush=True)


if __name__ == "__main__":
    main()
