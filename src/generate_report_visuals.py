"""
src/generate_report_visuals.py

Generates publication-quality visual figures and charts for the Phase 2 report:
1. Model architecture diagram.
2. Phase 1 corpus/tokenizer comparison chart.
3. Training/validation loss + perplexity convergence curves (real data, both languages overlaid).
4. Validation perplexity on a log scale (the raw scale hides the shape of the later, more
   important part of the curve since PPL starts in the thousands and ends under 40).
5. Learning rate schedule (warmup + cosine decay) actually used during training.
6. Training throughput (tokens/sec) over the run, from consecutive elapsed_sec deltas.
7. Attention entropy and mean attention distance, as full layer x head heatmaps (not just a
   single Layer-1/Head-1 number) -- computation lives in src/attention_analyzer.py, this module
   only reads the saved matrices and plots them.
8. Generation-quality (BLEU-4/chrF++/ROUGE-L) and diversity (Distinct-1/2, repetition) bar
   charts, per language and Model H vs. Model L side by side once both exist.
9. Intrinsic metrics (PPL, BPB) comparison, Model H vs. Model L.
10. A consolidated markdown results table for the report, from whichever eval_results.json
    files exist.

Every "generate_*" function here only plots; the small "_compute_*"/"_load_*" helpers that
prepare the numbers are kept separate, per the assignment's visualization/computation split.
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

# Set matplotlib style for publication-quality figures
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["figure.titlesize"] = 14

LANGUAGE_COLORS = {"Telugu (Model H)": "#e67e22", "Nepali (Model L)": "#2980b9"}


# ------------------------------------------------------------------ helpers (computation only)

def _load_json(path: Path) -> Optional[dict]:
    """Safely loads a JSON file, returning None (with a message) if missing or unreadable."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[Visuals] Could not read {path}: {e}")
        return None


def _load_histories(history_paths: Dict[str, Path]) -> Dict[str, list]:
    """Loads training_history.json for each language that has one yet."""
    out = {}
    for label, path in history_paths.items():
        history = _load_json(path)
        if history:
            out[label] = history
    return out


def _load_eval_results(eval_paths: Dict[str, Path]) -> Dict[str, dict]:
    """Loads eval_results.json for each language that has one yet."""
    out = {}
    for label, path in eval_paths.items():
        results = _load_json(path)
        if results:
            out[label] = results
    return out


def _tokens_per_sec_series(history: list, tokens_per_step: int) -> tuple:
    """
    Computes a tokens/sec throughput series from consecutive (step, elapsed_sec) log points.
    Returns (step_midpoints, tokens_per_sec) for plotting.
    """
    steps, tok_per_sec = [], []
    for prev, cur in zip(history, history[1:]):
        d_steps = cur["step"] - prev["step"]
        d_time = cur["elapsed_sec"] - prev["elapsed_sec"]
        if d_steps > 0 and d_time > 0:
            steps.append((cur["step"] + prev["step"]) / 2)
            tok_per_sec.append((d_steps * tokens_per_step) / d_time)
    return steps, tok_per_sec


# ------------------------------------------------------------------ existing plots

def generate_architecture_diagram(output_path: Path):
    """
    Generates a clean architectural diagram of the Decoder-Only GPT Transformer.
    The middle six boxes (Pre-LN 1 through the second residual connection) are one
    TransformerBlock; a dashed bracket marks that this block is stacked N=6 times before
    the final LayerNorm + LM head, matching src/model.py's actual nn.ModuleList of 6 blocks
    (the diagram previously omitted this and looked like a single-layer model).
    """
    fig, ax = plt.subplots(figsize=(6, 8.5))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    boxes = [
        ("Input Token IDs", 0.92, "#34495e"),
        ("Token + Positional Embeddings (d=512)", 0.835, "#2980b9"),
        ("Pre-LN 1 LayerNorm", 0.75, "#8e44ad"),
        ("Causal Multi-Head Self-Attention (8 Heads)\n[Scaled Dot-Product + Upper-Triangular Mask]", 0.655, "#27ae60"),
        ("Residual Connection (+)", 0.575, "#e67e22"),
        ("Pre-LN 2 LayerNorm", 0.49, "#8e44ad"),
        ("Position-wise Feed-Forward (d_ff=2048, GELU)", 0.405, "#16a085"),
        ("Residual Connection (+)", 0.325, "#e67e22"),
        ("Final LayerNorm (d=512)", 0.16, "#8e44ad"),
        ("LM Head Projection (Weight-Tied wte)", 0.075, "#c0392b")
    ]

    for title, y, color in boxes:
        ax.text(
            0.5, y, title,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.6", facecolor=color, alpha=0.85, edgecolor="none"),
            color="white", weight="bold", fontsize=10, zorder=3
        )

    # Arrows between every consecutive pair of boxes.
    for i in range(len(boxes) - 1):
        y_from, y_to = boxes[i][1], boxes[i + 1][1]
        ax.annotate("", xy=(0.5, y_to + 0.035), xytext=(0.5, y_from - 0.035),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="#2c3e50"), zorder=2)

    # Dashed bracket around the one repeated TransformerBlock (Pre-LN 1 .. second residual).
    block_top, block_bottom = 0.75 + 0.055, 0.325 - 0.055
    bracket_x = 0.06
    ax.plot([bracket_x, bracket_x], [block_bottom, block_top], ls="--", lw=1.3, color="#7f8c8d")
    ax.plot([bracket_x, bracket_x + 0.02], [block_top, block_top], ls="--", lw=1.3, color="#7f8c8d")
    ax.plot([bracket_x, bracket_x + 0.02], [block_bottom, block_bottom], ls="--", lw=1.3, color="#7f8c8d")
    ax.text(bracket_x - 0.025, (block_top + block_bottom) / 2, "$\\times$ N = 6 layers",
            ha="center", va="center", rotation=90, fontsize=10, weight="bold", color="#7f8c8d")

    ax.set_title("Decoder-Only Transformer Architecture (~25.3M Parameters)", pad=20, weight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Visuals] Architecture diagram saved -> {output_path}")


def generate_corpus_comparison_chart(output_path: Path):
    """Generates bar chart comparing Model H (Telugu) vs Model L (Nepali) corpus token counts and manual splits."""
    fig, ax = plt.subplots(figsize=(8, 5))

    languages = ["Telugu (Model H)", "Nepali (Model L)"]
    manual_tokens = [153.94, 106.47]     # In Millions
    downloaded_tokens = [393.54, 420.14] # In Millions

    x = np.arange(len(languages))
    width = 0.35

    rects1 = ax.bar(x - width/2, manual_tokens, width, label="Manual Collection (OCR/Scrape)", color="#27ae60")
    rects2 = ax.bar(x + width/2, downloaded_tokens, width, label="Downloaded Corpus (Sangraha)", color="#2980b9")

    ax.set_ylabel("Token Count (Millions)", weight="bold")
    ax.set_title("Phase 1 Monolingual Corpus Token Distribution", weight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(languages, weight="bold")
    ax.legend(frameon=True)

    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f"{height:.1f}M\n({height/547.5*100:.1f}%)" if height > 120 else f"{height:.1f}M\n({height/526.6*100:.1f}%)",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f"{height:.1f}M",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Visuals] Corpus comparison chart saved -> {output_path}")


def generate_loss_curves(history_paths: List[Path], output_path: Path):
    """
    Reads actual training_history.json log files produced by Trainer and plots convergence curves.
    Falls back gracefully to a clearly-labeled placeholder if no real logs exist yet.
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    real_data_found = False

    for history_path in history_paths:
        history_path = Path(history_path)
        history = _load_json(history_path)
        if history:
            steps = [entry["step"] for entry in history]
            train_loss = [entry["train_loss"] for entry in history]
            val_loss = [entry["val_loss"] for entry in history]
            ppl = [entry["perplexity"] for entry in history]

            lang_label = "Telugu (Model H)" if "TELUGU" in str(history_path) else "Nepali (Model L)"
            ax1.plot(steps, train_loss, label=f"{lang_label} Train Loss", lw=2)
            ax1.plot(steps, val_loss, label=f"{lang_label} Val Loss", lw=2, linestyle="--")
            ax2.plot(steps, ppl, alpha=0.4, lw=1.5, linestyle="-.")
            real_data_found = True

    if not real_data_found:
        steps = np.linspace(0, 10000, 100)
        train_loss = 9.4 * np.exp(-steps / 2500) + 3.1
        val_loss = 9.4 * np.exp(-steps / 2700) + 3.4
        ppl = np.exp(val_loss)

        ax1.plot(steps, train_loss, label="Train Loss (Placeholder)", color="#e74c3c", lw=2)
        ax1.plot(steps, val_loss, label="Val Loss (Placeholder)", color="#3498db", lw=2, linestyle="--")
        ax2.plot(steps, ppl, color="#9b59b6", alpha=0.3, lw=1.5, linestyle="-.")
        ax1.set_title("Pretraining Convergence & Perplexity [Placeholder - Run Training to Populate]", weight="bold", pad=15)
    else:
        ax1.set_title("Pretraining Loss Convergence & Validation Perplexity", weight="bold", pad=15)

    ax1.set_xlabel("Training Steps", weight="bold")
    ax1.set_ylabel("Cross-Entropy Loss (nats)", weight="bold", color="#2c3e50")
    ax2.set_ylabel("Validation Perplexity (PPL)", weight="bold", color="#8e44ad")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Visuals] Convergence curve saved -> {output_path}")


def generate_single_language_loss_curve(history_path: Path, output_path: Path, language_label: str):
    """
    Same data as generate_loss_curves, but for one language only, on its own figure --
    useful for inspecting or presenting each model's convergence independently rather
    than reading it off a shared, overlaid two-language plot. Skips cleanly if that language's
    training_history.json doesn't exist yet.
    """
    history = _load_json(Path(history_path))
    if not history:
        print(f"[Visuals] Skipping {language_label} loss curve -- no training_history.json found yet.")
        return

    steps = [entry["step"] for entry in history]
    train_loss = [entry["train_loss"] for entry in history]
    val_loss = [entry["val_loss"] for entry in history]
    ppl = [entry["perplexity"] for entry in history]
    color = LANGUAGE_COLORS.get(language_label, "#2c3e50")

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    ax1.plot(steps, train_loss, label="Train Loss", lw=2, color=color)
    ax1.plot(steps, val_loss, label="Val Loss", lw=2, linestyle="--", color=color, alpha=0.6)
    ax2.plot(steps, ppl, label="Val Perplexity", lw=1.5, linestyle="-.", color="#8e44ad", alpha=0.5)

    ax1.set_xlabel("Training Steps", weight="bold")
    ax1.set_ylabel("Cross-Entropy Loss (nats)", weight="bold", color="#2c3e50")
    ax2.set_ylabel("Validation Perplexity (PPL)", weight="bold", color="#8e44ad")
    ax1.set_title(f"{language_label}: Pretraining Loss Convergence", weight="bold", pad=15)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Merge both axes' legend entries into one box, since ax2's line wouldn't otherwise
    # appear in ax1.legend().
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Visuals] {language_label} loss curve saved -> {output_path}")


# ------------------------------------------------------------------ new plots

def generate_ppl_curve(history_paths: Dict[str, Path], output_path: Path):
    """Validation perplexity vs. training step, log-scale y-axis (PPL spans 4 orders of magnitude)."""
    histories = _load_histories(history_paths)
    if not histories:
        print("[Visuals] Skipping PPL curve -- no training_history.json found yet.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, history in histories.items():
        steps = [e["step"] for e in history]
        ppl = [e["perplexity"] for e in history]
        ax.plot(steps, ppl, label=label, lw=2, color=LANGUAGE_COLORS.get(label))

    ax.set_yscale("log")
    ax.set_xlabel("Training Steps", weight="bold")
    ax.set_ylabel("Validation Perplexity (log scale)", weight="bold")
    ax.set_title("Validation Perplexity vs. Training Steps", weight="bold", pad=15)
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend(frameon=True)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Visuals] PPL curve (log scale) saved -> {output_path}")


def generate_lr_schedule(history_paths: Dict[str, Path], output_path: Path):
    """Plots the actual learning-rate schedule (warmup + cosine decay) as logged during training."""
    histories = _load_histories(history_paths)
    if not histories:
        print("[Visuals] Skipping LR schedule -- no training_history.json found yet.")
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    # Both models use identical LR hyperparameters (same max_lr, warmup_steps, max_steps), so
    # the schedule is a pure function of step count and the two curves are mathematically
    # identical -- without distinct line styles, one plotted line fully occludes the other.
    linestyles = ["-", "--"]
    for (label, history), linestyle in zip(histories.items(), linestyles):
        steps = [e["step"] for e in history]
        lr = [e["lr"] for e in history]
        ax.plot(steps, lr, label=label, lw=2.5, linestyle=linestyle, alpha=0.8,
                 color=LANGUAGE_COLORS.get(label))

    ax.set_xlabel("Training Steps", weight="bold")
    ax.set_ylabel("Learning Rate", weight="bold")
    ax.set_title("Learning Rate Schedule (Linear Warmup + Cosine Decay) -- Identical for Both Models",
                 weight="bold", pad=15)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Visuals] LR schedule saved -> {output_path}")


def generate_throughput_chart(history_paths: Dict[str, Path], output_path: Path, tokens_per_step: int = 16384):
    """Plots training throughput (tokens/sec) over the run, from consecutive elapsed_sec deltas."""
    histories = _load_histories(history_paths)
    if not histories:
        print("[Visuals] Skipping throughput chart -- no training_history.json found yet.")
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    any_series = False
    for label, history in histories.items():
        steps, tok_per_sec = _tokens_per_sec_series(history, tokens_per_step)
        if steps:
            ax.plot(steps, tok_per_sec, label=label, lw=1.5, marker="o", markersize=3,
                     color=LANGUAGE_COLORS.get(label), alpha=0.85)
            any_series = True

    if not any_series:
        plt.close()
        print("[Visuals] Skipping throughput chart -- not enough log points to compute deltas.")
        return

    ax.set_xlabel("Training Steps", weight="bold")
    ax.set_ylabel("Throughput (tokens/sec)", weight="bold")
    ax.set_title("Training Throughput Over Time", weight="bold", pad=15)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Visuals] Throughput chart saved -> {output_path}")


def generate_attention_stat_heatmap(eval_paths: Dict[str, Path], metric_key: str, title: str,
                                     cmap: str, output_path: Path):
    """
    Plots a layer x head heatmap of an attention statistic (entropy or mean distance) for each
    language that has eval_results.json with a `{metric_key}` matrix saved. One subplot per
    language side by side.
    """
    results = _load_eval_results(eval_paths)
    available = {label: r[metric_key] for label, r in results.items() if metric_key in r}
    if not available:
        print(f"[Visuals] Skipping {metric_key} heatmap -- no eval_results.json with this field yet.")
        return

    fig, axes = plt.subplots(1, len(available), figsize=(6.5 * len(available), 5.5), squeeze=False)
    axes = axes[0]

    for ax, (label, matrix) in zip(axes, available.items()):
        matrix = np.array(matrix)  # (n_layer, n_head)
        im = ax.imshow(matrix, cmap=cmap, aspect="auto")
        ax.set_xticks(range(matrix.shape[1]))
        ax.set_xticklabels([f"H{i+1}" for i in range(matrix.shape[1])])
        ax.set_yticks(range(matrix.shape[0]))
        ax.set_yticklabels([f"L{i+1}" for i in range(matrix.shape[0])])
        ax.set_xlabel("Head", weight="bold")
        ax.set_ylabel("Layer", weight="bold")
        ax.set_title(label, weight="bold")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        color="white" if matrix[i, j] > matrix.mean() else "black", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title, weight="bold", fontsize=14)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Visuals] {title} saved -> {output_path}")


def generate_generation_quality_bar(eval_paths: Dict[str, Path], output_path: Path):
    """Grouped bar chart of BLEU-4 / chrF++ / ROUGE-L, per language (both if available)."""
    results = _load_eval_results(eval_paths)
    available = {label: r["generation_quality_metrics"] for label, r in results.items()
                 if "generation_quality_metrics" in r}
    if not available:
        print("[Visuals] Skipping generation-quality bar chart -- no eval_results.json yet.")
        return

    metrics = ["bleu_4", "chrf_plus_plus", "rouge_l"]
    metric_labels = ["BLEU-4", "chrF++", "ROUGE-L"]
    x = np.arange(len(metrics))
    width = 0.35 if len(available) > 1 else 0.5

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (label, m) in enumerate(available.items()):
        offset = (i - (len(available) - 1) / 2) * width
        values = [m.get(k, 0) for k in metrics]
        rects = ax.bar(x + offset, values, width, label=label, color=LANGUAGE_COLORS.get(label))
        ax.bar_label(rects, fmt="%.2f", fontsize=8, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, weight="bold")
    ax.set_ylabel("Score", weight="bold")
    ax.set_title("Generation Quality: BLEU-4 / chrF++ / ROUGE-L", weight="bold", pad=15)
    ax.legend(frameon=True)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Visuals] Generation quality bar chart saved -> {output_path}")


def generate_diversity_bar(eval_paths: Dict[str, Path], output_path: Path):
    """Grouped bar chart of Distinct-1 / Distinct-2 / 4-gram repetition rate, per language."""
    results = _load_eval_results(eval_paths)
    available = {label: r["diversity_metrics"] for label, r in results.items()
                 if "diversity_metrics" in r}
    if not available:
        print("[Visuals] Skipping diversity bar chart -- no eval_results.json yet.")
        return

    metrics = ["distinct_1", "distinct_2", "repetition_rate_4gram"]
    metric_labels = ["Distinct-1", "Distinct-2", "4-gram Repetition"]
    x = np.arange(len(metrics))
    width = 0.35 if len(available) > 1 else 0.5

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (label, m) in enumerate(available.items()):
        offset = (i - (len(available) - 1) / 2) * width
        values = [m.get(k, 0) for k in metrics]
        rects = ax.bar(x + offset, values, width, label=label, color=LANGUAGE_COLORS.get(label))
        ax.bar_label(rects, fmt="%.3f", fontsize=8, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, weight="bold")
    ax.set_ylabel("Score", weight="bold")
    ax.set_title("Generation Diversity & Repetition", weight="bold", pad=15)
    ax.legend(frameon=True)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Visuals] Diversity bar chart saved -> {output_path}")


def generate_intrinsic_comparison(eval_paths: Dict[str, Path], output_path: Path):
    """Side-by-side PPL and BPB comparison, Model H vs. Model L (different scales -> two subplots)."""
    results = _load_eval_results(eval_paths)
    available = {label: r["intrinsic_metrics"] for label, r in results.items()
                 if "intrinsic_metrics" in r}
    if not available:
        print("[Visuals] Skipping intrinsic metrics comparison -- no eval_results.json yet.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    labels = list(available.keys())
    colors = [LANGUAGE_COLORS.get(l, "#7f8c8d") for l in labels]

    ppl_vals = [available[l]["perplexity"] for l in labels]
    bpb_vals = [available[l]["bpb"] for l in labels]

    rects1 = ax1.bar(labels, ppl_vals, color=colors)
    ax1.bar_label(rects1, fmt="%.2f", padding=3)
    ax1.set_ylabel("Validation Perplexity", weight="bold")
    ax1.set_title("Perplexity", weight="bold")
    ax1.tick_params(axis="x", labelrotation=10)

    rects2 = ax2.bar(labels, bpb_vals, color=colors)
    ax2.bar_label(rects2, fmt="%.4f", padding=3)
    ax2.set_ylabel("Bits Per Byte (BPB)", weight="bold")
    ax2.set_title("Bits-Per-Byte", weight="bold")
    ax2.tick_params(axis="x", labelrotation=10)

    fig.suptitle("Intrinsic Language-Modeling Metrics: Model H vs. Model L", weight="bold", fontsize=14)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Visuals] Intrinsic metrics comparison saved -> {output_path}")


def generate_results_summary_table(eval_paths: Dict[str, Path], output_path: Path):
    """Writes a markdown table consolidating every scalar metric found across available languages."""
    results = _load_eval_results(eval_paths)
    if not results:
        print("[Visuals] Skipping results summary table -- no eval_results.json yet.")
        return

    rows = [
        ("Val Loss", "intrinsic_metrics", "val_loss", "{:.4f}"),
        ("Val Perplexity", "intrinsic_metrics", "perplexity", "{:.2f}"),
        ("Val BPB", "intrinsic_metrics", "bpb", "{:.4f}"),
        ("Test Loss", "test_intrinsic_metrics", "val_loss", "{:.4f}"),
        ("Test Perplexity", "test_intrinsic_metrics", "perplexity", "{:.2f}"),
        ("Test BPB", "test_intrinsic_metrics", "bpb", "{:.4f}"),
        ("BLEU-4", "generation_quality_metrics", "bleu_4", "{:.2f}"),
        ("chrF++", "generation_quality_metrics", "chrf_plus_plus", "{:.2f}"),
        ("ROUGE-L", "generation_quality_metrics", "rouge_l", "{:.2f}"),
        ("Distinct-1", "diversity_metrics", "distinct_1", "{:.4f}"),
        ("Distinct-2", "diversity_metrics", "distinct_2", "{:.4f}"),
        ("4-gram Repetition", "diversity_metrics", "repetition_rate_4gram", "{:.4f}"),
        ("Attention Entropy (L1,H1)", None, "attention_entropy_layer1_head1", "{:.4f}"),
        ("Attention Distance (L1,H1)", None, "attention_distance_layer1_head1", "{:.4f}"),
    ]
    labels = list(results.keys())

    lines = ["| Metric | " + " | ".join(labels) + " |", "|---|" + "---:|" * len(labels)]
    for name, section, key, fmt in rows:
        cells = []
        for label in labels:
            r = results[label]
            container = r.get(section, r) if section else r
            val = container.get(key) if isinstance(container, dict) else None
            cells.append(fmt.format(val) if val is not None else "--")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[Visuals] Results summary table saved -> {output_path}")


def sync_figures_to_drive(local_dirs: List[Path], remote_path: str = "gdrive:LMA_Project1/report_figures"):
    """
    Pushes every generated figure/table to a shared Drive folder via rclone, if rclone is
    available and configured. Silently skips (with a message) if rclone isn't set up --
    this is a convenience, not a requirement for the plots themselves to exist locally.
    """
    for local_dir in local_dirs:
        local_dir = Path(local_dir)
        if not local_dir.exists():
            continue
        try:
            subprocess.run(
                ["rclone", "copy", str(local_dir), remote_path, "-P"],
                check=False, timeout=600
            )
            print(f"[Visuals] Synced {local_dir} -> {remote_path}")
        except FileNotFoundError:
            print("[Visuals] rclone not found on this machine -- skipping Drive sync. "
                  "Figures are still saved locally.")
            return
        except Exception as e:
            print(f"[Visuals] Drive sync of {local_dir} failed: {e}")


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parents[1]
    out_dir = root_dir / "plots"

    history_paths = {
        "Telugu (Model H)": root_dir / "TELUGU/PHASE2/results/training_history.json",
        "Nepali (Model L)": root_dir / "NEPALI/PHASE2/results/training_history.json",
    }
    eval_paths = {
        "Telugu (Model H)": root_dir / "TELUGU/PHASE2/results/eval_results.json",
        "Nepali (Model L)": root_dir / "NEPALI/PHASE2/results/eval_results.json",
    }

    # Always-on figures (don't depend on training being done)
    generate_architecture_diagram(out_dir / "architecture_diagram.png")
    generate_corpus_comparison_chart(out_dir / "corpus_comparison.png")

    # Training diagnostics (need training_history.json)
    generate_loss_curves(list(history_paths.values()), out_dir / "pretrain_loss_curve.png")
    for lang_label, hist_path in history_paths.items():
        slug = "telugu" if "Telugu" in lang_label else "nepali"
        lang_plots_dir = root_dir / slug.upper() / "PHASE2" / "plots"
        generate_single_language_loss_curve(hist_path, lang_plots_dir / f"pretrain_loss_curve_{slug}.png", lang_label)
    generate_ppl_curve(history_paths, out_dir / "pretrain_ppl_curve_logscale.png")
    generate_lr_schedule(history_paths, out_dir / "lr_schedule.png")
    generate_throughput_chart(history_paths, out_dir / "training_throughput.png")

    # Evaluation-derived figures (need eval_results.json)
    generate_attention_stat_heatmap(
        eval_paths, "attention_entropy_all_layers_heads",
        "Attention Entropy by Layer and Head", "viridis",
        out_dir / "attention_entropy_heatmap.png"
    )
    generate_attention_stat_heatmap(
        eval_paths, "attention_distance_all_layers_heads",
        "Mean Attention Distance by Layer and Head", "magma",
        out_dir / "attention_distance_heatmap.png"
    )
    generate_generation_quality_bar(eval_paths, out_dir / "generation_quality_comparison.png")
    generate_diversity_bar(eval_paths, out_dir / "diversity_comparison.png")
    generate_intrinsic_comparison(eval_paths, out_dir / "intrinsic_metrics_comparison.png")
    generate_results_summary_table(eval_paths, out_dir / "results_summary.md")

    # Push just the figures + summary table to a shared Drive folder (best-effort; skips
    # cleanly if rclone unavailable).
    summary_dir = root_dir / "plots" / "_summary_for_drive"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_src = out_dir / "results_summary.md"
    if summary_src.exists():
        (summary_dir / "results_summary.md").write_text(summary_src.read_text(encoding="utf-8"), encoding="utf-8")
    sync_figures_to_drive([out_dir, summary_dir])

    print("\n[Visuals] Done. See plots/ for output.")
