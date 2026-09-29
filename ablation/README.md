# Bonus Ablation — Model L (Nepali)

The assignment's optional **Bonus** section: positional embeddings removed
entirely from Model L (Nepali), retrained from scratch on Kaggle, and evaluated
against the standard model. **This ablation is complete** — [`results/`](results/)
holds the real output, and [`nepali_ablation.ipynb`](nepali_ablation.ipynb) is the
actual Kaggle notebook run, with its outputs intact.

Language chosen: **Nepali** (the lower-resource of the two models) — a smaller,
less redundant corpus makes the effect of removing position information easier
to see.

## What's here

```
ablation/
├── README.md                    (this file)
├── nepali_ablation.ipynb        the actual Kaggle notebook run, outputs intact
├── train_ablation.py            python train_ablation.py --variant no_pos
├── eval_ablation.py             python eval_ablation.py  --variant no_pos
├── configs/
│   └── model_l_no_pos_config.json
├── src/
│   ├── __init__.py
│   └── model.py                 forked from the main src/model.py — adds a
│                                 use_positional_embeddings flag (default = standard
│                                 model, unchanged). This is the ONLY file that
│                                 differs; dataset.py/trainer.py/metrics.py/
│                                 attention_analyzer.py are unmodified, so
│                                 train_ablation.py/eval_ablation.py import those
│                                 straight from the main ../src/ instead of
│                                 duplicating them here.
└── results/
    ├── training_history.json    loss/PPL curve from the actual training run
    ├── plots/                   attention heatmaps, standard vs. ablated
    └── results/
        ├── comparison_no_pos_vs_standard.md   the required side-by-side table
        ├── eval_results_no_pos.json           full raw metrics, ablated model
        └── eval_results_standard.json         full raw metrics, standard model
```

The main graded `src/model.py` used by `NEPALI/PHASE2` and `TELUGU/PHASE2` is **not
touched** — `ablation/src/model.py` is a separate fork, so there was zero risk to
already-graded Phase 2 code. This folder lives inside the full repo (a sibling of
`NEPALI/`, `TELUGU/`, `src/`), so `train_ablation.py`/`eval_ablation.py` resolve
the Nepali data, tokenizer, and standard-model checkpoint from there automatically
— no bundled copies, no Drive pull needed for a local run.

`src/model.py` also supports an optional `norm_type="rmsnorm"` flag from an
earlier extra-ablation idea that isn't part of this submission — it's inert
unless explicitly passed, and `configs/model_l_no_pos_config.json` doesn't set
it, so it has no effect on the results below.

## The actual result

| Metric | Standard Model L | Ablated (no positional embeddings) |
|---|---|---|
| Val Perplexity | 33.58 | 36.48 |
| Val BPB | 0.428 | 0.439 |
| Test Perplexity | 33.09 | 36.00 |
| BLEU-4 | 3.21 | 2.28 |
| chrF++ | 22.73 | 21.82 |
| ROUGE-L | 6.94 | 5.76 |
| Distinct-1 / Distinct-2 | 0.620 / 0.665 | 0.571 / 0.661 |
| Attention Entropy (Layer 1, Head 1) | 1.108 | **0.546** |
| Attention Distance (Layer 1, Head 1) | 1.929 | **0.562** |

Full table: [`results/results/comparison_no_pos_vs_standard.md`](results/results/comparison_no_pos_vs_standard.md).

**What actually broke without position information**, based on these numbers
(not a generic guess — this is what this run showed): perplexity, BPB, and every
generation-quality metric (BLEU/chrF/ROUGE-L) got modestly worse, as expected.
The more striking effect is on attention: entropy roughly **halved** and mean
attention distance **dropped by ~3x**. That means attention got *more* localized
and peaked without positional embeddings, not more diffuse — with no positional
signal to distinguish "near" from "far," each head appears to fall back on
content-based matching to nearby/repeated tokens, rather than spreading attention
across the sequence the way the standard model's position-aware heads do. Compare
`results/plots/nepali_no_pos_attn_*` against `nepali_standard_attn_*` for the
visual version of this.

## Reproducing

```bash
cd ablation   # must be inside the full repo, sibling of NEPALI/
python train_ablation.py --variant no_pos
python eval_ablation.py  --variant no_pos
```



