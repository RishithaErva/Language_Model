# Building and Analyzing Monolingual Transformer Language Models in Two Indian Languages



---

## Executive Summary

This repository contains the complete codebase, data engine, training infrastructure, evaluation pipelines, and reports for building and analyzing two completely independent monolingual decoder-only (GPT-style) Transformer language models from scratch in PyTorch:

| Model | Language | Script | Family | Resource Tier | Manual Data Share |
|---|---|---|---|---|---|
| **Model H** | Telugu | Telugu | Dravidian | Higher-resource | **28.12%** ($\ge 20\%$ required) |
| **Model L** | Nepali | Devanagari | Indo-Aryan | Lower-resource | **20.22%** ($\ge 20\%$ required) |

Neither model shares parameters, vocabularies, tokenizers, pretraining data, synthetic reasoning datasets, or checkpoint weights at any point in the pipeline.

---

## Complete Project Results at a Glance

### Phase 1: Data Engine & Tokenization
- **Corpora Volume:** Telugu: **547,479,659 tokens** (983,814 docs); Nepali: **526,615,295 tokens** (1,186,541 docs).
- **Manual Data Verification:** Telugu **28.12%** manual share; Nepali **20.22%** manual share.
- **Tokenizers:** SentencePiece Byte-Pair Encoding (BPE), vocabulary size $V = 12,000$, byte fallback enabled ($0.0000\%$ UNK rate).
- **Fertility:** Telugu: **4.1275** chars/token; Nepali: **4.4793** chars/token.

### Phase 2: Primitive Transformer Implementation & Pretraining
- **Architecture:** 25,321,472 parameters, 6 layers, 8 heads ($d_k=64$), $d_\text{model}=512$, $d_\text{ff}=2048$, Pre-LN residual connections, tied embedding/output weights ($W_\text{out} \equiv W_\text{te}$).
- **Pretraining Budget:** 30,500 steps ($\sim$499.7M tokens) per model using AdamW ($6 \times 10^{-4}$ cosine decay).
- **Intrinsic Evaluation (Test Set):**
  - **Telugu (Model H):** Test Loss **3.6558**, Perplexity **38.70**, BPB **0.4827**.
  - **Nepali (Model L):** Test Loss **3.4991**, Perplexity **33.09**, BPB **0.4256**.
- **Extrinsic Generation (Greedy Continuation):**
  - **Telugu (Model H):** BLEU-4 **4.05**, chrF++ **19.69**, ROUGE-L **3.92**, Distinct-1/2 **0.60/0.67**.
  - **Nepali (Model L):** BLEU-4 **3.21**, chrF++ **22.73**, ROUGE-L **6.94**, Distinct-1/2 **0.63/0.69**.

### Phase 3: Synthetic Reasoning Finetuning & Attention Interpretability
- **Pretrained Baseline:** **0.00%** exact-match accuracy across all reasoning tasks (open-ended narrative output).
- **Finetuned Reasoning Results (Exact-Match Test Accuracy):**
  - **Telugu (Model H):** **64.69%** overall accuracy
    - Pairwise Comparison: **46.00%**
    - Threeway Comparison: **21.20%**
    - Transitive 2-Hop Chain: **99.60%**
    - Transitive 4-Hop Held-Out Probe: **81.95%**
  - **Nepali (Model L):** **52.74%** overall accuracy
    - Pairwise Comparison: **32.80%**
    - Threeway Comparison: **11.60%**
    - Transitive 2-Hop Chain: **88.40%**
    - Transitive 4-Hop Held-Out Probe: **68.67%**
- **Numeric vs. Relational Failure Mode:** Both models saturate relational chain reasoning (88--100%), but experience a severe bottleneck on multi-digit numerical comparison (11--46%) due to BPE digit-splitting.
- **Attention Dynamics:** Finetuning sharpens late-layer (Layer 6) attention onto answer entities, reducing attention entropy (Te 1.26 $\to$ 1.16, Ne 1.37 $\to$ 1.30) and shortening attention distance by $\sim$6.4--7.1%.

### Bonus Ablation: Positional Embeddings Removal (Model L)
- Retrained Model L from scratch without positional embeddings ($W_\text{wpe} \equiv 0$).
- **Impact:** Test PPL degrades ($33.09 \to 36.00$), BLEU-4 drops ($3.21 \to 2.28$), and Layer 1 attention entropy drops by half ($1.108 \to 0.546$), proving explicit position embeddings are necessary for long-range context dispersion.

---

## Deliverables & Documentation Index

| Deliverable | Description | File Path |
|---|---|---|
| **Phase 1 Report (PDF)** | Detailed data curation, deduplication, and tokenizer ablation report | [`reports/2025201027_phase1_report.pdf`](reports/2025201027_phase1_report.pdf) |
| **Phase 2 Report (PDF)** | Transformer architecture, pretraining dynamics, and attention analysis | [`reports/2025201027_phase2_report.pdf`](reports/2025201027_phase2_report.pdf) |
| **Phase 3 Report (PDF)** | Reasoning finetuning, loss masking, curriculum rebalancing, and post-finetune attention | [`reports/2025201027_phase3_report.pdf`](reports/2025201027_phase3_report.pdf) |
| **Bonus Ablation Summary** | Positional embedding removal experiment documentation | [`ablation/README.md`](ablation/README.md) |

---

## Repository Directory Structure

```
LMA_Project1/
├── README.md                            # Master repository README (this file)
│
├── reports/                             # COMPILED PHASE PDF REPORTS
│   ├── 2025201027_phase1_report.pdf     # Phase 1 compiled PDF report
│   ├── 2025201027_phase2_report.pdf     # Phase 2 compiled PDF report
│   └── 2025201027_phase3_report.pdf     # Phase 3 compiled PDF report
│
├── TELUGU/                              # MODEL H (TELUGU) PIPELINE
│   ├── PHASE1/                          # Phase 1: Data engine (collection, utils, tokenizer)
│   │   ├── data_collection_codes/       # Scrapers, API crawlers, OCR notebooks
│   │   ├── telugu_utils/                # Numbered data pipeline scripts (0_trim ... 7_stats)
│   │   ├── tokenizer/                   # telugu_bpe.model, telugu_bpe.vocab
│   │   └── telugu_dataset_stats.md      # Corpus statistics report
│   ├── PHASE2/                          # Phase 2: Pretraining & evaluation
│   │   ├── configs/                     # model_h_config.json
│   │   ├── train_telugu.py              # Pretraining entry script
│   │   ├── eval_telugu.py               # Evaluation entry script
│   │   ├── telugu_train_kaggle.ipynb    # Kaggle execution notebook
│   │   ├── plots/                       # Per-language loss curves & attention heatmaps
│   │   └── results/                     # eval_results.json, training_history.json
│   └── PHASE3/                          # Phase 3: Reasoning finetuning & attention
│       ├── reasoning_data/              # generate_telugu_reasoning.py
│       ├── finetune/                    # finetune_telugu_reasoning.py
│       ├── eval/                        # eval_reasoning_telugu.py & results
│       └── attention/                   # attention_reasoning_telugu.py & heatmaps
│
├── NEPALI/                              # MODEL L (NEPALI) PIPELINE
│   ├── PHASE1/                          # Phase 1: Data engine (collection, utils, tokenizer)
│   │   ├── data_collection_codes/       # Scrapers, web crawlers, OCR notebooks
│   │   ├── nepali_utils/                # Numbered data pipeline scripts (0_trim ... 7_stats)
│   │   ├── tokenizer/                   # nepali_bpe.model, nepali_bpe.vocab
│   │   └── nepali_dataset_stats.md      # Corpus statistics report
│   ├── PHASE2/                          # Phase 2: Pretraining & evaluation
│   │   ├── configs/                     # model_l_config.json
│   │   ├── train_nepali.py              # Pretraining entry script
│   │   ├── eval_nepali.py               # Evaluation entry script
│   │   ├── nepali_train_kaggle.ipynb    # Kaggle execution notebook
│   │   ├── plots/                       # Per-language loss curves & attention heatmaps
│   │   └── results/                     # eval_results.json, training_history.json
│   └── PHASE3/                          # Phase 3: Reasoning finetuning & attention
│       ├── reasoning_data/              # generate_nepali_reasoning.py
│       ├── finetune/                    # finetune_nepali_reasoning.py
│       ├── eval/                        # eval_reasoning_nepali.py & results
│       └── attention/                   # attention_reasoning_nepali.py & heatmaps
│
├── ablation/                            # BONUS ABLATION (MODEL L NO-POSITIONAL)
│   ├── README.md                        # Ablation methodology & comparison table
│   ├── train_ablation.py / eval_ablation.py # Entry scripts for no-positional model
│   ├── nepali_ablation.ipynb            # Kaggle execution notebook
│   ├── configs/                         # model_l_no_pos_config.json
│   ├── src/                             # Model fork with use_positional_embeddings flag
│   └── results/                         # Raw JSON results & comparison table
│
├── src/                                 # CORE SHARED PYTORCH MODULES
│   ├── model.py                         # Decoder-only GPT implementation from scratch
│   ├── dataset.py                       # Tokenizer wrapper & memmap TextDataset
│   ├── trainer.py                       # Pretraining loop, checkpointing, AMP
│   ├── finetuner.py                     # Finetuning loop with answer span loss masking
│   ├── metrics.py                       # PPL, BPB, BLEU, chrF++, ROUGE, diversity
│   ├── attention_analyzer.py            # Multi-head attention map extractor & entropy/distance math
│   ├── generate_report_visuals.py       # Automated visualization generator
│   └── reasoning_data_utils.py          # Synthetic prompt generator & pool partitioner
│
├── plots/                               # Cross-language comparative figures (10 PNGs)
└── tests/                               # PyTorch unit tests (causal mask check, parameter count)
```

---

## Cloud Storage Architecture & Drive Structure

To keep the git repository lightweight, raw datasets, preprocessed JSONL files, large data splits, and model checkpoints are archived in cloud storage across OneDrive and Google Drive.

### 1. OneDrive Archive — Primary Data Storage (`2025201027_LMA_PROJECT1`)
- **Direct Link:** [OneDrive Shared Folder](https://iiithydstudents-my.sharepoint.com/:f:/g/personal/erva_rishitha_students_iiit_ac_in/IgD-8rpK5sB2ToU2dHxInaDgAexe9qPoSAgxOmLYsoZzntU?e=NYLiqA)
- **Directory Structure:**
  ```
  2025201027_LMA_PROJECT1/
  ├── MODEL_H(TELUGU)/
  │   ├── RAW_DATA/
  │   │   ├── downloaded/          # Raw Sangraha dump (pre-trim)
  │   │   └── manual/              # Raw web-crawl & OCR extracted text
  │   ├── FINAL_DATA/              # Cleaned & deduplicated JSONL files
  │   ├── tokenizer_outputs/       # telugu_bpe.model, telugu_bpe.vocab
  │   └── data_splits/
  │       ├── train.jsonl          # Pretraining train split (~542M tokens)
  │       ├── val.jsonl            # Pretraining val split (~2.8M tokens)
  │       ├── test.jsonl           # Pretraining test split (~2.6M tokens)
  │       └── split_stats.json
  └── MODEL_L(NEPALI)/
      ├── RAW_DATA/
      │   ├── downloaded/          # Raw Sangraha dump (pre-trim)
      │   └── manual/              # Raw web-crawl & OCR extracted text
      ├── FINAL_DATA/              # Cleaned & deduplicated JSONL files
      ├── tokenizer_outputs/       # nepali_bpe.model, nepali_bpe.vocab
      └── data_splits/
          ├── train.jsonl          # Pretraining train split (~521M tokens)
          ├── val.jsonl            # Pretraining val split (~2.6M tokens)
          ├── test.jsonl           # Pretraining test split (~2.5M tokens)
          └── split_stats.json
  ```

### 2. Google Drive Phase 1 Archive — OCR & Reference (`2025201027_LMA_Phase1`)
- **Direct Link:** [Google Drive Phase 1 Folder](https://drive.google.com/drive/folders/1Glfk6Va4mmdiIPYsNO0sr5NLJoB5wZFk?usp=sharing)
- **Directory Structure:**
  ```
  2025201027_LMA_Phase1/
  ├── DATASET_TELUGU/
  │   ├── BOOKS/                   # Textbook PDFs & Internet Archive PDFs used for OCR
  │   └── OUTPUTS/                 # Raw per-source collection outputs (Colab/Kaggle runs)
  └── DATASET_NEPALI/
      ├── BOOKS/                   # Textbook PDFs used for OCR
      └── OUTPUTS/                 # Raw per-source collection outputs
  ```

### 3. Google Drive Phase 2 Archive — Pretraining Checkpoints (`2025201027_LMA_Phase2`)
- **Direct Link:** [Google Drive Phase 2 Folder](https://drive.google.com/drive/folders/1KBBBZxKp5BHbilisJKHcqwoTM8r9Quc9?usp=sharing)
- **Directory Structure:**
  ```
  2025201027_LMA_Phase2/
  ├── TELUGU_checkpoints/
  │   ├── checkpoint_final.pt          # Step 30,500 final weights + optimizer state
  │   ├── checkpoint_latest.pt         # Resume checkpoint
  │   ├── checkpoint_step25000.pt      # Step 25,000 milestone snapshot
  │   ├── checkpoint_step30000.pt      # Step 30,000 milestone snapshot
  │   ├── training_history.json        # Step-wise loss & perplexity logs
  │   └── report/                      # eval_results.json & attention heatmaps
  ├── NEPALI_checkpoints/
  │   ├── checkpoint_final.pt          # Step 30,500 final weights + optimizer state
  │   ├── checkpoint_latest.pt         # Resume checkpoint
  │   ├── checkpoint_step25000.pt      # Step 25,000 milestone snapshot
  │   ├── checkpoint_step30000.pt      # Step 30,000 milestone snapshot
  │   ├── training_history.json        # Step-wise loss & perplexity logs
  │   └── report/                      # eval_results.json & attention heatmaps
  └── report_figures/                  # Synced cross-language comparison plots
  ```

### 4. Google Drive Phase 3 Archive — Reasoning Checkpoints & Datasets (`2025201027_LMA_Phase3`)
- **Direct Link:** [Google Drive Phase 3 Folder](https://drive.google.com/drive/folders/17OPgdPoxmgmsE3n_r3qF5I3cgo4etBkn?usp=sharing)

- **Directory Structure:**
  ```
  2025201027_LMA_Phase3/
  ├── TELUGU/
  │   ├── data/                        # Synthetic reasoning split files
  │   │   ├── train.jsonl
  │   │   ├── val.jsonl
  │   │   └── test.jsonl
  │   └── checkpoints/                 # Finetuned checkpoint weights
  │       ├── checkpoint_latest.pt     # Resume checkpoint
  │       └── checkpoint_final.pt      # Step 600 selected checkpoint
  └── NEPALI/
      ├── data/                        # Synthetic reasoning split files
      │   ├── train.jsonl
      │   ├── val.jsonl
      │   └── test.jsonl
      └── checkpoints/                 # Finetuned checkpoint weights
          ├── checkpoint_latest.pt     # Resume checkpoint
          └── checkpoint_final.pt      # Step 600 selected checkpoint
  ```

---

## Reproduction Guide

### Environment Setup
```bash
# Clone the repository
cd LMA_Project1

# Create and activate a Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch sentencepiece sacrebleu rouge-score numpy matplotlib seaborn datasketch beautifulsoup4 requests pymupdf easyocr mmh3
```

### Phase 1: Data Preprocessing & Tokenizer Training
```bash
# Execute for Telugu (or NEPALI/nepali_utils for Nepali)
cd TELUGU/telugu_utils
python3 0_trim_raw_downloaded.py
python3 1_telugu_preprocess_fulldata.py
python3 2_telugu_exact_deduplication.py
python3 3_telugu_near_deduplication.py
python3 4_telugu_split_data.py
python3 5_telugu_ablation_vocabsize.py
python3 6_telugu_Tokenizer.py
python3 7_telugu_Tokenizer_stats.py
```

### Phase 2: Pretraining & Evaluation
```bash
# Pretrain decoder-only GPT (resumes automatically from checkpoints/ if present)
python3 TELUGU/PHASE2/train_telugu.py
python3 NEPALI/PHASE2/train_nepali.py

# Evaluate intrinsic/extrinsic metrics & generate attention heatmaps
python3 TELUGU/PHASE2/eval_telugu.py
python3 NEPALI/PHASE2/eval_nepali.py

# Generate comparison plots
python3 src/generate_report_visuals.py
```

### Phase 3: Synthetic Reasoning Data Generation & Finetuning
```bash
# Generate synthetic comparative reasoning datasets
python3 TELUGU/PHASE3/reasoning_data/generate_telugu_reasoning.py
python3 NEPALI/PHASE3/reasoning_data/generate_nepali_reasoning.py

# Finetune models on reasoning task
python3 TELUGU/PHASE3/finetune/finetune_telugu_reasoning.py
python3 NEPALI/PHASE3/finetune/finetune_nepali_reasoning.py

# Evaluate exact-match accuracy & run post-finetune attention analysis
python3 TELUGU/PHASE3/eval/eval_reasoning_telugu.py
python3 NEPALI/PHASE3/eval/eval_reasoning_nepali.py
python3 TELUGU/PHASE3/attention/attention_reasoning_telugu.py
python3 NEPALI/PHASE3/attention/attention_reasoning_nepali.py
```

### Bonus Ablation: No-Positional Model L
```bash
cd ablation
python train_ablation.py --variant no_pos
python eval_ablation.py --variant no_pos
```
