"""
7_telugu_Tokenizer_stats.py

Calculates final corpus statistics from the ACTUAL train/val/test splits.
Produces:
  1. Console summary (overall + per-split)
  2. Markdown report with:
     - Overall totals (docs, words, chars, tokens)
     - Manual vs Downloaded fraction (must be >= 20% manual)
     - Per-split breakdown
     - Per-source breakdown
     - Fertility (avg chars per token)
     - UNK rate on val/test sets
"""

import json
from pathlib import Path
import sentencepiece as spm

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE = Path("/Users/erva_rishitha/Documents/LMA project1/TELUGU")

SPLITS_DIR = BASE / "outputs/data_splits"
SPLIT_FILES = {
    "train": SPLITS_DIR / "train.jsonl",
    "val":   SPLITS_DIR / "val.jsonl",
    "test":  SPLITS_DIR / "test.jsonl",
}

# The tokenizer file we trained in Step 6
TOKENIZER_MODEL_PATH = BASE / "outputs/tokenizer/telugu_bpe.model"

# Human-readable labels for source_type values
SOURCE_TYPE_LABELS = {
    "news_website":      "News Websites (scraped)",
    "govt_portal":       "Government Portals (scraped)",
    "literature_website":"Literature Websites (scraped)",
    "wikipedia_api":     "Wikipedia API",
    "wiktionary_api":    "Wiktionary API",
    "wikibooks_api":     "Wikibooks API",
    "wikisource_api":    "Wikisource API",
    "ocr_textbook":      "OCR – Textbooks (manual)",
    "archive_ocr":       "Archive OCR",
    "sangraha":          "Sangraha / AI4Bharat (downloaded)",
    "web_scrape":        "Web Scrape (generic)",
}

# Source types that count as "manual" contribution
MANUAL_SOURCE_TYPES = {
    "news_website", "govt_portal", "literature_website",
    "wikipedia_api", "wiktionary_api", "wikibooks_api", "wikisource_api",
    "ocr_textbook", "archive_ocr", "web_scrape",
}

# ==============================================================================
# HELPERS
# ==============================================================================

def load_split(path: Path):
    docs = []
    if not path.exists():
        print(f"Warning: split file not found: {path}")
        return docs
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def compute_unk_rate(sp, texts):
    total_tokens = total_unk = 0
    for text in texts:
        tokens = sp.encode_as_ids(text)
        total_tokens += len(tokens)
        total_unk    += tokens.count(sp.unk_id())
    rate = total_unk / total_tokens * 100 if total_tokens else 0.0
    return round(rate, 6), total_tokens


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    # Load tokenizer if available
    sp = None
    if TOKENIZER_MODEL_PATH.exists():
        print(f"Loading tokenizer: {TOKENIZER_MODEL_PATH}")
        sp = spm.SentencePieceProcessor()
        sp.load(str(TOKENIZER_MODEL_PATH))
    else:
        print(f"WARNING: Tokenizer not found at {TOKENIZER_MODEL_PATH}. Using word-count estimate (x1.3).")

    overall = {
        "total_docs": 0, "total_words": 0, "total_chars": 0, "total_tokens": 0,
        "manual_tokens": 0, "downloaded_tokens": 0,
    }
    per_split    = {}
    source_stats = {}   # source_type -> {docs, words, chars, tokens}
    all_val_texts  = []
    all_test_texts = []

    # ---------------------------------------------------------------
    for split_name, path in SPLIT_FILES.items():
        docs = load_split(path)
        ss = {"docs": 0, "words": 0, "chars": 0, "tokens": 0,
              "manual_tokens": 0, "downloaded_tokens": 0}

        for rec in docs:
            text        = rec.get("text", "")
            source_type = rec.get("source_type", rec.get("src", "unknown"))
            # fallback
            if not source_type:
                source_type = rec.get("source_tier", "unknown")

            words  = len(text.split())
            chars  = len(text)
            tokens = len(sp.encode_as_ids(text)) if sp else int(words * 1.3)

            ss["docs"]   += 1
            ss["words"]  += words
            ss["chars"]  += chars
            ss["tokens"] += tokens

            if source_type in MANUAL_SOURCE_TYPES:
                ss["manual_tokens"] += tokens
            else:
                ss["downloaded_tokens"] += tokens

            if source_type not in source_stats:
                source_stats[source_type] = {"docs": 0, "words": 0, "chars": 0, "tokens": 0}
            source_stats[source_type]["docs"]   += 1
            source_stats[source_type]["words"]  += words
            source_stats[source_type]["chars"]  += chars
            source_stats[source_type]["tokens"] += tokens

            if split_name == "val":
                all_val_texts.append(text)
            elif split_name == "test":
                all_test_texts.append(text)

        per_split[split_name] = ss
        overall["total_docs"]       += ss["docs"]
        overall["total_words"]      += ss["words"]
        overall["total_chars"]      += ss["chars"]
        overall["total_tokens"]     += ss["tokens"]
        overall["manual_tokens"]    += ss["manual_tokens"]
        overall["downloaded_tokens"]+= ss["downloaded_tokens"]

    # ---------------------------------------------------------------
    total_tokens = overall["total_tokens"]
    manual_pct   = overall["manual_tokens"] / total_tokens * 100 if total_tokens else 0.0
    dl_pct       = overall["downloaded_tokens"] / total_tokens * 100 if total_tokens else 0.0
    avg_cpt      = overall["total_chars"] / total_tokens if total_tokens else 0.0

    # UNK rates on held-out sets
    val_unk_pct  = test_unk_pct = 0.0
    if sp:
        val_unk_pct,  _ = compute_unk_rate(sp, all_val_texts)
        test_unk_pct, _ = compute_unk_rate(sp, all_test_texts)

    # ---------------------------------------------------------------
    # Console output
    print("\n" + "=" * 65)
    print("     FINAL DATASET STATISTICAL REPORT — TELUGU")
    print("=" * 65)
    print(f"Total Documents       : {overall['total_docs']:,}")
    print(f"Total Words           : {overall['total_words']:,}")
    print(f"Total Characters      : {overall['total_chars']:,}")
    print(f"Total Token Count     : {total_tokens:,}")
    print(f"Fertility (Chars/Tok) : {avg_cpt:.3f}")
    print(f"UNK Rate (val)        : {val_unk_pct:.4f}%")
    print(f"UNK Rate (test)       : {test_unk_pct:.4f}%")
    print("-" * 65)
    print(f"Manual     : {overall['manual_tokens']:,} tokens  ({manual_pct:.2f}%)")
    print(f"Downloaded : {overall['downloaded_tokens']:,} tokens  ({dl_pct:.2f}%)")
    if manual_pct < 20.0:
        print(">>> WARNING: Below the 20% manual requirement!")
    print("-" * 65)
    print("Per-split breakdown:")
    for sn, s in per_split.items():
        mp = s["manual_tokens"] / s["tokens"] * 100 if s["tokens"] else 0.0
        print(f"  {sn:>5}: {s['docs']:>8,} docs | {s['tokens']:>12,} tokens | manual={mp:.2f}%")
    print("=" * 65)
    print("Per-source breakdown:")
    for src, d in sorted(source_stats.items(), key=lambda x: x[1]["tokens"], reverse=True):
        pct   = d["tokens"] / total_tokens * 100 if total_tokens else 0.0
        label = SOURCE_TYPE_LABELS.get(src, src)
        print(f"  {label:<40}: {d['docs']:>8,} docs | {d['tokens']:>12,} tokens ({pct:.2f}%)")
    print("=" * 65 + "\n")

    # ---------------------------------------------------------------
    # Markdown report
    md = []
    md.append("## Telugu Corpus — Final Dataset Statistics\n")
    md.append("### Overall Statistics\n")
    md.append("| Metric | Value |")
    md.append("| :--- | :--- |")
    md.append(f"| **Total Documents** | {overall['total_docs']:,} |")
    md.append(f"| **Total Words** | {overall['total_words']:,} |")
    md.append(f"| **Total Characters** | {overall['total_chars']:,} |")
    md.append(f"| **Total Tokens** | **{total_tokens:,}** |")
    md.append(f"| **Fertility (Avg Chars/Token)** | {avg_cpt:.4f} |")
    md.append(f"| **UNK Rate (val set)** | {val_unk_pct:.4f}% |")
    md.append(f"| **UNK Rate (test set)** | {test_unk_pct:.4f}% |")
    md.append(f"| **Manual Tokens** | **{overall['manual_tokens']:,}** ({manual_pct:.2f}%) |")
    md.append(f"| **Downloaded Tokens** | {overall['downloaded_tokens']:,} ({dl_pct:.2f}%) |")
    if manual_pct < 20.0:
        md.append("\n> ⚠️ **Warning**: Manual fraction is below the 20% requirement.\n")
    md.append("")

    md.append("### Per-Split Breakdown\n")
    md.append("| Split | Docs | Tokens | Manual % |")
    md.append("| :--- | ---: | ---: | ---: |")
    for sn, s in per_split.items():
        mp = s["manual_tokens"] / s["tokens"] * 100 if s["tokens"] else 0.0
        md.append(f"| {sn} | {s['docs']:,} | {s['tokens']:,} | {mp:.2f}% |")
    md.append("")

    md.append("### Per-Source Breakdown\n")
    md.append("| Source | Docs | Words | Tokens | % of Total |")
    md.append("| :--- | ---: | ---: | ---: | ---: |")
    for src, d in sorted(source_stats.items(), key=lambda x: x[1]["tokens"], reverse=True):
        pct   = d["tokens"] / total_tokens * 100 if total_tokens else 0.0
        label = SOURCE_TYPE_LABELS.get(src, src)
        md.append(f"| {label} | {d['docs']:,} | {d['words']:,} | {d['tokens']:,} | {pct:.2f}% |")
    md.append("")

    report_dir  = BASE / "outputs/report"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_file = report_dir / "telugu_dataset_stats.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"Markdown report saved -> {report_file}")


if __name__ == "__main__":
    main()
