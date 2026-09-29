"""
5_telugu_ablation_vocabsize.py

Runs a vocabulary size ablation study on a ~10M-word subset of the training data.
Vocab sizes tested: [8000, 10000, 12000, 16000, 32000] -- matches Nepali's ablation set.

Results (bytes-per-token and UNK-rate) guide the final vocab choice in step 6.
"""

import json
import random
import time
from pathlib import Path
from typing import Dict, List

import sentencepiece as spm

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE = Path("/Users/erva_rishitha/Documents/LMA project1/TELUGU")

TRAIN_JSONL_PATH = BASE / "outputs/data_splits/train.jsonl"
VAL_JSONL_PATH   = BASE / "outputs/data_splits/val.jsonl"
TEST_JSONL_PATH  = BASE / "outputs/data_splits/test.jsonl"

OUTPUT_DIR       = BASE / "outputs/ablation"

VOCAB_SIZES          = [8000, 10000, 12000, 16000, 32000]
TARGET_WORD_COUNT    = 10_000_000
MODEL_TYPE           = "bpe"
CHARACTER_COVERAGE   = 0.9998
INPUT_SENTENCE_SIZE  = 5_000_000
RANDOM_SEED          = 42

# ==============================================================================
# HELPERS
# ==============================================================================

def load_jsonl_texts_by_words(path: Path, target_words: int = None, seed: int = 42):
    texts = []
    total_words = 0

    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    if target_words:
        random.seed(seed)
        random.shuffle(lines)

    for line in lines:
        line = line.strip()
        if not line:
            continue
        rec  = json.loads(line)
        text = rec.get("text", "")
        if text:
            wc = len(text.split())
            texts.append(text)
            total_words += wc
            if target_words and total_words >= target_words:
                break

    return texts, total_words


def create_plain_text_corpus(texts: List[str], out_txt_path: Path) -> int:
    line_count = 0
    with out_txt_path.open("w", encoding="utf-8") as f_out:
        for doc in texts:
            for line in doc.split("\n"):
                line = line.strip()
                if line:
                    f_out.write(line + "\n")
                    line_count += 1
    return line_count


def evaluate_tokenizer(sp_model_path: Path, eval_texts: List[str]) -> Dict:
    sp = spm.SentencePieceProcessor()
    sp.load(str(sp_model_path))

    total_bytes = total_tokens = total_unk = 0
    for doc in eval_texts:
        tokens = sp.encode_as_ids(doc)
        total_bytes  += len(doc.encode("utf-8"))
        total_tokens += len(tokens)
        total_unk    += tokens.count(sp.unk_id())

    n = len(eval_texts)
    return {
        "num_documents":    n,
        "total_bytes":      total_bytes,
        "total_tokens":     total_tokens,
        "bytes_per_token":  round(total_bytes / total_tokens, 4) if total_tokens else 0,
        "avg_tokens_per_doc": round(total_tokens / n, 2) if n else 0,
        "unk_token_count":  total_unk,
        "unk_rate_pct":     round(total_unk / total_tokens * 100, 6) if total_tokens else 0,
    }


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("--- Loading Datasets ---")
    train_texts, actual_train_words = load_jsonl_texts_by_words(
        TRAIN_JSONL_PATH, target_words=TARGET_WORD_COUNT, seed=RANDOM_SEED
    )
    print(f"Train subset: {len(train_texts):,} docs, {actual_train_words:,} words")

    val_texts,  val_words  = load_jsonl_texts_by_words(VAL_JSONL_PATH)
    test_texts, test_words = load_jsonl_texts_by_words(TEST_JSONL_PATH)
    print(f"Val : {len(val_texts):,} docs, {val_words:,} words")
    print(f"Test: {len(test_texts):,} docs, {test_words:,} words")

    plain_corpus_path = OUTPUT_DIR / f"train_subset_{actual_train_words}_words.txt"
    print(f"\nFlattening corpus -> {plain_corpus_path}")
    n_lines = create_plain_text_corpus(train_texts, plain_corpus_path)
    print(f"Created {n_lines:,} lines.")

    ablation_summary = []

    print("\n" + "=" * 80)
    print(f"VOCABULARY SIZE ABLATION ON ~{actual_train_words:,} WORDS (TELUGU)")
    print("=" * 80)

    for vocab_size in VOCAB_SIZES:
        print(f"\n>>> Ablation vocab_size={vocab_size:,}")
        model_prefix   = OUTPUT_DIR / f"spm_telugu_vocab_{vocab_size}"
        spm_model_file = OUTPUT_DIR / f"spm_telugu_vocab_{vocab_size}.model"

        t0 = time.time()
        spm.SentencePieceTrainer.train(
            input=str(plain_corpus_path),
            model_prefix=str(model_prefix),
            vocab_size=vocab_size,
            model_type=MODEL_TYPE,
            character_coverage=CHARACTER_COVERAGE,
            input_sentence_size=max(INPUT_SENTENCE_SIZE, n_lines),
            shuffle_input_sentence=True,
            byte_fallback=True,
            normalization_rule_name="nmt_nfkc",
            pad_id=0, unk_id=1, bos_id=2, eos_id=3,
            pad_piece="<pad>", unk_piece="<unk>", bos_piece="<s>", eos_piece="</s>",
            num_threads=4,
        )
        train_secs = round(time.time() - t0, 2)
        print(f"  Trained in {train_secs}s")

        val_eval  = evaluate_tokenizer(spm_model_file, val_texts)
        test_eval = evaluate_tokenizer(spm_model_file, test_texts)

        ablation_summary.append({
            "vocab_size":             vocab_size,
            "training_subset_words":  actual_train_words,
            "training_time_seconds":  train_secs,
            "val_metrics":            val_eval,
            "test_metrics":           test_eval,
        })

        print(f"  Val  -> Bytes/Token: {val_eval['bytes_per_token']}  UNK%: {val_eval['unk_rate_pct']}")
        print(f"  Test -> Bytes/Token: {test_eval['bytes_per_token']}  UNK%: {test_eval['unk_rate_pct']}")

    report_path = OUTPUT_DIR / "vocab_ablation_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(ablation_summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("ABLATION RESULTS SUMMARY (TEST SET) — TELUGU")
    print("=" * 80)
    print(f"{'Vocab':>10} | {'Train(s)':>10} | {'Bytes/Tok':>10} | {'Tok/Doc':>10} | {'UNK%':>10}")
    print("-" * 80)
    for res in ablation_summary:
        t = res["test_metrics"]
        print(f"{res['vocab_size']:>10,} | {res['training_time_seconds']:>10.2f} | "
              f"{t['bytes_per_token']:>10.4f} | {t['avg_tokens_per_doc']:>10.2f} | {t['unk_rate_pct']:>10.6f}")
    print("=" * 80)
    print(f"\nFull report -> {report_path}")


if __name__ == "__main__":
    main()
