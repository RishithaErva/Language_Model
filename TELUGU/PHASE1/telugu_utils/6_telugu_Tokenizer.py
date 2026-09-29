"""
6_telugu_Tokenizer.py

Trains a SentencePiece BPE tokenizer from scratch on the ENTIRE training dataset.

VOCAB_SIZE finalized at 12000 after reviewing the ablation results in
5_telugu_ablation_vocabsize.py: it keeps embedding parameters to a modest share of the
~25M-parameter budget while still capturing most of the compression gain over smaller
vocab sizes, and it matches the Nepali tokenizer's vocab size for cross-model comparability.
"""

import json
from pathlib import Path
import sentencepiece as spm

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE = Path("/Users/erva_rishitha/Documents/LMA project1/TELUGU")

TRAIN_JSONL_PATH   = BASE / "outputs/data_splits/train.jsonl"
TOKENIZER_OUT_DIR  = BASE / "outputs/tokenizer"
OUT_PREFIX         = TOKENIZER_OUT_DIR / "telugu_bpe"

VOCAB_SIZE         = 12000    # finalized after ablation (see 5_telugu_ablation_vocabsize.py)
MODEL_TYPE         = "bpe"
CHARACTER_COVERAGE = 0.9998
NUM_THREADS        = 8

# ==============================================================================
# HELPERS
# ==============================================================================

def jsonl_to_plain_text(jsonl_path: Path, out_txt_path: Path) -> int:
    n_lines = 0
    with jsonl_path.open("r", encoding="utf-8") as f_in, \
         out_txt_path.open("w", encoding="utf-8") as f_out:

        for raw_line in f_in:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            rec  = json.loads(raw_line)
            text = rec.get("text", "")
            if not text:
                continue
            for ln in text.split("\n"):
                ln = ln.strip()
                if ln:
                    f_out.write(ln + "\n")
                    n_lines += 1

            if n_lines % 5_000_000 == 0 and n_lines > 0:
                print(f"[tokenizer] Processed {n_lines:,} lines so far...")

    return n_lines


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    TOKENIZER_OUT_DIR.mkdir(parents=True, exist_ok=True)
    plain_txt = TOKENIZER_OUT_DIR / f"telugu_bpe_{VOCAB_SIZE}_sp_input.txt"

    print(f"[tokenizer] Flattening full training dataset: {TRAIN_JSONL_PATH}")
    n_lines = jsonl_to_plain_text(TRAIN_JSONL_PATH, plain_txt)
    print(f"[tokenizer] Prepared {n_lines:,} lines -> {plain_txt}")

    print(f"[tokenizer] Training {MODEL_TYPE.upper()} tokenizer (vocab={VOCAB_SIZE:,}) on full data...")
    spm.SentencePieceTrainer.train(
        input=str(plain_txt),
        model_prefix=str(OUT_PREFIX),
        vocab_size=VOCAB_SIZE,
        model_type=MODEL_TYPE,
        character_coverage=CHARACTER_COVERAGE,
        input_sentence_size=0,       # 0 = use all lines
        shuffle_input_sentence=True,
        byte_fallback=True,
        normalization_rule_name="nmt_nfkc",
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        pad_piece="<pad>", unk_piece="<unk>", bos_piece="<s>", eos_piece="</s>",
        num_threads=NUM_THREADS,
    )

    if plain_txt.exists():
        plain_txt.unlink()
        print(f"[tokenizer] Cleaned up {plain_txt}")

    print(f"[tokenizer] Done.")
    print(f"[tokenizer] Model -> {OUT_PREFIX}.model")
    print(f"[tokenizer] Vocab -> {OUT_PREFIX}.vocab")


if __name__ == "__main__":
    main()
