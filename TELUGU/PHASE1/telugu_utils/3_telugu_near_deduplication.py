"""
3_telugu_near_deduplication.py

Near-duplicate detection using MinHash + LSH (via datasketch).
Runs AFTER 2_telugu_exact_deduplication.py.

Shingling: word 5-grams (whitespace split).
Priority: Manual inserted into index first and always kept;
Sangraha documents landing in the same LSH bucket are dropped.
"""

import json
from pathlib import Path
from tqdm import tqdm
from datasketch import MinHash, MinHashLSH

# ==============================================================================
# PATHS
# ==============================================================================
BASE = Path("/Users/erva_rishitha/Documents/LMA project1/TELUGU")

MANUAL_INPUT      = BASE / "outputs/exact_dedup/manual.jsonl"
DOWNLOADED_INPUT  = BASE / "outputs/exact_dedup/downloaded.jsonl"
MANUAL_OUTPUT     = BASE / "outputs/near_dedup/manual.jsonl"
DOWNLOADED_OUTPUT = BASE / "outputs/near_dedup/downloaded.jsonl"

# LSH Configuration
THRESHOLD    = 0.85
NUM_PERM     = 128
SHINGLE_SIZE = 5

# ==============================================================================
# HELPERS
# ==============================================================================

def word_shingles(words: list, n: int = 5):
    if len(words) < n:
        yield " ".join(words)
        return
    for i in range(len(words) - n + 1):
        yield " ".join(words[i: i + n])


def make_minhash(words: list, num_perm: int, shingle_size: int) -> MinHash:
    mh = MinHash(num_perm=num_perm)
    for sh in word_shingles(words, shingle_size):
        mh.update(sh.encode("utf-8"))
    return mh


def process_stream(in_path: Path, out_path: Path, lsh: MinHashLSH,
                   num_perm: int, shingle_size: int, id_prefix: str, desc_label: str):
    n_total = n_kept = n_dropped = total_words_kept = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(in_path, "r", encoding="utf-8") as f_in, \
         open(out_path, "w", encoding="utf-8") as f_out, \
         tqdm(desc=desc_label, unit="rec") as pbar:

        for line in f_in:
            line = line.strip()
            if not line:
                continue

            n_total += 1
            rec = json.loads(line)
            words = rec["text"].split()
            mh    = make_minhash(words, num_perm, shingle_size)
            key   = f"{id_prefix}_{n_total}"

            if lsh.query(mh):
                n_dropped += 1
            else:
                lsh.insert(key, mh)
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_kept           += 1
                total_words_kept += len(words)

            pbar.update(1)

    return n_total, n_kept, n_dropped, total_words_kept


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    print(f"\nInitializing MinHash LSH (threshold={THRESHOLD}, num_perm={NUM_PERM}, shingle={SHINGLE_SIZE})...")
    lsh = MinHashLSH(threshold=THRESHOLD, num_perm=NUM_PERM)

    print("\n[1/2] Processing Manual Dataset (Near-Dedup)...")
    m_total, m_kept, m_dropped, m_words = process_stream(
        MANUAL_INPUT, MANUAL_OUTPUT, lsh, NUM_PERM, SHINGLE_SIZE, "manual", "Manual Records"
    )

    print("\n[2/2] Processing Downloaded Dataset (Near-Dedup)...")
    d_total, d_kept, d_dropped, d_words = process_stream(
        DOWNLOADED_INPUT, DOWNLOADED_OUTPUT, lsh, NUM_PERM, SHINGLE_SIZE, "downloaded", "Downloaded Records"
    )

    total_docs  = m_kept + d_kept
    total_words = m_words + d_words

    print("\n" + "=" * 65)
    print(" === NEAR-DEDUPLICATION SUMMARY REPORT (TELUGU) ===")
    print("=" * 65)
    print(f" Manual    : {m_total:,} docs -> {m_kept:,} kept  (dropped {m_dropped:,})")
    print(f"             Words kept: {m_words:,}")
    print("-" * 65)
    print(f" Downloaded: {d_total:,} docs -> {d_kept:,} kept  (dropped {d_dropped:,})")
    print(f"             Words kept: {d_words:,}")
    print("=" * 65)
    print(f" TOTAL COMBINED CORPUS: {total_docs:,} docs | {total_words:,} words")
    if total_words > 0:
        manual_pct = m_words / total_words * 100
        print(f" MANUAL WORD SHARE: {manual_pct:.2f}%  (quota target: >= 20%)")
    print("=" * 65 + "\n")
