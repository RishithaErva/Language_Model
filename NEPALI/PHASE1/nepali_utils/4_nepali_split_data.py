"""
4_split_data_nepali.py

Splits manual and sangraha JSONL files independently (99% train, 0.5% val, 0.5% test)
then merges them into combined split files, preserving source proportions.
"""

import json
import random
from pathlib import Path

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE = Path("/Users/erva_rishitha/Documents/LMA project1/NEPALI")

MANUAL_INPUT_PATH     = BASE / "outputs/near_dedup/manual.jsonl"
DOWNLOADED_INPUT_PATH = BASE / "outputs/near_dedup/downloaded.jsonl"
OUTPUT_DIRECTORY    = BASE / "outputs/data_splits"

VAL_FRACTION  = 0.005
TEST_FRACTION = 0.005
RANDOM_SEED   = 42

# ==============================================================================
# HELPERS
# ==============================================================================

def load_jsonl(path: Path, default_source_tag: str):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                rec["source"] = default_source_tag
                records.append(rec)
    return records


def split_list(records, val_frac, test_frac):
    n       = len(records)
    n_val   = int(n * val_frac)
    n_test  = int(n * test_frac)
    n_train = n - n_val - n_test
    return records[:n_train], records[n_train:n_train + n_val], records[n_train + n_val:]


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    random.seed(RANDOM_SEED)
    out_dir = OUTPUT_DIRECTORY
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading manual data from: {MANUAL_INPUT_PATH}")
    manual_recs   = load_jsonl(MANUAL_INPUT_PATH,   "manual")

    print(f"Loading downloaded data from: {DOWNLOADED_INPUT_PATH}")
    downloaded_recs = load_jsonl(DOWNLOADED_INPUT_PATH, "downloaded")

    random.shuffle(manual_recs)
    random.shuffle(downloaded_recs)

    man_train, man_val, man_test = split_list(manual_recs,     VAL_FRACTION, TEST_FRACTION)
    dl_train,  dl_val,  dl_test  = split_list(downloaded_recs, VAL_FRACTION, TEST_FRACTION)

    train_recs = man_train + dl_train
    val_recs   = man_val   + dl_val
    test_recs  = man_test  + dl_test

    random.shuffle(train_recs)
    random.shuffle(val_recs)
    random.shuffle(test_recs)

    splits = {"train": train_recs, "val": val_recs, "test": test_recs}
    stats  = {}

    for split_name, recs in splits.items():
        out_path = out_dir / f"{split_name}.jsonl"
        manual_words = downloaded_words = 0

        with out_path.open("w", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                wc = len(rec.get("text", "").split())
                if rec.get("source") == "manual":
                    manual_words      += wc
                else:
                    downloaded_words  += wc

        total_words = manual_words + downloaded_words
        manual_pct  = 100.0 * manual_words / total_words if total_words else 0.0
        stats[split_name] = {
            "n_docs":           len(recs),
            "manual_words":     manual_words,
            "downloaded_words": downloaded_words,
            "total_words":      total_words,
            "manual_pct":       round(manual_pct, 2),
        }

    stats_path = out_dir / "split_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n=== Split report (NEPALI) ===")
    for split_name, s in stats.items():
        print(f"  {split_name:>5}: {s['n_docs']:>8,} docs | {s['total_words']:>12,} words | manual={s['manual_pct']}%")
    print(f"\nFull stats -> {stats_path}")


if __name__ == "__main__":
    main()
