"""
2_exact_deduplication_nepali.py

Exact-match deduplication using SHA-256 hashes of whitespace-normalized text.
Runs BOTH within each source (manual-vs-manual, sangraha-vs-sangraha)
and across sources (manual-vs-sangraha).

Priority Rule:
Manual documents take priority over Sangraha documents to safeguard your
manual-token quota.
"""

import hashlib
import json
import re
from pathlib import Path
from tqdm import tqdm

# ==============================================================================
# PATHS
# ==============================================================================
BASE = Path("/Users/erva_rishitha/Documents/LMA project1/NEPALI")

MANUAL_INPUT      = BASE / "outputs/preprocessed/manual_clean.jsonl"
DOWNLOADED_INPUT  = BASE / "outputs/preprocessed/downloaded_clean.jsonl"
MANUAL_OUTPUT     = BASE / "outputs/exact_dedup/manual.jsonl"
DOWNLOADED_OUTPUT = BASE / "outputs/exact_dedup/downloaded.jsonl"

# ==============================================================================
# HELPER
# ==============================================================================
_WS_RE = re.compile(r"\s+")


def hash_key(text: str) -> str:
    key = _WS_RE.sub(" ", text).strip()
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def process_manual(manual_path: Path, out_manual_path: Path, seen_hashes: set):
    n_total, n_kept, n_intra_dupes = 0, 0, 0
    out_manual_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/2] Processing Manual Exact Dedup...")
    with open(manual_path, "r", encoding="utf-8") as f_in, \
         open(out_manual_path, "w", encoding="utf-8") as f_out, \
         tqdm(desc="Manual Records", unit="rec") as pbar:

        for line in f_in:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            rec = json.loads(line)
            h = hash_key(rec["text"])

            if h in seen_hashes:
                n_intra_dupes += 1
            else:
                seen_hashes.add(h)
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_kept += 1
            pbar.update(1)

    print(f"  -> Kept: {n_kept:,} | Dropped intra-source dupes: {n_intra_dupes:,}")
    return n_total, n_kept, n_intra_dupes


def process_downloaded(downloaded_path: Path, out_downloaded_path: Path, seen_hashes: set):
    n_total, n_kept, n_intra_dupes, n_cross_dupes = 0, 0, 0, 0
    # Processes downloaded (Sangraha) dataset, deduplicating against itself and manual.
    downloaded_seen = set()
    out_downloaded_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[2/2] Processing Downloaded Exact Dedup...")
    with open(downloaded_path, "r", encoding="utf-8") as f_in, \
         open(out_downloaded_path, "w", encoding="utf-8") as f_out, \
         tqdm(desc="Downloaded Records", unit="rec") as pbar:

        for line in f_in:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            rec = json.loads(line)
            h = hash_key(rec["text"])

            if h in downloaded_seen:
                n_intra_dupes += 1
            elif h in seen_hashes:
                n_cross_dupes += 1
            else:
                downloaded_seen.add(h)
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_kept += 1
            pbar.update(1)

    print(f"  -> Kept: {n_kept:,} | Intra-dupes: {n_intra_dupes:,} | Cross-dupes vs manual: {n_cross_dupes:,}")
    return n_total, n_kept, n_intra_dupes, n_cross_dupes


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    seen_hashes = set()

    m_total, m_kept, m_intra = process_manual(MANUAL_INPUT, MANUAL_OUTPUT, seen_hashes)
    d_total, d_kept, d_intra, d_cross = process_downloaded(DOWNLOADED_INPUT, DOWNLOADED_OUTPUT, seen_hashes)

    print("\n" + "=" * 55)
    print(" === EXACT DEDUPLICATION SUMMARY REPORT (NEPALI) ===")
    print("=" * 55)
    print(f" Manual    : {m_total:,} total -> {m_kept:,} kept  (dropped {m_intra:,} intra-dupes)")
    print(f" Downloaded: {d_total:,} total -> {d_kept:,} kept  (dropped {d_intra:,} intra, {d_cross:,} cross-dupes)")
    print("=" * 55 + "\n")
