import os
import json
from datasets import load_dataset
from tqdm import tqdm

# DOWNLOADED data (pre-cleaned corpora), not manual -- same bucket as OSCAR.
# Runs locally: no Colab session limit, no separate monitor process needed -- this script
# downloads and shows live progress in the same loop.
#
# Tries Sangraha FIRST (same as your Telugu script). Nepali is a low-resource language
# in Sangraha though, so once Sangraha's subsets run dry, this automatically moves on to
# other pre-cleaned sources (IndicCorpV2, then CC-100) to close the gap toward the target.
# Everything is written with the same <<SRC:...>>/<<SUBSET:...>>/<<END>> markers, just with
# a different SRC tag per origin, so one preprocess_downloaded.py pass handles all of it.

OUT_DIR = os.path.abspath('nepali_dataset_output/sangraha_output')
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE      = os.path.join(OUT_DIR, 'raw_downloaded_nepali.txt')
PROGRESS_FILE = os.path.join(OUT_DIR, 'download_progress.json')

TARGET_TOKENS   = 400_000_000
TOKENS_PER_WORD = 1                                   # matches the ratio used elsewhere in your pipeline
TARGET_WORDS    = int(TARGET_TOKENS / TOKENS_PER_WORD)    # ~500M words

# Ordered list of sources to try, in priority order. Each has its own loader because the
# three datasets don't share a load_dataset() call signature. Sangraha's two subsets are
# tried first (verified before unverified); IndicCorpV2 and CC-100 are the fallback sources
# once Sangraha is exhausted and TARGET_WORDS still hasn't been reached.
SOURCES = [
    {
        "src_tag": "sangraha_downloaded",
        "subset": "verified/nep",
        "loader": lambda: load_dataset(
            "ai4bharat/sangraha", data_dir="verified/nep", split="train", streaming=True
        ),
    },
    {
        "src_tag": "sangraha_downloaded",
        "subset": "unverified/nep",
        "loader": lambda: load_dataset(
            "ai4bharat/sangraha", data_dir="unverified/nep", split="train", streaming=True
        ),
    },
    {
        "src_tag": "indiccorp_v2_downloaded",
        "subset": "npi_Deva",
        "loader": lambda: load_dataset(
            "ai4bharat/IndicCorpV2", "indiccorp_v2", split="npi_Deva", streaming=True
        ),
    },
    {
        "src_tag": "cc100_downloaded",
        "subset": "ne",
        "loader": lambda: load_dataset(
            "statmt/cc100", lang="ne", split="train", streaming=True
        ),
    },
]


def source_key(cfg):
    """Unique progress-tracking key per (src_tag, subset) pair -- src_tag alone isn't
    unique since Sangraha contributes two subsets."""
    return f"{cfg['src_tag']}::{cfg['subset']}"


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {source_key(s): 0 for s in SOURCES}


def save_progress(p):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(p, f)


progress = load_progress()

# Resume word count from what's actually on disk (source of truth), not just the progress file
current_words = 0
if os.path.exists(OUT_FILE):
    with open(OUT_FILE, 'r', encoding='utf-8') as f:
        current_words = len(f.read().split())
docs_count = 0

print(f"Output folder : {OUT_DIR}")
print(f"Already saved : {current_words:,} words (~{int(current_words*TOKENS_PER_WORD):,} tokens)")
print(f"Target        : {TARGET_WORDS:,} words (~{TARGET_TOKENS:,} tokens)")
print("Source order  : " + " -> ".join(f"{s['src_tag']}[{s['subset']}]" for s in SOURCES))
print("(Nepali likely won't hit the target even after all sources -- see the note printed")
print(" at the end; report the exact number you land on in your Phase 1 statistics.)\n")

pbar = tqdm(total=TARGET_WORDS, initial=current_words, unit=" words",
            desc="Nepali corpus", dynamic_ncols=True)

with open(OUT_FILE, "a", encoding="utf-8") as outfile:   # append -- never overwrites what's saved
    for cfg in SOURCES:
        if current_words >= TARGET_WORDS:
            break

        key = source_key(cfg)
        src_tag, subset = cfg["src_tag"], cfg["subset"]
        pbar.set_postfix_str(f"source={src_tag}[{subset}]")
        already_done = progress.get(key, 0)

        try:
            dataset = cfg["loader"]()

            for i, item in enumerate(dataset):
                if i < already_done:
                    continue   # already written in a previous run

                text = (item.get("text", "") or "").strip()
                if text:
                    outfile.write(f"<<SRC:{src_tag}>>\n<<SUBSET:{subset}>>\n{text}\n<<END>>\n\n")
                    wc = len(text.split())
                    current_words += wc
                    docs_count += 1
                    pbar.update(wc)

                progress[key] = i + 1

                if docs_count % 2000 == 0 and docs_count > 0:
                    outfile.flush()
                    save_progress(progress)
                    pbar.set_postfix_str(
                        f"source={src_tag}[{subset}] | ~{int(current_words*TOKENS_PER_WORD):,} tokens | "
                        f"{os.path.getsize(OUT_FILE)/1024/1024:.0f} MB on disk"
                    )

                if current_words >= TARGET_WORDS:
                    break
            else:
                # loop completed without hitting TARGET_WORDS -> this source is exhausted;
                # move on to the next one in SOURCES automatically
                tqdm.write(f"Source {src_tag}[{subset}] exhausted -- moving to next source.")

        except Exception as e:
            tqdm.write(f"Error loading {src_tag}[{subset}]: {e} (continuing to next source)")

    outfile.flush()
    save_progress(progress)

pbar.close()
print(f"\nDone. ~{current_words:,} words on disk ({docs_count:,} docs written this run).")
print(f"Estimated tokens: ~{int(current_words * TOKENS_PER_WORD):,}")
print(f"File: {OUT_FILE}")
if current_words < TARGET_WORDS:
    print(f"\nAll sources exhausted below target ({current_words:,} / {TARGET_WORDS:,} words).")
    print("This is expected for Nepali (allowed low-resource list). Report the exact")
    print("token count above in your Phase 1 corpus statistics, with a brief justification")
    print("of the shortfall (public availability is skewed toward higher-resource languages).")