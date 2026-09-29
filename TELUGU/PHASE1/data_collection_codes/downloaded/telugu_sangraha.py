import os
import json
from datasets import load_dataset
from tqdm import tqdm

# DOWNLOADED data (AI4Bharat's own pre-cleaned corpus)

OUT_DIR = os.path.abspath('telugu_dataset_output/sangraha_output')
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE      = os.path.join(OUT_DIR, 'raw_sangraha_telugu.txt')
PROGRESS_FILE = os.path.join(OUT_DIR, 'sangraha_progress.json')

TARGET_TOKENS   = 500_000_000
TOKENS_PER_WORD = 1                                   
TARGET_WORDS    = int(TARGET_TOKENS / TOKENS_PER_WORD)    # ~294M words

SUBDIRS = ["verified/tel", "unverified/tel"]

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {d: 0 for d in SUBDIRS}   # items already consumed per subset, for resume

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
print(f"Target        : {TARGET_WORDS:,} words (~{TARGET_TOKENS:,} tokens)\n")

# Overall progress bar tracks words toward target, persists across subsets, resumes from current_words
pbar = tqdm(total=TARGET_WORDS, initial=current_words, unit=" words",
            desc="Sangraha Telugu", dynamic_ncols=True)

with open(OUT_FILE, "a", encoding="utf-8") as outfile:   # append -- never overwrites what's saved
    for data_dir in SUBDIRS:
        if current_words >= TARGET_WORDS:
            break

        pbar.set_postfix_str(f"subset={data_dir}")
        already_done = progress.get(data_dir, 0)

        try:
            dataset = load_dataset(
                "ai4bharat/sangraha",
                data_dir=data_dir,
                split="train",
                streaming=True
            )

            for i, item in enumerate(dataset):
                if i < already_done:
                    continue   # already written in a previous run

                text = item.get("text", "").strip()
                if text:
                    outfile.write(f"<<SRC:sangraha_downloaded>>\n<<SUBSET:{data_dir}>>\n{text}\n<<END>>\n\n")
                    wc = len(text.split())
                    current_words += wc
                    docs_count += 1
                    pbar.update(wc)

                progress[data_dir] = i + 1

                if docs_count % 2000 == 0 and docs_count > 0:
                    outfile.flush()
                    save_progress(progress)
                    pbar.set_postfix_str(
                        f"subset={data_dir} | ~{int(current_words*TOKENS_PER_WORD):,} tokens | "
                        f"{os.path.getsize(OUT_FILE)/1024/1024:.0f} MB on disk"
                    )

                if current_words >= TARGET_WORDS:
                    break

        except Exception as e:
            tqdm.write(f"Error loading {data_dir}: {e} (continuing to next subset)")

    outfile.flush()
    save_progress(progress)

pbar.close()
print(f"\nDone. ~{current_words:,} words on disk ({docs_count:,} docs written this run).")
print(f"Estimated tokens: ~{int(current_words * TOKENS_PER_WORD):,}")
print(f"File: {OUT_FILE}")