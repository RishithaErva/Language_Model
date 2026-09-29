"""
0_trim_raw_downloaded.py

Trims raw_sangraha_telugu.txt directly on disk to a target 290M word budget,
saving the result safely to raw_sangraha_telugu_trimmed.txt.
"""

from pathlib import Path

BASE = Path("/Users/erva_rishitha/Documents/LMA project1/TELUGU")

RAW_DOWNLOADED_PATH = BASE / "raw_data/downloaded/raw_sangraha_telugu.txt"
TRIMMED_OUTPUT_PATH = BASE / "raw_data/downloaded/raw_sangraha_telugu_trimmed.txt"

TARGET_WORD_COUNT = 200_000_000 

def trim_raw_file():
    if not RAW_DOWNLOADED_PATH.exists():
        print(f"Error: Raw file not found at {RAW_DOWNLOADED_PATH}")
        return

    print(f"Trimming {RAW_DOWNLOADED_PATH.name} down to {TARGET_WORD_COUNT:,} words...")
    
    current_block = []
    current_words = 0
    total_words_kept = 0
    docs_kept = 0

    # Ensure target output directory exists
    TRIMMED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(RAW_DOWNLOADED_PATH, "r", encoding="utf-8", errors="ignore") as f_in, \
         open(TRIMMED_OUTPUT_PATH, "w", encoding="utf-8") as f_out:

        for line in f_in:
            current_block.append(line)
            
            # Count words on text content lines
            if not line.strip().startswith("<<"):
                current_words += len(line.split())

            # Complete block reached
            if line.strip() == "<<END>>":
                f_out.writelines(current_block)
                total_words_kept += current_words
                docs_kept += 1

                current_block = []
                current_words = 0

                if total_words_kept >= TARGET_WORD_COUNT:
                    print(f"\nReached word threshold! Stopped at {total_words_kept:,} words.")
                    break

    print(f"\nSaved trimmed file to: {TRIMMED_OUTPUT_PATH}")
    print(f"Retained Documents:   {docs_kept:,}")
    print(f"Retained Words:       {total_words_kept:,}")

if __name__ == "__main__":
    trim_raw_file()
