"""
1_preprocess_fulldata_nepali.py

Unified streaming Phase 1 preprocessor with tqdm progress tracking.
Converts tagged raw dumps (both Manual and Downloaded/Sangraha) into clean .jsonl files.

Data Sources:
  Manual (raw_data/manual/):
    - nepali_web_crawl.txt      : news, gov, literature, wiki API scrapes
    - nepali_textbooks_ocr.txt  : OCR of Nepali textbooks

  Downloaded (raw_data/downloaded/):
    - raw_downloaded_nepali.txt : Sangraha/AI4Bharat downloaded corpus

SRC tag -> source_type mapping:
  news / news_extra            -> news_website
  news_gov / gov               -> govt_portal
  literature                   -> literature_website
  API_Nepali_Wikipedia         -> wikipedia_api
  API_Nepali_Wiktionary        -> wiktionary_api
  API_Nepali_Wikibooks         -> wikibooks_api
  archive_downloaded_ocr       -> archive_ocr
  ocr_manual                   -> ocr_textbook
  sangraha / sangraha_downloaded -> sangraha
"""

import re
import json
import html
import unicodedata
from pathlib import Path
from tqdm import tqdm

# ==============================================================================
# PATHS 
# ==============================================================================
BASE = Path("/Users/erva_rishitha/Documents/LMA project1/NEPALI")

MANUAL_INPUT_FILES = {
    str(BASE / "raw_data/manual/nepali_web_crawl.txt"):     "web_scrape",
    str(BASE / "raw_data/manual/nepali_textbooks_ocr.txt"): "ocr_textbook",
}

DOWNLOADED_INPUT_FILES = {
    str(BASE / "raw_data/downloaded/raw_downloaded_nepali_trimmed.txt"): "sangraha",
}

MANUAL_JSONL_OUT     = BASE / "outputs/preprocessed/manual_clean.jsonl"
DOWNLOADED_JSONL_OUT = BASE / "outputs/preprocessed/downloaded_clean.jsonl"

# ==============================================================================
# REGEX & ARTIFACT PATTERNS
# ==============================================================================
NEPALI_RE    = re.compile(r'[\u0900-\u097F]')
LATIN_RE     = re.compile(r'[A-Za-z]')
BLOCK_TAG_RE = re.compile(r'^\s*<<(?P<key>[A-Z_a-z0-9]+):(?P<value>.*?)>>\s*$')
END_TAG_RE   = re.compile(r'^\s*<<END>>\s*$')
URL_RE       = re.compile(r'https?://\S+')
FILEPATH_RE  = re.compile(r'(?:[A-Za-z]:)?(?:[/\\][\w .\-+%]+){2,}\.\w{2,5}')

WIKI_MARKUP_RES = [
    re.compile(r'(?:\|[A-Za-z0-9]*\}\}\s*)+'),
    re.compile(r'\b(?:center|left|right|thumb|border)\|[^\s]*'),
    re.compile(r'\[\[[^\]]*\]\]'),
    re.compile(r'\{\{[^}]*\}\}'),
    re.compile(r'\|{2,}'),
]

LOG_ARTIFACT_SUBSTRINGS = [
    'Downloading:', 'Download completed', 'Download done', 'Download failed',
    'Temp    :', 'Temp:', '\U0001F5D1\uFE0F', '\u2705', '\u274C',
    'IncompleteRead', 'Parse error', 'articles |', 'words |', 'new URLs',
    'CORPUS SUMMARY', 'Estimated tokens', 'Output file', 'Progress toward',
    'Retry ', 'processed ->', 'running totals', 'Synced to',
]

STRAY_LATIN_RE    = re.compile(r'(?<![A-Za-z])[A-Za-z]{1,2}(?![A-Za-z])')
MULTI_SPACE_RE    = re.compile(r'[ \t\u00a0]+')
ZERO_WIDTH_RE     = re.compile(r'[\u200b\ufeff\u00ad]')
DEVANAGARI_DIGITS = str.maketrans('0123456789', '0123456789')  # ASCII passthrough
_DEV_DIGIT_TRANS  = str.maketrans('\u0966\u0967\u0968\u0969\u096a\u096b\u096c\u096d\u096e\u096f',
                                   '0123456789')

LINE_FILTERS = {
    'web_scrape':        dict(min_chars=15, min_te_ratio=0.45),
    'news_website':      dict(min_chars=15, min_te_ratio=0.45),
    'govt_portal':       dict(min_chars=15, min_te_ratio=0.45),
    'literature_website':dict(min_chars=15, min_te_ratio=0.45),
    'wikipedia_api':     dict(min_chars=15, min_te_ratio=0.45),
    'wiktionary_api':    dict(min_chars=15, min_te_ratio=0.45),
    'wikibooks_api':     dict(min_chars=15, min_te_ratio=0.45),
    'archive':           dict(min_chars=20, min_te_ratio=0.60),
    'archive_ocr':       dict(min_chars=20, min_te_ratio=0.60),
    'ocr_textbook':      dict(min_chars=20, min_te_ratio=0.75),
    'sangraha':          dict(min_chars=15, min_te_ratio=0.45),
}

MIN_DOC_WORDS = 30

# Maps the raw <<SRC:...>> tag value to a human-readable source_type label
SRC_TAG_TO_SOURCE_TYPE = {
    'news':                   'news_website',
    'news_gov':               'govt_portal',
    'news_extra':             'news_website',
    'gov':                    'govt_portal',
    'literature':             'literature_website',
    'API_Nepali_Wikipedia':   'wikipedia_api',
    'API_Nepali_Wiktionary':  'wiktionary_api',
    'API_Nepali_Wikibooks':   'wikibooks_api',
    'API_Nepali_Wikisource':  'wikisource_api',
    'archive_downloaded_ocr': 'archive_ocr',
    'ocr_manual':             'ocr_textbook',
    'sangraha_downloaded':    'sangraha',
    'sangraha':               'sangraha',
}

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def ne_ratio(text: str) -> float:
    dev   = len(NEPALI_RE.findall(text))
    la    = len(LATIN_RE.findall(text))
    total = dev + la
    return dev / total if total else 0.0


def strip_artifacts(line: str) -> str:
    if any(s in line for s in LOG_ARTIFACT_SUBSTRINGS):
        return ''
    line = URL_RE.sub(' ', line)
    line = FILEPATH_RE.sub(' ', line)
    for pat in WIKI_MARKUP_RES:
        line = pat.sub(' ', line)
    return line


def clean_line(line: str, rules: dict, strip_stray_latin: bool = False):
    line = html.unescape(line)
    line = unicodedata.normalize('NFC', line)
    line = line.translate(_DEV_DIGIT_TRANS)
    line = ZERO_WIDTH_RE.sub('', line)
    line = strip_artifacts(line)
    if strip_stray_latin:
        line = STRAY_LATIN_RE.sub(' ', line)
    line = MULTI_SPACE_RE.sub(' ', line).strip()
    if len(line) < rules['min_chars']:
        return None
    if ne_ratio(line) < rules['min_te_ratio']:
        return None
    return line


def dedup_consecutive(lines: list) -> list:
    out = []
    for l in lines:
        if out and out[-1] == l:
            continue
        out.append(l)
    return out


def flush_doc(meta, body_lines, base_source_type, dataset_tier, inp_name, out_f, counters):
    raw_src     = meta.get('src', '')
    source_type = SRC_TAG_TO_SOURCE_TYPE.get(raw_src, base_source_type)
    rules       = LINE_FILTERS.get(source_type, LINE_FILTERS['sangraha'])
    strip_latin = source_type in ('ocr_textbook', 'archive_ocr')

    kept = [clean_line(l, rules, strip_latin) for l in body_lines]
    kept = [l for l in kept if l]
    kept = dedup_consecutive(kept)
    text = '\n'.join(kept).strip()
    wc   = len(text.split())

    if wc >= MIN_DOC_WORDS:
        rec = {
            'text':        text,
            'source_tier': dataset_tier,
            'source_type': source_type,
            'source_file': inp_name,
            'src':         raw_src,
            'domain':      meta.get('domain'),
            'url':         meta.get('url'),
            'subset':      meta.get('subset'),
            'word_count':  wc,
        }
        out_f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        counters['docs']  += 1
        counters['words'] += wc


# ==============================================================================
# STREAMING PROCESSOR
# ==============================================================================

def process_dataset(input_files_map: dict, output_jsonl: Path, dataset_tier: str):
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    counters = {'docs': 0, 'words': 0}

    print(f'\n{"="*50}')
    print(f'Processing [{dataset_tier.upper()}] Corpus -> JSONL')
    print(f'Output: {output_jsonl}')
    print(f'{"="*50}')

    with open(output_jsonl, 'w', encoding='utf-8') as out_f:
        for file_path_str, base_source_type in input_files_map.items():
            inp_path = Path(file_path_str)
            if not inp_path.exists():
                print(f'  (skipping missing: {inp_path})')
                continue

            file_size  = inp_path.stat().st_size
            meta       = {}
            body_lines = []

            with open(inp_path, 'r', encoding='utf-8', errors='ignore') as f_in, \
                 tqdm(total=file_size, unit='B', unit_scale=True,
                      desc=f' {inp_path.name[:28]:<28}') as pbar:

                for raw_line in f_in:
                    pbar.update(len(raw_line.encode('utf-8', errors='ignore')))

                    if END_TAG_RE.match(raw_line):
                        flush_doc(meta, body_lines, base_source_type,
                                  dataset_tier, inp_path.name, out_f, counters)
                        pbar.set_postfix(docs=f"{counters['docs']:,}")
                        meta       = {}
                        body_lines = []
                        continue

                    tag_m = BLOCK_TAG_RE.match(raw_line)
                    if tag_m:
                        meta[tag_m.group('key').lower()] = tag_m.group('value').strip()
                        continue

                    body_lines.append(raw_line)

                # flush trailing doc (no <<END>>)
                if body_lines:
                    flush_doc(meta, body_lines, base_source_type,
                              dataset_tier, inp_path.name, out_f, counters)

    print(f"Finished [{dataset_tier}]: {counters['docs']:,} docs | {counters['words']:,} words\n")


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == '__main__':
    process_dataset(MANUAL_INPUT_FILES,     MANUAL_JSONL_OUT,     dataset_tier='manual')
    process_dataset(DOWNLOADED_INPUT_FILES, DOWNLOADED_JSONL_OUT, dataset_tier='sangraha')
