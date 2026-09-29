"""
src/dataset.py

Dataset loading and tokenization packing utilities for causal language modeling.
Reads JSONL splits (manual + downloaded) generated in Phase 1, tokenizes text using
SentencePiece BPE tokenizers, and packs tokens into sequence chunks of length max_seq_len + 1.

Tokenized ids are streamed to a disk-backed binary cache and served via np.memmap rather
than held in one giant Python list. At ~500M tokens, buffering token ids as native Python
ints (~28-37 bytes each, since ids outside CPython's small-int cache are never interned)
costs on the order of 20GB of RAM -- easily enough to OOM a Kaggle/Colab session. The memmap
cache keeps steady-state memory flat regardless of corpus size, and lets repeated runs
(e.g. resuming across Kaggle's 12-hour session limit) skip re-tokenization entirely.
"""

import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm


class SentencePieceTokenizer:
    """
    Wrapper around trained SentencePiece model file (.model).
    """
    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        if not self.sp.load(model_path):
            raise RuntimeError(f"Failed to load SentencePiece model from {model_path}")
            
    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list:
        tokens = self.sp.encode_as_ids(text)
        if add_bos:
            tokens = [self.sp.bos_id()] + tokens
        if add_eos:
            tokens = tokens + [self.sp.eos_id()]
        return tokens

    def decode(self, ids: list) -> str:
        return self.sp.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    @property
    def pad_id(self) -> int:
        return self.sp.pad_id()

    @property
    def unk_id(self) -> int:
        return self.sp.unk_id()

    @property
    def bos_id(self) -> int:
        return self.sp.bos_id()

    @property
    def eos_id(self) -> int:
        return self.sp.eos_id()


class TextDataset(Dataset):
    """
    Dataset that loads text documents from a JSONL file, tokenizes them, and returns
    input (x) / target (y) sequences of fixed length.

    Token ids are streamed to a disk-backed binary cache (int32) the first time a given
    (jsonl_path, tokenizer vocab, max_documents) combination is seen, then served via
    np.memmap on every call after that -- so re-running or resuming a script re-tokenizes
    nothing and never holds the full token stream in RAM at once.
    """
    def __init__(
        self,
        jsonl_path: Path,
        tokenizer: SentencePieceTokenizer,
        max_seq_len: int = 512,
        max_documents: int = None,
        cache_dir: Path = None,
        rebuild_cache: bool = False
    ):
        self.jsonl_path = Path(jsonl_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.chunk_size = max_seq_len + 1

        cache_dir = Path(cache_dir) if cache_dir is not None else self.jsonl_path.parent
        cache_dir.mkdir(parents=True, exist_ok=True)
        doc_tag = str(max_documents) if max_documents else "all"
        self.cache_path = cache_dir / f"{self.jsonl_path.stem}.v{tokenizer.vocab_size}.d{doc_tag}.ids.bin"

        if rebuild_cache or not self.cache_path.exists():
            self._build_token_cache(max_documents)

        # Memory-map the cached token ids: pages are loaded lazily by the OS on access,
        # so RAM usage stays flat no matter how large the corpus is.
        self.token_mmap = np.memmap(self.cache_path, dtype=np.int32, mode="r")
        self.num_tokens = int(self.token_mmap.shape[0])
        self.num_chunks = self.num_tokens // self.chunk_size

    def _build_token_cache(self, max_documents: int = None):
        """
        Tokenizes the JSONL file document-by-document, appending each document's ids
        straight to disk (as int32) instead of accumulating a giant in-memory list.
        """
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"Data split file not found: {self.jsonl_path}")

        tmp_path = self.cache_path.with_suffix(".tmp")
        doc_count = 0
        token_count = 0

        with open(self.jsonl_path, "r", encoding="utf-8") as f_in, open(tmp_path, "wb") as f_out:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue

                doc = json.loads(line)
                text = doc.get("text", "")
                if not text:
                    continue

                # Encode text with BOS and EOS tags, write straight to disk
                doc_tokens = self.tokenizer.encode(text, add_bos=True, add_eos=True)
                np.asarray(doc_tokens, dtype=np.int32).tofile(f_out)
                token_count += len(doc_tokens)

                doc_count += 1
                if max_documents and doc_count >= max_documents:
                    break

        tmp_path.rename(self.cache_path)
        print(
            f"[TextDataset] Cached {doc_count:,} docs / {token_count:,} tokens -> "
            f"{self.cache_path.name} ({self.cache_path.stat().st_size / 1e6:.1f} MB)"
        )

    def __len__(self) -> int:
        return self.num_chunks

    def __getitem__(self, idx: int):
        start = idx * self.chunk_size
        end = start + self.chunk_size
        # Copy the small slice out of the memmap and upcast to int64 for embedding lookup
        chunk = torch.from_numpy(self.token_mmap[start:end].astype(np.int64))
        x = chunk[:-1]  # input sequence (0 to seq_len - 1)
        y = chunk[1:]   # target sequence (1 to seq_len)
        return x, y


def create_dataloader(
    jsonl_path: Path,
    tokenizer: SentencePieceTokenizer,
    batch_size: int = 32,
    max_seq_len: int = 512,
    shuffle: bool = True,
    max_documents: int = None,
    num_workers: int = 0,
    cache_dir: Path = None,
    rebuild_cache: bool = False
) -> DataLoader:
    """
    Creates PyTorch DataLoader for causal language modeling training or evaluation.
    """
    dataset = TextDataset(
        jsonl_path=jsonl_path,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        max_documents=max_documents,
        cache_dir=cache_dir,
        rebuild_cache=rebuild_cache
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
