"""
NEPALI/train_nepali.py

Pretraining script for Model L (Nepali decoder-only Transformer LM).
Loads Nepali SentencePiece BPE tokenizer and training data split, instantiates ~25.3M GPTLanguageModel,
and executes pretraining with checkpoint saving.
"""

import sys
import json
from pathlib import Path

# Add project root to path -- this script lives at NEPALI/PHASE2/train_nepali.py, so
# parents[2] is the repo root (where src/ lives), and parents[1] is the NEPALI/ language dir.
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

import torch
from src.model import GPTLanguageModel
from src.dataset import SentencePieceTokenizer, create_dataloader
from src.trainer import Trainer


def main():
    nepali_dir = Path(__file__).resolve().parent
    config_path = nepali_dir / "configs" / "model_l_config.json"
    tokenizer_path = nepali_dir / "tokenizer" / "nepali_bpe.model"
    data_dir = nepali_dir / "data"
    output_dir = nepali_dir / "checkpoints"
    cache_dir = nepali_dir / "cache"

    # Load configuration
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    print(f"=== Starting Model L (Nepali) Pretraining ===")
    print(f"Config: {config}")

    # Load tokenizer
    tokenizer = SentencePieceTokenizer(str(tokenizer_path))
    print(f"Loaded Nepali tokenizer. Vocab size: {tokenizer.vocab_size:,}")

    # Load DataLoaders
    train_jsonl = data_dir / "train.jsonl"
    val_jsonl = data_dir / "val.jsonl"

    if not train_jsonl.exists():
        print(f"[Warning] Local {train_jsonl} not found. Please ensure dataset split is available or linked.")
        return

    train_loader = create_dataloader(
        jsonl_path=train_jsonl,
        tokenizer=tokenizer,
        batch_size=config.get("batch_size", 32),
        max_seq_len=config.get("max_seq_len", 512),
        shuffle=True,
        cache_dir=cache_dir
    )

    val_loader = create_dataloader(
        jsonl_path=val_jsonl,
        tokenizer=tokenizer,
        batch_size=config.get("batch_size", 32),
        max_seq_len=config.get("max_seq_len", 512),
        shuffle=False,
        cache_dir=cache_dir
    )

    # Instantiate Model
    model = GPTLanguageModel(
        vocab_size=config.get("vocab_size", 12000),
        d_model=config.get("d_model", 512),
        n_layer=config.get("n_layer", 6),
        n_head=config.get("n_head", 8),
        d_ff=config.get("d_ff", 2048),
        max_seq_len=config.get("max_seq_len", 512),
        dropout=config.get("dropout", 0.1),
        tie_weights=config.get("tie_weights", True)
    )

    params = model.count_parameters()
    print(f"Model Parameters: Total = {params['total_trainable']:,} | Embedding = {params['embeddings']:,} | Blocks = {params['transformer_blocks']:,}")

    # Empirical Causal Mask Test
    assert model.verify_causal_masking(), "Causal mask test failed!"
    print("Causal mask verification test PASSED!")

    # Instantiate Trainer and Run
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        config=config,
        output_dir=output_dir
    )

    resume_ckpt = output_dir / "checkpoint_latest.pt"
    trainer.train(resume_path=resume_ckpt if resume_ckpt.exists() else None)


if __name__ == "__main__":
    main()
