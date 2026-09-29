"""
ablation/train_ablation.py

Pretrains Model L (Nepali) with positional embeddings removed entirely --
the assignment's Bonus-section ablation (configs/model_l_no_pos_config.json).

Reuses the exact same data, tokenizer, and training hyperparameters as the
standard Model L run in NEPALI/PHASE2/train_nepali.py -- only the architecture
flag differs -- so any difference in the eval results is attributable to the
ablated component and nothing else.

Checkpointing works like the standard training script: checkpoint_latest.pt is
written every save_interval steps and used to auto-resume if this script is
re-run after an interruption.
"""

import os
import sys
import json
import argparse
import importlib.util
from pathlib import Path

ABLATION_DIR = Path(__file__).resolve().parent
# dataset.py/trainer.py are unmodified, so they are not duplicated here -- imported
# from the main project's src/ package (this folder lives inside the full repo, as a
# sibling of NEPALI/, TELUGU/, and src/).
ROOT_DIR = ABLATION_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

import torch
from src.dataset import SentencePieceTokenizer, create_dataloader
from src.trainer import Trainer

# model.py is not shared -- ablation/src/model.py carries the use_positional_embeddings
# flag the main src/model.py doesn't have, so it is loaded directly from its own file
# instead of through the "src" package above (which would resolve to the main model.py).
_model_spec = importlib.util.spec_from_file_location("ablation_model", ABLATION_DIR / "src" / "model.py")
_ablation_model = importlib.util.module_from_spec(_model_spec)
_model_spec.loader.exec_module(_ablation_model)
GPTLanguageModel = _ablation_model.GPTLanguageModel

VARIANT_CONFIGS = {
    "no_pos": "model_l_no_pos_config.json",
}


def find_data_root() -> Path:
    """
    Locates NEPALI/PHASE1/outputs/ (train/val/test.jsonl, and the tokenizer/ dir next
    to them). Checks, in order: a bundled copy inside this folder (for a standalone
    Kaggle upload); the main repo, one level up (the normal case, since this folder is
    a sibling of NEPALI/, TELUGU/, and src/); or a dataset mounted under /kaggle/input.
    """
    bundled = ABLATION_DIR / "NEPALI" / "PHASE1" / "outputs"
    if (bundled / "data_splits" / "train.jsonl").exists():
        return bundled

    local = ABLATION_DIR.parent / "NEPALI" / "PHASE1" / "outputs"
    if (local / "data_splits" / "train.jsonl").exists():
        return local

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for root, dirs, _ in os.walk(kaggle_input):
            if "NEPALI" in dirs:
                candidate = Path(root) / "NEPALI" / "PHASE1" / "outputs"
                if (candidate / "data_splits" / "train.jsonl").exists():
                    return candidate

    raise FileNotFoundError(
        "Could not find NEPALI/PHASE1/outputs/data_splits/train.jsonl locally or under "
        "/kaggle/input."
    )


def main():
    parser = argparse.ArgumentParser(description="Train the no-positional-embeddings ablation of Model L (Nepali).")
    parser.add_argument("--variant", choices=list(VARIANT_CONFIGS.keys()), default="no_pos")
    args = parser.parse_args()

    config_path = ABLATION_DIR / "configs" / VARIANT_CONFIGS[args.variant]
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    print(f"=== Starting Ablation Pretraining: variant='{args.variant}' ===", flush=True)
    print(f"Ablation: {config.get('ablation_description')}", flush=True)
    print(f"Config: {config}", flush=True)

    data_root = find_data_root()
    # tokenizer/ is not duplicated in ablation/ -- it lives alongside the data splits,
    # at NEPALI/PHASE1/outputs/tokenizer/ (same outputs/ dir find_data_root() found).
    tokenizer_path = data_root / "tokenizer" / "nepali_bpe.model"
    train_jsonl = data_root / "data_splits" / "train.jsonl"
    val_jsonl = data_root / "data_splits" / "val.jsonl"

    # The tokenized-id cache keys only on (jsonl path, vocab size, max_documents), not
    # on model architecture, so re-running this reuses the cache and skips re-tokenizing
    # ~500M tokens from scratch.
    cache_dir = ABLATION_DIR / "outputs" / "_shared_cache"

    output_dir = ABLATION_DIR / "outputs" / args.variant / "checkpoints"

    tokenizer = SentencePieceTokenizer(str(tokenizer_path))
    print(f"Loaded Nepali tokenizer. Vocab size: {tokenizer.vocab_size:,}", flush=True)

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

    model = GPTLanguageModel(
        vocab_size=config.get("vocab_size", 12000),
        d_model=config.get("d_model", 512),
        n_layer=config.get("n_layer", 6),
        n_head=config.get("n_head", 8),
        d_ff=config.get("d_ff", 2048),
        max_seq_len=config.get("max_seq_len", 512),
        dropout=config.get("dropout", 0.1),
        tie_weights=config.get("tie_weights", True),
        use_positional_embeddings=config.get("use_positional_embeddings", True),
        norm_type=config.get("norm_type", "layernorm")
    )

    params = model.count_parameters()
    print(f"Model Parameters: Total = {params['total_trainable']:,} | Embedding = {params['embeddings']:,} | "
          f"Blocks = {params['transformer_blocks']:,} | use_positional_embeddings = {params['use_positional_embeddings']} | "
          f"norm_type = {params['norm_type']}", flush=True)

    # Causal masking is enforced entirely by the attention mask, not by position info,
    # so this check must (and does) still pass under the ablation.
    assert model.verify_causal_masking(), "Causal mask test failed!"
    print("Causal mask verification test PASSED!", flush=True)

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
