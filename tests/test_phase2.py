"""
tests/test_phase2.py

Automated unit tests for Phase 2 implementation:
1. Exact model parameter count verification (~25.3M parameters).
2. Tensor shape correctness throughout forward pass.
3. Strict Causal Masking property (token t+1 modification must not affect position t).
4. Weight tying assertion (wte.weight is lm_head.weight).
5. Tokenizer integration with Phase 1 trained BPE models.
"""

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import torch
from src.model import GPTLanguageModel
from src.dataset import SentencePieceTokenizer


class TestPhase2Model(unittest.TestCase):
    def setUp(self):
        self.model = GPTLanguageModel(
            vocab_size=12000,
            d_model=512,
            n_layer=6,
            n_head=8,
            d_ff=2048,
            max_seq_len=512,
            dropout=0.1,
            tie_weights=True
        )

    def test_parameter_count(self):
        """
        Verify parameter count is around ~25.3M.
        """
        params = self.model.count_parameters()
        total = params["total_trainable"]
        print(f"\n[Test] Total Trainable Parameters: {total:,}")
        self.assertGreater(total, 24_000_000, "Parameter count is below target")
        self.assertLess(total, 26_000_000, "Parameter count exceeds ~25M budget")

    def test_weight_tying(self):
        """
        Verify that input embedding weights are tied to the output LM head.
        """
        self.assertTrue(
            self.model.lm_head.weight is self.model.embeddings.wte.weight,
            "Weight tying failed: lm_head and wte do not share the same memory tensor."
        )

    def test_forward_shape(self):
        """
        Verify output tensor shapes during forward pass.
        """
        B, T = 2, 32
        x = torch.randint(0, 12000, (B, T))
        y = torch.randint(0, 12000, (B, T))

        logits, loss = self.model(x, targets=y)
        self.assertEqual(logits.shape, (B, T, 12000), f"Unexpected logits shape: {logits.shape}")
        self.assertIsNotNone(loss, "Loss should not be None when targets are provided")

    def test_causal_masking(self):
        """
        Empirically verify causal mask: changing position t+1 must NOT change logits at position t.
        """
        is_causal = self.model.verify_causal_masking()
        self.assertTrue(is_causal, "Causal masking verification failed! Future tokens leak to earlier positions.")

    def test_telugu_tokenizer(self):
        """
        Verify Telugu SentencePiece tokenizer loads and encodes properly.
        """
        tok_path = ROOT_DIR / "TELUGU/PHASE2/tokenizer/telugu_bpe.model"
        self.assertTrue(tok_path.exists(), f"Telugu tokenizer not found at {tok_path}")
        tokenizer = SentencePieceTokenizer(str(tok_path))
        self.assertEqual(tokenizer.vocab_size, 12000)
        text = "తెలుగు సమాచారం"
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        self.assertIn("తెలుగు", decoded)

    def test_nepali_tokenizer(self):
        """
        Verify Nepali SentencePiece tokenizer loads and encodes properly.
        """
        tok_path = ROOT_DIR / "NEPALI/PHASE2/tokenizer/nepali_bpe.model"
        self.assertTrue(tok_path.exists(), f"Nepali tokenizer not found at {tok_path}")
        tokenizer = SentencePieceTokenizer(str(tok_path))
        self.assertEqual(tokenizer.vocab_size, 12000)
        text = "नेपाल एक देश हो"
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        self.assertIn("नेपाल", decoded)


if __name__ == "__main__":
    unittest.main()
