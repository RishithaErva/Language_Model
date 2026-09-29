"""
src/metrics.py

Evaluation suite for decoder-only language models:
- Intrinsic LM metrics: Loss, Perplexity (PPL), Bits-Per-Byte (BPB).
- Text generation decoding (Greedy, Temperature sampling).
- Generation quality metrics: BLEU-4, chrF++, ROUGE-L.
- Diversity & Fluency diagnostics: Repetition rate (4-gram repeats), Distinct-1, Distinct-2.
"""

import json
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from typing import List, Dict


def extract_test_prefix_reference_pairs(
    jsonl_path: Path,
    tokenizer,
    num_samples: int = 20,
    prefix_len: int = 20,
    continuation_len: int = 40
) -> List[Dict[str, str]]:
    """
    Extracts (prefix_prompt, reference_continuation) pairs from held-out test.jsonl file.
    Splits each document's token sequence into a prefix prompt and reference target text.
    """
    jsonl_path = Path(jsonl_path)
    pairs = []
    
    if jsonl_path.exists():
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if len(pairs) >= num_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                except Exception:
                    continue
                text = doc.get("text", "")
                if not text:
                    continue
                tokens = tokenizer.encode(text, add_bos=False, add_eos=False)
                
                if len(tokens) >= (prefix_len + continuation_len):
                    prefix_ids = tokens[:prefix_len]
                    ref_ids = tokens[prefix_len : prefix_len + continuation_len]
                    
                    prefix_str = tokenizer.decode(prefix_ids)
                    ref_str = tokenizer.decode(ref_ids)
                    
                    pairs.append({
                        "prompt": prefix_str,
                        "reference": ref_str
                    })
    return pairs


def compute_perplexity_and_bpb(model: torch.nn.Module, val_dataloader: torch.utils.data.DataLoader, tokenizer, device: str = "cuda") -> Dict[str, float]:
    """
    Computes validation cross-entropy loss, Perplexity (PPL), and Bits-Per-Byte (BPB).
    BPB normalizes loss by total UTF-8 bytes to enable fair cross-lingual comparison.
    """
    model.eval()
    total_loss_nats = 0.0
    total_tokens = 0
    total_utf8_bytes = 0

    with torch.no_grad():
        for x, y in val_dataloader:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, targets=y)
            
            # Loss is average per-token cross entropy in nats
            batch_tokens = (y != tokenizer.pad_id).sum().item()
            total_loss_nats += loss.item() * batch_tokens
            total_tokens += batch_tokens

            # Count total UTF-8 bytes for targets in batch
            for seq in y.cpu().tolist():
                clean_ids = [tok for tok in seq if tok not in (tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id)]
                text = tokenizer.decode(clean_ids)
                total_utf8_bytes += len(text.encode("utf-8"))

    avg_loss_nats = total_loss_nats / max(1, total_tokens)
    perplexity = math.exp(avg_loss_nats)
    
    # BPB = total_nats / (ln(2) * total_bytes)
    bpb = (total_loss_nats) / (math.log(2) * max(1, total_utf8_bytes))

    return {
        "val_loss": avg_loss_nats,
        "perplexity": perplexity,
        "bpb": bpb,
        "total_tokens": total_tokens,
        "total_utf8_bytes": total_utf8_bytes
    }


@torch.no_grad()
def generate_text(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = None,
    device: str = "cuda"
) -> str:
    """
    Generates text continuation from a given prompt using greedy decoding or temperature sampling.
    """
    model.eval()
    prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        # Crop context to model's maximum sequence length
        idx_cond = idx[:, -model.max_seq_len:]
        logits, _ = model(idx_cond)
        
        # Focus only on last time step
        logits = logits[:, -1, :]
        
        if temperature == 0.0:
            # Greedy decoding
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

        # Stop if EOS token generated
        if idx_next.item() == tokenizer.eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)

    generated_ids = idx[0].cpu().tolist()
    return tokenizer.decode(generated_ids)


def compute_diversity_metrics(texts: List[str]) -> Dict[str, float]:
    """
    Computes Repetition Rate (4-gram repetitions), Distinct-1, and Distinct-2 metrics.
    """
    unigrams = set()
    bigrams = set()
    total_unigrams = 0
    total_bigrams = 0
    
    fourgrams = []
    
    for text in texts:
        tokens = text.split()
        if not tokens:
            continue
            
        total_unigrams += len(tokens)
        unigrams.update(tokens)
        
        for i in range(len(tokens) - 1):
            total_bigrams += 1
            bigrams.add((tokens[i], tokens[i+1]))
            
        for i in range(len(tokens) - 3):
            fourgrams.append(tuple(tokens[i:i+4]))

    distinct_1 = len(unigrams) / max(1, total_unigrams)
    distinct_2 = len(bigrams) / max(1, total_bigrams)
    
    # Repetition rate: 1 - (unique 4-grams / total 4-grams)
    if fourgrams:
        repetition_rate = 1.0 - (len(set(fourgrams)) / len(fourgrams))
    else:
        repetition_rate = 0.0

    return {
        "distinct_1": distinct_1,
        "distinct_2": distinct_2,
        "repetition_rate_4gram": repetition_rate
    }


def compute_generation_metrics(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """
    Computes generation quality metrics: BLEU-4, chrF++, and ROUGE-L.
    Uses sacrebleu and rouge-score if installed; falls back to pure-Python implementations otherwise.
    """
    bleu_score = 0.0
    chrf_score = 0.0
    rouge_l_score = 0.0

    # Try sacrebleu and rouge_score
    try:
        import sacrebleu
        from rouge_score import rouge_scorer

        # BLEU-4 (corpus level)
        bleu = sacrebleu.corpus_bleu(hypotheses, [references])
        bleu_score = bleu.score

        # chrF++
        chrf = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2)
        chrf_score = chrf.score

        # ROUGE-L
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
        rouge_scores = [scorer.score(ref, hyp)['rougeL'].fmeasure for ref, hyp in zip(references, hypotheses)]
        rouge_l_score = (sum(rouge_scores) / max(1, len(rouge_scores))) * 100.0

    except ImportError:
        # Fallback pure-Python calculation
        total_bleu, total_chrf, total_rouge = 0.0, 0.0, 0.0
        n_samples = len(hypotheses)

        for ref, hyp in zip(references, hypotheses):
            ref_tokens = ref.split()
            hyp_tokens = hyp.split()

            # Word-level LCS for ROUGE-L
            m, n = len(hyp_tokens), len(ref_tokens)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(m):
                for j in range(n):
                    if hyp_tokens[i] == ref_tokens[j]:
                        dp[i+1][j+1] = dp[i][j] + 1
                    else:
                        dp[i+1][j+1] = max(dp[i+1][j], dp[i][j+1])
            lcs = dp[m][n]
            r_lcs = lcs / max(1, n)
            p_lcs = lcs / max(1, m)
            beta = 1.2
            rouge_f = ((1 + beta**2) * r_lcs * p_lcs) / (beta**2 * p_lcs + r_lcs + 1e-8)
            total_rouge += rouge_f * 100.0

            # Character n-gram overlap for chrF
            ref_chars = list(ref)
            hyp_chars = list(hyp)
            matched_chars = sum(min(ref_chars.count(c), hyp_chars.count(c)) for c in set(hyp_chars))
            p_char = matched_chars / max(1, len(hyp_chars))
            r_char = matched_chars / max(1, len(ref_chars))
            chrf_f = (5 * p_char * r_char) / (4 * p_char + r_char + 1e-8)
            total_chrf += chrf_f * 100.0

            # Simple BLEU-4 approximation
            matches = sum(1 for h in hyp_tokens if h in ref_tokens)
            prec = matches / max(1, len(hyp_tokens))
            bp = math.exp(min(0, 1 - len(ref_tokens) / max(1, len(hyp_tokens))))
            total_bleu += bp * prec * 100.0

        bleu_score = total_bleu / max(1, n_samples)
        chrf_score = total_chrf / max(1, n_samples)
        rouge_l_score = total_rouge / max(1, n_samples)

    return {
        "bleu_4": round(bleu_score, 2),
        "chrf_plus_plus": round(chrf_score, 2),
        "rouge_l": round(rouge_l_score, 2)
    }

