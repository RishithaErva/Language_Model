# Ablation Comparison: Standard Model L vs. `no_pos`

| Metric | Standard Model L | Ablated (`no_pos`) |
|---|---|---|
| Val Loss | 3.5140 | 3.5969 |
| Val Perplexity | 33.5828 | 36.4844 |
| Val BPB | 0.4284 | 0.4385 |
| Test Loss | 3.4991 | 3.5834 |
| Test Perplexity | 33.0859 | 35.9974 |
| Test BPB | 0.4256 | 0.4359 |
| BLEU-4 | 3.2100 | 2.2800 |
| chrF++ | 22.7300 | 21.8200 |
| ROUGE-L | 6.9400 | 5.7600 |
| Distinct-1 | 0.6201 | 0.5714 |
| Distinct-2 | 0.6647 | 0.6611 |
| 4-gram Repetition | 0.2829 | 0.2593 |
| Attention Entropy (L1H1) | 1.1077 | 0.5459 |
| Attention Distance (L1H1) | 1.9287 | 0.5623 |