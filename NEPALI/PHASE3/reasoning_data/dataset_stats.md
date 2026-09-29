# Nepali (Model L) — Synthetic Reasoning Dataset Statistics

## Split sizes

| Split | Examples |
|---|---|
| train | 4497 |
| val | 735 |
| test | 1149 |

## Template counts per split

| Split | pairwise | threeway | transitive | transitive_4hop |
|---|---|---|---|---|
| train | 1500 | 1500 | 1497 | 0 |
| val | 250 | 250 | 235 | 0 |
| test | 250 | 250 | 250 | 399 |

## Leakage avoidance

Person/object names are split into disjoint train/val/test pools before any example is generated. `transitive_4hop` (a 3-hop chain over 4 entities) is generated only for test -- a held-out relation pattern never seen during finetuning. Every template carries multiple statement/question phrasing variants (randomly chosen per example) to avoid one-shape overfitting.