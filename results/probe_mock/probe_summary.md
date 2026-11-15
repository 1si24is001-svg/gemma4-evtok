# Tier A.5 localization probe (budget 8000, model MOCK)

| Method | n | Gold in packet (full) | Gold in packet (any tier) | Acc@1 | Acc@5 | Acc@5 given gold in full | Parse OK |
|---|---|---|---|---|---|---|---|
| BM25 top-k | 120 | 0.733 | 0.733 | 0.317 | 0.317 | 0.432 | 1.000 |
| Hybrid + tiers (no graph) | 120 | 0.708 | 0.725 | 0.300 | 0.300 | 0.424 | 1.000 |
| Connected + tiers (EpT) | 120 | 0.683 | 0.717 | 0.308 | 0.308 | 0.451 | 1.000 |
| Oracle (gold first) | 120 | 1.000 | 1.000 | 0.325 | 0.325 | 0.325 | 1.000 |

## Waterfall (share of tasks)

| Method | missed by retrieval | in packet, model misses | model hits (Acc@5) |
|---|---|---|---|
| BM25 top-k | 0.267 | 0.417 | 0.317 |
| Hybrid + tiers (no graph) | 0.275 | 0.425 | 0.300 |
| Connected + tiers (EpT) | 0.283 | 0.408 | 0.308 |
| Oracle (gold first) | 0.000 | 0.675 | 0.325 |
