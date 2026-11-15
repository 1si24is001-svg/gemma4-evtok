# Tier A.5 localization probe (budget 8000, model MOCK)

| Method | n | Gold in packet (full) | Gold in packet (any tier) | Acc@1 | Acc@5 | Acc@5 given gold in full | Parse OK |
|---|---|---|---|---|---|---|---|
| BM25 top-k | 11 | 0.727 | 0.727 | 0.364 | 0.364 | 0.500 | 1.000 |
| Hybrid + tiers (no graph) | 11 | 0.727 | 0.727 | 0.364 | 0.364 | 0.500 | 1.000 |
| Connected + tiers (EpT) | 11 | 0.818 | 0.818 | 0.364 | 0.364 | 0.444 | 1.000 |
| Oracle (gold first) | 11 | 1.000 | 1.000 | 0.364 | 0.364 | 0.364 | 1.000 |
| EpT 0.9B, no graph text | 11 | 0.818 | 0.818 | 0.364 | 0.364 | 0.444 | 1.000 |
| EpT 0.9B + edge list | 11 | 0.818 | 0.818 | 0.364 | 0.364 | 0.444 | 1.000 |
| EpT 0.9B + per-entry adjacency | 11 | 0.818 | 0.818 | 0.364 | 0.364 | 0.444 | 1.000 |

## Waterfall (share of tasks)

| Method | missed by retrieval | in packet, model misses | model hits (Acc@5) |
|---|---|---|---|
| BM25 top-k | 0.273 | 0.364 | 0.364 |
| Hybrid + tiers (no graph) | 0.273 | 0.364 | 0.364 |
| Connected + tiers (EpT) | 0.182 | 0.455 | 0.364 |
| Oracle (gold first) | 0.000 | 0.636 | 0.364 |
| EpT 0.9B, no graph text | 0.182 | 0.455 | 0.364 |
| EpT 0.9B + edge list | 0.182 | 0.455 | 0.364 |
| EpT 0.9B + per-entry adjacency | 0.182 | 0.455 | 0.364 |
