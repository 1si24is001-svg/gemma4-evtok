# Tier A results (out-of-fold, 5-fold task-level CV)

Tasks: 2; with gold function nodes: 1; addition-only: 1.

| Method | Recall@2k | Recall@4k | Recall@8k | Recall@16k | Recall-AUC | Hit@4k | Hit@8k | AnyTier@4k | File@4k |
|---|---|---|---|---|---|---|---|---|---|
| BM25 top-k | 0.043 | 0.087 | 0.087 | 0.087 | 0.076 | 1.000 | 1.000 | 0.087 | 1.000 |
| Dense top-k (anchor query) | 0.087 | 0.087 | 0.217 | 0.391 | 0.196 | 1.000 | 1.000 | 0.087 | 0.500 |
| Dense top-k (BM25-centroid query) | 0.087 | 0.087 | 0.174 | 0.435 | 0.196 | 1.000 | 1.000 | 0.087 | 0.500 |
| Hybrid top-k | 0.043 | 0.087 | 0.087 | 0.087 | 0.076 | 1.000 | 1.000 | 0.087 | 1.000 |
| Hybrid + tiers (no graph) | 0.174 | 0.217 | 0.261 | 0.304 | 0.239 | 1.000 | 1.000 | 0.217 | 1.000 |
| Graph expansion (full only) | 0.043 | 0.043 | 0.174 | 0.174 | 0.109 | 1.000 | 1.000 | 0.043 | 1.000 |
| Connected (full only) | 0.000 | 0.000 | 0.087 | 0.174 | 0.065 | 0.000 | 1.000 | 0.000 | 1.000 |
| Connected + tiers (EpT) | 0.000 | 0.174 | 0.217 | 0.304 | 0.174 | 1.000 | 1.000 | 0.217 | 1.000 |
| Random connected (control) | 0.000 | 0.029 | 0.058 | 0.072 | 0.040 | 0.667 | 1.000 | 0.029 | 0.667 |
| Oracle (gold first) | 0.435 | 0.609 | 0.783 | 0.913 | 0.685 | 1.000 | 1.000 | 0.652 | 1.000 |
| Hybrid top-k [default] | 0.174 | 0.130 | 0.174 | 0.304 | 0.196 | 1.000 | 1.000 | 0.130 | 1.000 |
| Connected + tiers (EpT) [default] | 0.087 | 0.174 | 0.174 | 0.304 | 0.185 | 1.000 | 1.000 | 0.652 | 0.500 |

## Paired differences vs Hybrid top-k (bootstrap 95% CI over 1 tasks; McNemar exact on Hit)

| Method | ΔRecall-AUC [95% CI] | ΔRecall@2k [CI] | ΔRecall@4k [CI] | ΔRecall@8k [CI] | ΔRecall@16k [CI] | Hit@4k b/c, p | Hit@8k b/c, p |
|---|---|---|---|---|---|---|---|
| BM25 top-k | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 0/0, p=1.000 | 0/0, p=1.000 |
| Dense top-k (anchor query) | +0.120 [+0.120, +0.120] | +0.043 [+0.043, +0.043] | +0.000 [+0.000, +0.000] | +0.130 [+0.130, +0.130] | +0.304 [+0.304, +0.304] | 0/0, p=1.000 | 0/0, p=1.000 |
| Dense top-k (BM25-centroid query) | +0.120 [+0.120, +0.120] | +0.043 [+0.043, +0.043] | +0.000 [+0.000, +0.000] | +0.087 [+0.087, +0.087] | +0.348 [+0.348, +0.348] | 0/0, p=1.000 | 0/0, p=1.000 |
| Hybrid + tiers (no graph) | +0.163 [+0.163, +0.163] | +0.130 [+0.130, +0.130] | +0.130 [+0.130, +0.130] | +0.174 [+0.174, +0.174] | +0.217 [+0.217, +0.217] | 0/0, p=1.000 | 0/0, p=1.000 |
| Graph expansion (full only) | +0.033 [+0.033, +0.033] | +0.000 [+0.000, +0.000] | -0.043 [-0.043, -0.043] | +0.087 [+0.087, +0.087] | +0.087 [+0.087, +0.087] | 0/0, p=1.000 | 0/0, p=1.000 |
| Connected (full only) | -0.011 [-0.011, -0.011] | -0.043 [-0.043, -0.043] | -0.087 [-0.087, -0.087] | +0.000 [+0.000, +0.000] | +0.087 [+0.087, +0.087] | 0/1, p=1.000 | 0/0, p=1.000 |
| Connected + tiers (EpT) | +0.098 [+0.098, +0.098] | -0.043 [-0.043, -0.043] | +0.087 [+0.087, +0.087] | +0.130 [+0.130, +0.130] | +0.217 [+0.217, +0.217] | 0/0, p=1.000 | 0/0, p=1.000 |
| Random connected (control) | -0.036 [-0.036, -0.036] | -0.043 [-0.043, -0.043] | -0.058 [-0.058, -0.058] | -0.029 [-0.029, -0.029] | -0.014 [-0.014, -0.014] | 0/0, p=1.000 | 0/0, p=1.000 |
| Oracle (gold first) | +0.609 [+0.609, +0.609] | +0.391 [+0.391, +0.391] | +0.522 [+0.522, +0.522] | +0.696 [+0.696, +0.696] | +0.826 [+0.826, +0.826] | 0/0, p=1.000 | 0/0, p=1.000 |
| Hybrid top-k [default] | +0.120 [+0.120, +0.120] | +0.130 [+0.130, +0.130] | +0.043 [+0.043, +0.043] | +0.087 [+0.087, +0.087] | +0.217 [+0.217, +0.217] | 0/0, p=1.000 | 0/0, p=1.000 |
| Connected + tiers (EpT) [default] | +0.109 [+0.109, +0.109] | +0.043 [+0.043, +0.043] | +0.087 [+0.087, +0.087] | +0.087 [+0.087, +0.087] | +0.217 [+0.217, +0.217] | 0/0, p=1.000 | 0/0, p=1.000 |

## Recall-AUC by repository

| Method | fastapi/fastapi (n=1) |
|---|---|
| BM25 top-k | 0.076 |
| Dense top-k (anchor query) | 0.196 |
| Dense top-k (BM25-centroid query) | 0.196 |
| Hybrid top-k | 0.076 |
| Hybrid + tiers (no graph) | 0.239 |
| Graph expansion (full only) | 0.109 |
| Connected (full only) | 0.065 |
| Connected + tiers (EpT) | 0.174 |
| Random connected (control) | 0.040 |
| Oracle (gold first) | 0.685 |
| Hybrid top-k [default] | 0.196 |
| Connected + tiers (EpT) [default] | 0.185 |

## Packet composition at 8k (mean)

| Method | nodes | full-source nodes | components | cost | evidence density |
|---|---|---|---|---|---|
| BM25 top-k | 8.0 | 8.0 | 8.0 | 7988 | 0.402 |
| Dense top-k (anchor query) | 19.0 | 19.0 | 10.0 | 8000 | 0.650 |
| Dense top-k (BM25-centroid query) | 29.0 | 29.0 | 25.0 | 8000 | 0.487 |
| Hybrid top-k | 8.0 | 8.0 | 8.0 | 7988 | 0.402 |
| Hybrid + tiers (no graph) | 113.0 | 112.0 | 89.0 | 8000 | 0.087 |
| Graph expansion (full only) | 10.0 | 10.0 | 10.0 | 7998 | 0.502 |
| Connected (full only) | 7.0 | 7.0 | 3.0 | 5454 | 0.330 |
| Connected + tiers (EpT) | 25.0 | 14.0 | 2.0 | 7997 | 0.335 |
| Random connected (control) | 10.3 | 10.3 | 2.3 | 7941 | 0.351 |
| Oracle (gold first) | 25.0 | 19.0 | 20.0 | 7957 | 0.895 |
| Hybrid top-k [default] | 10.0 | 10.0 | 7.0 | 7991 | 0.440 |
| Connected + tiers (EpT) [default] | 83.0 | 31.0 | 1.0 | 7981 | 0.104 |

## CV choices per fold

- BM25 top-k: flat|wd0.0_wa0.0_anchor; flat|wd0.0_wa0.0_anchor; flat|wd0.0_wa0.0_anchor; flat|wd0.0_wa0.0_anchor; flat|wd0.0_wa0.0_anchor
- Dense top-k (anchor query): flat|dense_anchor; flat|dense_anchor; flat|dense_anchor; flat|dense_anchor; flat|dense_anchor
- Dense top-k (BM25-centroid query): flat|dense_bm; flat|dense_bm; flat|dense_bm; flat|dense_bm; flat|dense_bm
- Hybrid top-k: flat|wd0.5_wa1.0_anchor; flat|wd0.0_wa0.0_anchor; flat|wd0.5_wa1.0_anchor; flat|wd0.5_wa1.0_anchor; flat|wd0.5_wa1.0_anchor
- Hybrid + tiers (no graph): flatT|wd0.0_wa1.0_anchor|rho0.5_a; flatT|wd0.0_wa0.0_anchor|rho0.5_a; flatT|wd0.0_wa1.0_anchor|rho0.5_a; flatT|wd0.0_wa1.0_anchor|rho0.5_a; flatT|wd0.0_wa1.0_anchor|rho0.5_a
- Graph expansion (full only): gx|wd0.5_wa1.0_anchor|b0.0_h1_k10; gx|wd0.0_wa0.0_anchor|b0.0_h1_k10; gx|wd0.5_wa1.0_anchor|b0.0_h1_k10; gx|wd0.5_wa1.0_anchor|b0.0_h1_k10; gx|wd0.5_wa1.0_anchor|b0.0_h1_k10
- Connected (full only): conn|wd0.5_wa1.0_anchor|b0.0_h1_k10; conn|wd0.0_wa0.0_anchor|b0.0_h1_k10; conn|wd0.5_wa1.0_anchor|b0.0_h1_k10; conn|wd0.5_wa1.0_anchor|b0.0_h1_k10; conn|wd0.5_wa1.0_anchor|b0.0_h1_k10
- Connected + tiers (EpT): connT|wd0.5_wa1.0_anchor|b0.0_h1_k10_rho0.5_a; connT|wd0.0_wa0.0_anchor|b0.0_h1_k10_rho0.5_a; connT|wd0.5_wa1.0_anchor|b0.0_h1_k10_rho0.5_a; connT|wd0.5_wa1.0_anchor|b0.0_h1_k10_rho0.5_a; connT|wd0.5_wa1.0_anchor|b0.0_h1_k10_rho0.5_a
