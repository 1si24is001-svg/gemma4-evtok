# Tier A results (out-of-fold, 5-fold task-level CV)

Tasks: 129; with gold function nodes: 120; addition-only: 9.

| Method | Recall@2k | Recall@4k | Recall@8k | Recall@16k | Recall-AUC | Hit@4k | Hit@8k | AnyTier@4k | File@4k |
|---|---|---|---|---|---|---|---|---|---|
| BM25 top-k | 0.357 | 0.457 | 0.549 | 0.592 | 0.489 | 0.625 | 0.733 | 0.457 | 0.806 |
| Dense top-k (anchor query) | 0.099 | 0.147 | 0.177 | 0.268 | 0.173 | 0.250 | 0.308 | 0.147 | 0.637 |
| Dense top-k (BM25-centroid query) | 0.065 | 0.118 | 0.185 | 0.289 | 0.164 | 0.233 | 0.317 | 0.118 | 0.677 |
| Hybrid top-k | 0.354 | 0.442 | 0.541 | 0.583 | 0.480 | 0.592 | 0.700 | 0.442 | 0.790 |
| Hybrid + tiers (no graph) | 0.334 | 0.432 | 0.527 | 0.615 | 0.477 | 0.600 | 0.708 | 0.474 | 0.871 |
| Graph expansion (full only) | 0.343 | 0.426 | 0.514 | 0.556 | 0.460 | 0.608 | 0.683 | 0.426 | 0.774 |
| Connected (full only) | 0.284 | 0.366 | 0.444 | 0.481 | 0.394 | 0.525 | 0.575 | 0.366 | 0.742 |
| Connected + tiers (EpT) | 0.345 | 0.411 | 0.497 | 0.554 | 0.452 | 0.592 | 0.650 | 0.475 | 0.790 |
| Random connected (control) | 0.230 | 0.286 | 0.338 | 0.417 | 0.318 | 0.461 | 0.525 | 0.286 | 0.675 |
| Oracle (gold first) | 0.840 | 0.946 | 0.976 | 0.987 | 0.937 | 1.000 | 1.000 | 0.947 | 0.992 |
| Hybrid top-k [default] | 0.360 | 0.437 | 0.548 | 0.623 | 0.492 | 0.600 | 0.717 | 0.437 | 0.815 |
| Connected + tiers (EpT) [default] | 0.271 | 0.329 | 0.427 | 0.491 | 0.380 | 0.467 | 0.575 | 0.555 | 0.726 |

## Paired differences vs Hybrid top-k (bootstrap 95% CI over 120 tasks; McNemar exact on Hit)

| Method | ΔRecall-AUC [95% CI] | ΔRecall@2k [CI] | ΔRecall@4k [CI] | ΔRecall@8k [CI] | ΔRecall@16k [CI] | Hit@4k b/c, p | Hit@8k b/c, p |
|---|---|---|---|---|---|---|---|
| BM25 top-k | +0.009 [-0.007, +0.027] | +0.004 [-0.023, +0.032] | +0.015 [-0.012, +0.043] | +0.009 [-0.019, +0.037] | +0.009 [-0.005, +0.024] | 5/1, p=0.219 | 5/1, p=0.219 |
| Dense top-k (anchor query) | -0.307 [-0.381, -0.234] | -0.255 [-0.327, -0.186] | -0.295 [-0.381, -0.208] | -0.364 [-0.453, -0.277] | -0.315 [-0.406, -0.225] | 6/47, p=0.000 | 7/54, p=0.000 |
| Dense top-k (BM25-centroid query) | -0.316 [-0.389, -0.243] | -0.289 [-0.370, -0.210] | -0.324 [-0.413, -0.236] | -0.356 [-0.443, -0.271] | -0.294 [-0.381, -0.209] | 7/50, p=0.000 | 5/51, p=0.000 |
| Hybrid + tiers (no graph) | -0.003 [-0.050, +0.045] | -0.019 [-0.077, +0.039] | -0.009 [-0.066, +0.050] | -0.014 [-0.077, +0.051] | +0.032 [-0.030, +0.093] | 10/9, p=1.000 | 11/10, p=1.000 |
| Graph expansion (full only) | -0.020 [-0.056, +0.013] | -0.010 [-0.049, +0.026] | -0.016 [-0.058, +0.025] | -0.026 [-0.071, +0.014] | -0.027 [-0.077, +0.018] | 7/5, p=0.774 | 5/7, p=0.774 |
| Connected (full only) | -0.086 [-0.131, -0.043] | -0.070 [-0.121, -0.021] | -0.076 [-0.133, -0.019] | -0.096 [-0.156, -0.040] | -0.102 [-0.165, -0.044] | 4/12, p=0.077 | 3/18, p=0.001 |
| Connected + tiers (EpT) | -0.028 [-0.069, +0.012] | -0.009 [-0.049, +0.031] | -0.030 [-0.087, +0.027] | -0.043 [-0.103, +0.015] | -0.029 [-0.084, +0.020] | 9/9, p=1.000 | 5/11, p=0.210 |
| Random connected (control) | -0.162 [-0.209, -0.117] | -0.124 [-0.174, -0.076] | -0.156 [-0.221, -0.092] | -0.203 [-0.262, -0.145] | -0.166 [-0.234, -0.101] | 3/10, p=0.092 | 1/13, p=0.002 |
| Oracle (gold first) | +0.457 [+0.389, +0.526] | +0.486 [+0.413, +0.559] | +0.504 [+0.427, +0.579] | +0.435 [+0.359, +0.513] | +0.404 [+0.327, +0.483] | 49/0, p=0.000 | 36/0, p=0.000 |
| Hybrid top-k [default] | +0.012 [-0.002, +0.029] | +0.006 [-0.016, +0.033] | -0.005 [-0.020, +0.012] | +0.007 [-0.020, +0.035] | +0.040 [+0.011, +0.074] | 1/0, p=1.000 | 3/1, p=0.625 |
| Connected + tiers (EpT) [default] | -0.100 [-0.153, -0.050] | -0.083 [-0.144, -0.025] | -0.113 [-0.177, -0.051] | -0.114 [-0.182, -0.047] | -0.092 [-0.159, -0.028] | 4/19, p=0.003 | 6/21, p=0.006 |

## Recall-AUC by repository

| Method | Textualize/rich (n=45) | encode/httpx (n=1) | fastapi/fastapi (n=62) | psf/requests (n=12) |
|---|---|---|---|---|
| BM25 top-k | 0.490 | 0.516 | 0.428 | 0.798 |
| Dense top-k (anchor query) | 0.203 | 0.312 | 0.076 | 0.546 |
| Dense top-k (BM25-centroid query) | 0.128 | 0.125 | 0.156 | 0.344 |
| Hybrid top-k | 0.474 | 0.531 | 0.415 | 0.829 |
| Hybrid + tiers (no graph) | 0.467 | 0.578 | 0.428 | 0.760 |
| Graph expansion (full only) | 0.477 | 0.500 | 0.387 | 0.771 |
| Connected (full only) | 0.411 | 0.438 | 0.320 | 0.706 |
| Connected + tiers (EpT) | 0.477 | 0.453 | 0.376 | 0.752 |
| Random connected (control) | 0.307 | 0.302 | 0.282 | 0.544 |
| Oracle (gold first) | 0.966 | 0.984 | 0.904 | 0.996 |
| Hybrid top-k [default] | 0.477 | 0.547 | 0.436 | 0.829 |
| Connected + tiers (EpT) [default] | 0.397 | 0.312 | 0.300 | 0.731 |

## Packet composition at 8k (mean)

| Method | nodes | full-source nodes | components | cost | evidence density |
|---|---|---|---|---|---|
| BM25 top-k | 26.2 | 26.2 | 16.7 | 7995 | 0.095 |
| Dense top-k (anchor query) | 44.5 | 44.5 | 31.0 | 7995 | 0.027 |
| Dense top-k (BM25-centroid query) | 42.9 | 42.9 | 29.9 | 7995 | 0.024 |
| Hybrid top-k | 25.1 | 25.1 | 15.6 | 7995 | 0.092 |
| Hybrid + tiers (no graph) | 91.8 | 89.9 | 60.8 | 7998 | 0.058 |
| Graph expansion (full only) | 19.7 | 19.7 | 9.0 | 7870 | 0.092 |
| Connected (full only) | 35.6 | 35.6 | 4.8 | 7655 | 0.072 |
| Connected + tiers (EpT) | 36.6 | 33.0 | 4.5 | 7860 | 0.079 |
| Random connected (control) | 23.7 | 23.7 | 2.6 | 7783 | 0.076 |
| Oracle (gold first) | 26.4 | 25.0 | 16.2 | 7946 | 0.232 |
| Hybrid top-k [default] | 23.9 | 23.9 | 15.6 | 7995 | 0.092 |
| Connected + tiers (EpT) [default] | 68.9 | 35.3 | 1.9 | 7540 | 0.060 |

## CV choices per fold

- BM25 top-k: flat|wd0.0_wa0.0_anchor; flat|wd0.0_wa0.0_anchor; flat|wd0.0_wa0.0_anchor; flat|wd0.0_wa0.0_anchor; flat|wd0.0_wa0.0_anchor
- Dense top-k (anchor query): flat|dense_anchor; flat|dense_anchor; flat|dense_anchor; flat|dense_anchor; flat|dense_anchor
- Dense top-k (BM25-centroid query): flat|dense_bm; flat|dense_bm; flat|dense_bm; flat|dense_bm; flat|dense_bm
- Hybrid top-k: flat|wd0.0_wa2.0_anchor; flat|wd0.0_wa0.0_anchor; flat|wd0.5_wa2.0_bm; flat|wd0.5_wa1.0_bm; flat|wd0.5_wa2.0_bm
- Hybrid + tiers (no graph): flatT|wd0.5_wa1.0_bm|rho0.5_a; flatT|wd0.5_wa1.0_bm|rho0.5_a; flatT|wd0.5_wa1.0_bm|rho0.5_a; flatT|wd0.5_wa1.0_bm|rho0.5_a; flatT|wd1.0_wa1.0_bm|rho0.5_a
- Graph expansion (full only): gx|wd0.0_wa0.0_anchor|b0.0_h2_k10; gx|wd0.0_wa0.0_anchor|b1.0_h2_k10; gx|wd0.0_wa2.0_anchor|b0.0_h2_k10; gx|wd0.0_wa0.0_anchor|b0.0_h2_k10; gx|wd0.0_wa0.0_anchor|b1.0_h2_k10
- Connected (full only): conn|wd0.0_wa0.0_anchor|b1.0_h2_k10; conn|wd0.0_wa0.0_anchor|b1.0_h2_k10; conn|wd0.5_wa1.0_anchor|b1.0_h2_k10; conn|wd0.0_wa1.0_anchor|b1.0_h2_k10; conn|wd1.0_wa1.0_bm|b1.0_h2_k10
- Connected + tiers (EpT): connT|wd0.0_wa0.0_anchor|b1.0_h2_k10_rho0.5_a; connT|wd0.0_wa0.0_anchor|b1.0_h2_k10_rho0.5_a; connT|wd0.5_wa1.0_anchor|b1.0_h2_k10_rho0.5_a; connT|wd0.0_wa0.0_anchor|b1.0_h2_k10_rho0.5_a; connT|wd0.0_wa0.0_anchor|b1.0_h2_k10_rho0.5_a
