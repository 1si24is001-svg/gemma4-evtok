# Tier A extensions (out-of-fold, same 5 folds as Tier A)

Tasks with gold: 120. Gold inside learned candidate set: 110.

| Method | R@2k | R@4k | R@8k | R@16k | Recall-AUC | Hit@8k | ΔAUC vs BM25 [95% CI] | Hit@8k wins/losses, p |
|---|---|---|---|---|---|---|---|---|
| BM25 top-k | 0.357 | 0.457 | 0.549 | 0.592 | 0.489 | 0.733 | — | — |
| Budget-split graph rescue | 0.343 | 0.419 | 0.538 | 0.608 | 0.477 | 0.725 | -0.012 [-0.030, +0.008] | 3/4, p=1.000 |
| Steiner-connected top-k (shortest paths) | 0.356 | 0.454 | 0.548 | 0.589 | 0.487 | 0.733 | -0.002 [-0.005, +0.000] | 0/0, p=1.000 |
| Learned ranker (lr) | 0.356 | 0.417 | 0.501 | 0.581 | 0.464 | 0.658 | -0.025 [-0.074, +0.023] | 7/16, p=0.093 |
| Learned ranker + tiers (lr) | 0.279 | 0.355 | 0.475 | 0.588 | 0.424 | 0.642 | -0.065 [-0.133, +0.005] | 9/20, p=0.061 |
| Learned ranker (gb) | 0.309 | 0.383 | 0.463 | 0.579 | 0.433 | 0.633 | -0.055 [-0.102, -0.012] | 3/15, p=0.008 |
| Learned ranker + tiers (gb) | 0.244 | 0.364 | 0.511 | 0.605 | 0.431 | 0.675 | -0.058 [-0.120, +0.006] | 8/15, p=0.210 |

## By number of gold functions (post hoc): ΔAUC vs BM25

| Method | 1 | 2–3 | ≥4 |
|---|---|---|---|
| Budget-split graph rescue | -0.023 [-0.055, +0.009] (n=55) | -0.012 [-0.052, +0.029] (n=32) | +0.007 [-0.011, +0.025] (n=33) |
| Steiner-connected top-k (shortest paths) | +0.000 [+0.000, +0.000] (n=55) | -0.003 [-0.008, +0.000] (n=32) | -0.005 [-0.014, +0.001] (n=33) |
| Learned ranker (lr) | +0.014 [-0.059, +0.091] (n=55) | -0.113 [-0.228, -0.009] (n=32) | -0.004 [-0.056, +0.045] (n=33) |
| Learned ranker + tiers (lr) | -0.136 [-0.255, -0.018] (n=55) | -0.121 [-0.238, -0.005] (n=32) | +0.110 [+0.036, +0.190] (n=33) |
| Learned ranker (gb) | -0.023 [-0.091, +0.041] (n=55) | -0.158 [-0.275, -0.051] (n=32) | -0.011 [-0.060, +0.033] (n=33) |
| Learned ranker + tiers (gb) | -0.105 [-0.209, +0.005] (n=55) | -0.111 [-0.236, +0.009] (n=32) | +0.072 [-0.001, +0.153] (n=33) |

## CV choices

- rescue: rescue|f0.25_m3_t2; rescue|f0.5_m10_t2; rescue|f0.5_m3_t2; rescue|f0.5_m3_t2; rescue|f0.5_m3_t2
- steiner: steiner|k10_t1; steiner|k5_t1; steiner|k5_t1; steiner|k5_t1; steiner|k5_t1
- logistic coefficients per fold (standardized features):
  - dense_bm_z +0.73, id_token_overlap +0.71, is_class -0.53, bm25_z +0.42, name_in_query -0.42, dense_anchor_z -0.35, log_len +0.27, depth +0.22
  - dense_bm_z +0.68, id_token_overlap +0.67, is_class -0.57, name_in_query -0.40, bm25_z +0.39, dense_anchor_z -0.24, depth +0.23, log_len +0.23
  - dense_bm_z +0.77, id_token_overlap +0.75, is_class -0.69, bm25_z +0.51, hop_top1 -0.44, name_in_query -0.43, dense_anchor_z -0.41, log_degree +0.34
  - id_token_overlap +0.55, bm25_z +0.51, is_class -0.40, log_len +0.38, dense_bm_z +0.36, name_in_query -0.33, hop_top1 -0.26, in_bm25_top200 +0.23
  - is_class -1.60, id_token_overlap +0.51, hop_top1 +0.45, bm25_z +0.42, dense_anchor_z +0.33, log_len +0.31, same_parent_top1 +0.29, name_in_query -0.28
