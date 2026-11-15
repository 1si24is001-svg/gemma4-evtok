# RL with token penalties for context selection (REINFORCE, out-of-fold)

Tasks with gold: 120. Reward per step = gold captured / |gold| − λ · tokens/1000; cap 16000 tokens. Baseline: BM25 top-k given exactly the tokens the policy spent on each task.

| λ per 1k tokens | mean tokens used | RL recall | BM25 recall at same tokens | Δ recall [95% CI] | RL Hit | BM25 Hit | Hit wins/losses, p | mean stop step |
|---|---|---|---|---|---|---|---|---|
| 0.02 | 14869 | 0.524 | 0.589 | -0.065 [-0.124, -0.009] | 0.650 | 0.742 | 3/14, p=0.013 | 39.7 |
| 0.05 | 8571 | 0.432 | 0.513 | -0.081 [-0.140, -0.026] | 0.583 | 0.683 | 4/16, p=0.012 | 42.1 |
| 0.1 | 9244 | 0.262 | 0.511 | -0.250 [-0.333, -0.169] | 0.392 | 0.683 | 6/41, p=0.000 | 39.9 |

Token spend varies per task (10th–90th percentile): λ=0.02: 12094–15999; λ=0.05: 3008–15974; λ=0.1: 2625–15995
