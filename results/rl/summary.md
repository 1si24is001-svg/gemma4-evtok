# RL with token penalties for context selection (REINFORCE, out-of-fold)

Tasks with gold: 120. Reward per step = gold captured / |gold| − λ · tokens/1000; cap 16000 tokens. Baseline: BM25 top-k given exactly the tokens the policy spent on each task.

| λ per 1k tokens | mean tokens used | RL recall | BM25 recall at same tokens | Δ recall [95% CI] | RL Hit | BM25 Hit | Hit wins/losses, p | mean stop step |
|---|---|---|---|---|---|---|---|---|
| 0.02 | 12951 | 0.457 | 0.569 | -0.111 [-0.178, -0.048] | 0.625 | 0.717 | 6/17, p=0.035 | 47.2 |
| 0.05 | 4443 | 0.377 | 0.444 | -0.067 [-0.135, -0.003] | 0.525 | 0.617 | 5/16, p=0.027 | 16.5 |
| 0.1 | 2719 | 0.289 | 0.387 | -0.099 [-0.164, -0.036] | 0.433 | 0.558 | 6/21, p=0.006 | 20.2 |

Token spend varies per task (10th–90th percentile): λ=0.02: 7772–15996; λ=0.05: 1430–9508; λ=0.1: 1104–5183
