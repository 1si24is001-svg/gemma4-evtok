# When does code-graph structure help a token-budgeted coding agent?

**A matched-budget study of context selection on the Gemma 4 Developer Agent graphs**

## Abstract

A coding agent on a small local model must fit its evidence into a few thousand tokens. Recent systems claim repository graphs beat similarity search at finding that evidence.

We test the claim at fixed token budgets on the competition's 129 tasks, call graphs and node embeddings. Ten context selectors, from BM25 to a connected selector that packs each function at one of three levels of detail, are scored at 2k to 16k tokens by whether the functions the reference fix edits appear in full source, with 5-fold cross-validation.

No graph-based selector beats BM25. Requiring connectivity costs 8.6 recall points [95% CI 4.3, 13.1]. Multi-level packing recovers most of that loss. A supervised ranker, a Steiner-connected selector, a budget-split graph rescue and a token-penalized RL policy also fail to beat BM25. The embeddings retrieve a third as much as BM25 because they have no text encoder. Structure has headroom: when BM25 misses, the edited function is one or two calls from the retrieved code in 17 of 32 cases, and on edits spanning four or more functions multi-level packing adds 9.2 points [3.1, 16.2]. An oracle reaches 0.94 recall against 0.49 for the best method. Finding the code is the bottleneck.

> **Key findings**
>
> - **No graph-based selector beats BM25** at any budget from 2k to 16k tokens (Table 1).
> - **Forcing connectivity hurts:** a connected subgraph costs 8.6 recall points (Section 5).
> - **Multi-level packing is the one gain:** +9.2 points when a fix edits four or more functions (Section 6.2), replicated by a learned ranker (+11.0).
> - **Learning does not close the gap:** a supervised ranker and token-penalized RL stay below BM25 (Sections 6.3, 6.4).
> - **Finding the code is the bottleneck:** an oracle reaches 0.94 recall against 0.49.

---

## 1 Introduction

The competition harness gives Gemma 4 a 32,768-token window, compacts history at 14,336 tokens and truncates tool results to 5,000 characters. The agent sees a small slice of the repository, and that slice bounds everything after it.

The competition ships a call graph and node embeddings for every task, and recent work argues such structure helps: RepoGraph [1] says similarity search finds similar rather than related code. The graph localizers in Section 2 do not compare against flat retrieval at a fixed token budget, which matters for local models.

- **RQ1.** At a fixed token budget, does a connected, graph-based selection contain more of the code a fix edits than flat top-k retrieval?
- **RQ2.** Which part matters: graph expansion, the connectivity constraint, or packing nodes at lower detail?
- **RQ3.** When structure fails, why, and on which tasks does it help?

Contributions:

1. A matched-budget evaluation of ten selectors plus four robustness checks, including a learned ranker and token-penalized RL. The main result is negative: structure buys no recall at any budget.
2. A diagnosis of why: sparse single-type edges, no query encoder, and gold functions adjacent to retrieved code but not identifiable from text.
3. An exploratory finding that multi-level packing helps on multi-function edits.
4. A reproducible protocol and a research agenda (Section 7).

No LLM runs in these results. We measure what reaches the context window (Section 8).

---

## 2 Related work

**Repository graphs.** RepoGraph [1] adds ego-graphs and improves several agents on SWE-bench [5]. LocAgent [2] and CoSIL [3] let an LLM search code graphs. OrcaLoca [4] prunes context by graph distance. KGCompass [6] finds many bugs need multi-hop paths. None controls context size; our selectors make no LLM calls and are compared at equal budgets.

**Flat retrieval.** Agentless [7] localizes hierarchically without an agent loop. SweRank [8] trains a retriever and reranker that outperform agent-based localization. Both suggest flat ranking is a strong baseline, and our results agree.

**Connected subgraphs.** G-Retriever [9] retrieves a connected subgraph as a prize-collecting Steiner tree. Our connected selector is a greedy, token-budgeted relative. We do not claim connected selection as new.

**Graph serialization and fault localization.** Fatemi et al. [10] show that graph-to-text encoding changes LLM accuracy; our probe notebook tests three encodings. AutoFL [11] uses failing-test coverage, which these tasks lack.

---

## 3 Data

| Item | What the dataset contains |
|---|---|
| Tasks | 129. fastapi 67, rich 48, requests 13, httpx 1. Created 2023-07 to 2026-06. |
| Problem statements | PR titles and descriptions, median 418 characters, often features or refactors. One traceback. No hints, no fail-to-pass list. |
| Gold patches | 91 touch one file; 11 add a file. |
| Graphs | 127, one per commit. Nodes carry a dotted id and source text only, no file paths or line numbers. One edge type, `calls`, which also encodes class-to-method containment. |
| Embeddings | One 256-d vector per node. No encoder is provided; `search_similar_code` accepts symbol names, not free text. |

**Leakage check.** 77.5% of lines the fix deletes appear in node text, but only 4.6% of lines it adds. The graphs reflect the base commit.

**Gold labels.** Each removed line maps to the most specific node in the patched file containing it; pure additions use adjacent context lines. This labels 120 tasks. The other 9 only add code and are excluded from function-level metrics.

---

## 4 Method

### 4.1 Relevance signals (problem statement only)

- **BM25** [12] over each node's identifier-split id (counted twice) and source, queried with the cleaned problem statement.
- **Identifier anchors.** Backticked, dotted, CamelCase and snake_case names resolved as the harness does (exact id, dotted suffix, last component); names matching over eight nodes are dropped. Anchors resolve in 62 of 129 tasks.
- **Dense pseudo-query.** The mean embedding of anchor nodes, or of the BM25 top 5, ranked by cosine similarity.
- **Hybrid score** s(v) = z(log(1 + BM25)) + w_d · z(dense) + 3 · w_a · anchor(v), where z is a z-score over the candidate pool. Test modules are excluded from the pool.

### 4.2 Token costs and tiers

A node enters at one of three tiers: outline (dotted id), signature (definition line plus first docstring paragraph) or full source, each with a header line. Costs are characters / 3.6 (Section 8). Tier values w = (w_outline, w_sig, 1) take two settings, (0.02, 0.2) and (0.1, 0.45).

### 4.3 Selectors

Every selector returns a packet costing at most B; unit tests check this.

| ID | Selector |
|---|---|
| BM25 / Dense / Hybrid top-k | Score order, full source, first-fit until B. |
| Hybrid + tiers | Greedy over (node, tier) actions by value per token, no graph. |
| Graph expansion | h-hop neighbourhood of top-K seeds, re-ranked by s(v) + β·exp(−d). |
| Connected | Greedy from top-K seeds; a node is eligible once a neighbour is selected. |
| Connected + tiers (EpT) | Connected, plus tier upgrades. |
| Random connected | Random growth from the same seeds (control). |
| Oracle | Gold nodes first, then hybrid top-k. |

The connected greedy works as follows:

```
utility u(v) = exp((s(v) + β·exp(−dist(v, seeds)) − max) / T)
heap ← actions (v, tier) for seeds, priority = Δvalue / Δcost^ρ
while heap and budget left:
    pop best action; skip if stale or over budget
    apply it; if v is newly selected, push actions for its unseen neighbours within h hops
    push upgrade actions for v
```

The grid covers β ∈ {0, 1}, h ∈ {1, 2}, K ∈ {3, 5, 10}, ρ ∈ {0.5, 1}, both tier settings, and 15 hybrid weightings, with T = 1.5 fixed.

### 4.4 Protocol and metrics

- **Recall@B.** The fraction of gold functions present in full source.
- **Hit@B.** At least one gold function present in full source.
- **Recall-AUC.** Mean Recall@B over the four budgets, giving equal weight to each doubling.

We use 5 task-level folds stratified by repository (seed 0), choosing each method's configuration on the other four folds. Every reported number is out-of-fold. Differences against hybrid top-k use a paired bootstrap (10,000 resamples) and an exact McNemar test on Hit.

---

## 5 Results

![Recall vs. context budget for each selector](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F32549161%2Ff0d1b00d30e6f7e7a04d7c129f4f3005%2Ffig_recall_vs_budget.png?generation=1794318886942714&alt=media)

*Figure 1. Function Recall@B, out-of-fold, 120 tasks. Bands are 95% bootstrap intervals.*

**Table 1.** Out-of-fold results on 120 tasks.

| Method | R@2k | R@4k | R@8k | R@16k | Recall-AUC | Hit@8k |
|---|---|---|---|---|---|---|
| BM25 top-k | 0.357 | 0.457 | 0.549 | 0.592 | **0.489** | 0.733 |
| Hybrid top-k | 0.354 | 0.442 | 0.541 | 0.583 | 0.480 | 0.700 |
| Hybrid + tiers (no graph) | 0.334 | 0.432 | 0.527 | 0.615 | 0.477 | 0.708 |
| Graph expansion | 0.343 | 0.426 | 0.514 | 0.556 | 0.460 | 0.683 |
| Connected + tiers (EpT) | 0.345 | 0.411 | 0.497 | 0.554 | 0.452 | 0.650 |
| Connected (full only) | 0.284 | 0.366 | 0.444 | 0.481 | 0.394 | 0.575 |
| Random connected | 0.230 | 0.286 | 0.338 | 0.417 | 0.318 | 0.525 |
| Dense top-k (anchor query) | 0.099 | 0.147 | 0.177 | 0.268 | 0.173 | 0.308 |
| Dense top-k (BM25-centroid query) | 0.065 | 0.118 | 0.185 | 0.289 | 0.164 | 0.317 |
| Oracle | 0.840 | 0.946 | 0.976 | 0.987 | 0.937 | 1.000 |

**Table 2.** Paired differences against hybrid top-k.

| Method | ΔRecall-AUC [95% CI] | Hit@8k wins/losses, McNemar p |
|---|---|---|
| BM25 top-k | +0.009 [−0.007, +0.027] | 5/1, 0.22 |
| Hybrid + tiers | −0.003 [−0.050, +0.045] | 11/10, 1.00 |
| Graph expansion | −0.020 [−0.056, +0.013] | 5/7, 0.77 |
| Connected + tiers (EpT) | −0.028 [−0.069, +0.012] | 5/11, 0.21 |
| Connected (full only) | −0.086 [−0.131, −0.043] | 3/18, 0.001 |
| Random connected | −0.162 [−0.209, −0.117] | 1/13, 0.002 |

**RQ1.** No graph-based selector beats flat retrieval at any budget. Untuned BM25 is best; per-fold hybrid tuning overfit slightly (−0.009).

**RQ2.** Connectivity costs 8.6 points (interval excludes zero). Connected packets average 4.5 components at 8k against 15.6 for hybrid top-k, spending budget on bridges instead of high-scoring functions. Tiers recover 5.8 of those points, and without a graph they give the best file coverage at 4k (87.1% against 79.0%). Random growth from the same seeds is 16 points worse, so the graph carries signal, just less than text. Every connected method chose K = 10 seeds, the grid maximum, in every fold. The selectors did best when they behaved least like graph walks.

**RQ3** is answered in Section 6.

---

## 6 Analysis

### 6.1 Why structure does not help here

1. **Sparse, single-type edges.** 23% of non-test nodes have no `calls` edge, and there are no import or inheritance edges.
2. **No query encoder.** Dense retrieval reaches 0.258 Recall-AUC on the 58 tasks with resolved anchors and 0.092 on the rest. Code-to-code embeddings cannot replace a text query.
3. **The gold is near but not identifiable.** In 17 of the 32 tasks BM25 misses at 8k, a gold function is one or two hops from the packet (29 within four). Expansion is guided by the same text scores that missed.

### 6.2 When structure helps (exploratory)

We split tasks by the number of gold functions after seeing the main results. These comparisons are post hoc and uncorrected for multiplicity.

| Gold functions | n | EpT − BM25 | Tiers (no graph) − BM25 | Graph expansion − BM25 |
|---|---|---|---|---|
| 1 | 55 | −0.091 [−0.164, −0.023] | −0.055 [−0.136, +0.023] | −0.068 [−0.132, −0.014] |
| 2–3 | 32 | −0.010 [−0.104, +0.077] | −0.046 [−0.143, +0.051] | +0.012 [−0.036, +0.057] |
| ≥ 4 | 33 | +0.027 [−0.017, +0.067] | **+0.092 [+0.031, +0.162]** | −0.003 [−0.037, +0.023] |

For single-function edits, spreading the budget hurts because the best text match is usually the answer. For four or more functions, many signatures help. Together BM25 and EpT reach 76.7% Hit@8k against 73.3% for BM25 alone.

### 6.3 Robustness: learning and other structures

We then tested three agenda items with the same folds, against BM25.

| Method | Recall-AUC | ΔAUC vs BM25 [95% CI] |
|---|---|---|
| Budget-split rescue: 25–50% of B to neighbours of top hits | 0.477 | −0.012 [−0.030, +0.008] |
| Steiner-connected top-k (shortest-path approximation of [9]) | 0.487 | −0.002 [−0.005, +0.000] |
| Learned ranker, logistic regression | 0.464 | −0.025 [−0.074, +0.023] |
| Learned ranker, boosted trees | 0.433 | −0.055 [−0.102, −0.012] |
| Learned ranker + tiers, logistic | 0.424 | −0.065 [−0.133, +0.005] |

The ranker scores candidates (BM25 top 200 plus two-hop neighbours of the top 10; these contain a gold function in 110 of 120 tasks) from 18 features including BM25, anchors, dense scores, hop distances, degree and node type, trained on training folds only. None beats BM25. With tiers, the logistic ranker again helps on edits of four or more functions (+0.110 [+0.036, +0.190]), replicating Section 6.2.

### 6.4 RL with token penalties

A REINFORCE policy, trained on training folds, adds functions or stops; reward is the share of gold captured minus λ per 1,000 tokens. BM25 gets the same tokens per task.

| λ per 1k tokens | Mean tokens | RL recall | BM25 recall | Δ [95% CI] |
|---|---|---|---|---|
| 0.02 | 12,960 | 0.489 | 0.562 | −0.073 [−0.134, −0.014] |
| 0.05 | 2,766 | 0.281 | 0.397 | −0.116 [−0.180, −0.058] |
| 0.10 | 3,048 | 0.280 | 0.395 | −0.115 [−0.187, −0.046] |

Numbers are from the public Kaggle run. The penalty cuts spend from about 13k to 3k tokens, yet with 96 training tasks the policy ranks below BM25 at every λ. Our local run differed numerically (Δ −0.067 to −0.111) but agreed. This is RL over selection, not fine-tuning of Gemma.

---

## 7 Research agenda: what we tested, what the data ruled out, what comes next

We started from three ideas: a relevance field over the graph cut off at a token limit, failure signals propagated from failing tests, and RL rewarding short repair trajectories. Only *tested* rows are results.

| Idea | Status | Evidence or reason |
|---|---|---|
| Graph relevance under a hard budget | Tested | Connectivity costs 8.6 points. |
| Full / signature / outline packing | Tested | +9.2 on four-plus-function edits; replicated in 6.3. |
| Hybrid retrieval, identifier anchors | Tested | Dense reaches a third of BM25. |
| Learned neighbour ranker | Tested | Does not beat BM25 (6.3). |
| Budget-split rescue (static "expand after a miss") | Tested | −0.012 (6.3). A model-triggered version needs an agent loop. |
| Steiner-connected selection | Tested (approximation) | −0.002. Exact PCST [9] needs a GPL library we did not ship. |
| Failure propagation from failing tests | Ruled out by the data | No fail-to-pass lists; `test_patch` would leak the answer. |
| Gemma 4 E4B probe, waterfall, serialization ablation [10] | Built, not yet run | GPU notebook: four packets plus three graph encodings of one packet. |
| Selector as an in-harness skill | Built | Stdlib BM25 script; ranked a gold file first on 3 of 3 snapshots tested. Not yet scored. |
| Localization labels as an eval set | Built | Script regenerates labels for 120 tasks; raw data is not redistributed. |
| RL with token penalties | Tested on selection (6.4) | Learns a budget dial, not better ranking. RL fine-tuning of Gemma on tool calls is deferred: costly rollouts, test gaming. |

Three rules follow for a budgeted local agent: localize lexically first and call `search_similar_code` only with a symbol name; use neighbours to recover from a miss, not to build the first packet; for multi-location changes, show many signatures before reading any function in full.

---

## 8 Limitations

- **No model in the loop.** We measure whether evidence reaches the context, not whether Gemma 4 uses it, and ran no patches against tests.
- **Approximate token costs.** Shared across methods, so comparisons are matched, but absolute budgets are approximate.
- **Approximate gold labels.** Text matching can label moved code as edited (two tasks).
- **Narrow data.** Four Python libraries, 52% fastapi, PR descriptions. Tracebacks or richer edges could favour structure.
- **Finite grid.** Connected selectors chose the largest seed count tried; larger K approaches flat retrieval.
- **Post-hoc subgroups.** Section 6.2 was not pre-registered; 6.3 replicates it once.

---

## 9 Conclusion

At fixed budgets on the competition graphs, neither hand-tuned nor learned use of structure beats BM25. Multi-level packing is the one robust gain, on multi-function edits. The embeddings need a text-side encoder. The 45-point oracle gap says localization is where a budgeted local agent loses most; the graph's headroom sits one or two hops from what text search found, but text features cannot yet pick the right neighbour.

---

## Reproducibility

- **Code.** Public notebook with an extended research log: https://www.kaggle.com/code/aadarshmukundan/ept-tier-a-reproduction. CPU only; numpy, matplotlib, scikit-learn; Apache-2.0. Unit tests check the budget bound, connectivity and patch parsing.
- **Data.** Only `tasks.jsonl`, `graphs/` and `embeddings/`, attached as input, not redistributed.
- **Determinism.** Seed 0 throughout; folds and per-fold choices are saved.
- **Leakage boundary.** Selection reads only `problem_statement`, graphs and embeddings. `patch` is used only for labels, the oracle and metrics. `test_patch` is never read.

---

## References

- [1] S. Ouyang et al. RepoGraph: Enhancing AI Software Engineering with Repository-level Code Graph. ICLR 2025. arXiv:2410.14684.
- [2] LocAgent: Graph-Guided LLM Agents for Code Localization. arXiv:2503.09089, 2025.
- [3] CoSIL: Issue Localization via LLM-Driven Iterative Code Graph Searching. ASE 2025. arXiv:2503.22424.
- [4] Yu et al. OrcaLoca: An LLM Agent Framework for Software Issue Localization. ICML 2025. arXiv:2502.00350.
- [5] C. E. Jimenez et al. SWE-bench: Can Language Models Resolve Real-World GitHub Issues? ICLR 2024. arXiv:2310.06770.
- [6] B. Yang et al. KGCompass: Knowledge Graph Enhanced Repository-Level Software Repair. arXiv:2503.21710, 2025.
- [7] C. S. Xia, Y. Deng, S. Dunn, L. Zhang. Agentless: Demystifying LLM-based Software Engineering Agents. arXiv:2407.01489, 2024.
- [8] R. G. Reddy et al. SweRank: Software Issue Localization with Code Ranking. ICLR 2026. arXiv:2505.07849.
- [9] X. He et al. G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering. arXiv:2402.07630, 2024.
- [10] B. Fatemi, J. Halcrow, B. Perozzi. Talk like a Graph: Encoding Graphs for Large Language Models. ICLR 2024. arXiv:2310.04560.
- [11] S. Kang, G. An, S. Yoo. A Quantitative and Qualitative Evaluation of LLM-Based Explainable Fault Localization. FSE 2024. arXiv:2308.05487.
- [12] S. Robertson, H. Zaragoza. The Probabilistic Relevance Framework: BM25 and Beyond. Foundations and Trends in Information Retrieval 3(4), 2009.
