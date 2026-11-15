## Extended research log

This log records how the study was designed, what we decided at each step, and why. It goes beyond the 3,000-word Writeup. Every number in it comes from this notebook's outputs or from the dataset.

### Starting ideas

We began with three design ideas for local coding agents, first stated as physical metaphors and then rewritten as testable methods.

| Original idea | Technical form | What happened |
|---|---|---|
| Graph "gravity" with a sluice gate: the issue pulls on relevant code, and context flow stops at a hard token line | Score nodes by relevance and graph proximity; select a connected subgraph under a hard token budget, greedily by value per token | Implemented and tested (connected and connected + tiers selectors). Connectivity costs 8.6 recall points. |
| Elastic AST lattice: "pluck" the failing node and follow the vibration through dependencies | Propagate a failure signal from failing tests or stack frames along call paths (static test-path proximity) | Designed, then ruled out by the data audit: no fail-to-pass lists, empty hints, one traceback in 129 tasks. `test_patch` is the only test signal, and using it would leak the answer. |
| Staged trajectories: RL that penalizes tokens, tool calls, graph hops and retries | Reward = captured gold − λ·tokens on the context-selection MDP (REINFORCE) | **Tested on selection (section 9).** The penalty cuts mean spend from 13.0k tokens (λ = 0.02) to 2.8k and 3.0k (λ = 0.05 and 0.1), chosen per task. But the policy ranks worse than BM25 given the same tokens (Δ −0.073 to −0.116 in this notebook). RL training is sensitive to the platform: the same code on our machine gave Δ −0.067 to −0.111, with the same conclusion. Section 9 therefore repeats training with three seeds and reports each one. RL fine-tuning of Gemma over tool calls is still deferred: 129 tasks, costly containerized rollouts, and the risk of agents gaming tests. |

We kept one contribution: a deterministic, budgeted context selector. We dropped claims of optimality and of matching cloud models, since nothing here could support them.

### Asset audit (what the data actually contains)

- 129 tasks: fastapi 67, rich 48, requests 13, httpx 1. Problem statements are PR descriptions (median 418 characters), many of them features or refactors.
- Graphs carry only node id, name and source text, with no file paths or line numbers. There is one edge type (`calls`), and 23% of non-test nodes have no edges.
- Embeddings are 256-d with full coverage. The encoder is not provided, and `search_similar_code` only accepts symbol names. So we built dense queries from identifier anchors or from BM25 seeds.
- **Leakage check:** 77.5% of the lines each fix deletes appear in the graph text, against 4.6% of the lines it adds. The graphs reflect the base commit.
- **Gold mapping:** we match the reference patch's removed and context lines against node text. This labels 120 tasks; 9 only add code.

### Pre-registered decision gates

| Gate | Condition | Outcome |
|---|---|---|
| G0 leakage | No gold-derived input to selection; graphs pre-fix | Passed |
| G1 retrieval | EpT beats hybrid top-k at 3 or more budgets, with the CI excluding 0 | Failed. ΔAUC −0.028 [−0.069, +0.012]. Pivoted, as planned, to "when does structure help". |
| G2 harness | End-to-end tests runnable | The `swegemma` harness ships in the Kaggle wheelhouse but not in the dataset zip. No end-to-end repair was run. |

### Alternatives considered

| Alternative | Decision | Reason |
|---|---|---|
| Pure heuristic packing | Main method | Deterministic and CPU-only |
| Multi-level (semantic) compression | Integrated as tiers | Recovers most of the connectivity cost; +9.2 on multi-function edits |
| Hybrid BM25 + dense | Baseline and seed retrieval | Per-fold tuning overfit slightly; untuned BM25 is best |
| Learned node / edit-location ranker | Tested (section 8 of this notebook) | Logistic regression and boosted trees on 18 features, trained per fold on training tasks only. Neither beats BM25 (ΔAUC −0.025 and −0.055). With tiers, logistic regression gains +0.110 on edits of four or more functions, replicating the tier effect. |
| Iterative refinement | Tested as a static budget split | 25–50% of the budget goes to neighbours of the top hits: −0.012 vs BM25. A model-triggered version needs an agent loop. |
| Prize-collecting Steiner tree (G-Retriever) | Tested as an approximation | Seeds connected by shortest paths: −0.002 vs BM25. Exact PCST would need `pcst_fast`, which is GPL-licensed, so it was not shipped. |
| Graph serialization ablation (Talk like a Graph) | Built (GPU notebook) | The same EpT packet rendered with no graph text, with a call-edge list, or with per-entry adjacency lines |
| Gemma 4 localization probe + loss waterfall | Built (GPU notebook) | Tests whether the model uses evidence that is present |
| Selector as an ADK skill script | Built | A stdlib BM25 localizer packaged as a skill in a main-track submission variant. It ranked a gold file first on 3 of 3 real snapshots tested offline. Not yet scored by the harness. |
| Localization labels as an evaluation set | Built (`evtok.export_labels`) | Regenerates gold function labels for 120 tasks from the dataset. The output is not redistributed. |

### Competition rules that shaped the release

Competition data may not be redistributed, so this notebook regenerates everything from the attached dataset. Only derived metrics are kept as outputs. Code is public on Kaggle under Apache-2.0. Dependencies are limited to numpy, matplotlib, scikit-learn and pytest, all preinstalled on Kaggle.

### Threats to validity

- Token costs are estimates (characters / 3.6).
- Gold labels come from text matching.
- Four Python libraries, mostly fastapi.
- The subgroup analysis is post hoc.
- No resolve-rate measurement.
