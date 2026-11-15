# Deliverables 0 and 1 — Feasibility, Novelty, and Research Plan

Prepared 2026-11-07. Deadline: 2026-11-12 23:59 UTC (= 2026-11-13 05:29 IST). **Time remaining: ~5.5 days.**

Status of the asset audit: **provisional**. Everything marked `[VERIFY]` depends on the Kaggle documentation and file samples you will send next. No field names below are confirmed.

---

## Executive decision

| Item | Decision |
|---|---|
| Recommended approach | **Budgeted, multi-resolution, connected context selection** over the provided code graph ("Evidence-per-Token"). Evaluate it mainly as a **retrieval/localization problem at matched token budgets** over all 129 tasks (cheap, deterministic, no LLM), plus a **smaller end-to-end Gemma repair confirmation** (3 methods × 1 budget). |
| Why feasible | The retrieval half needs only graph JSON, embeddings, issue text, and gold patches (as labels). It runs on CPU in minutes. The end-to-end half needs ~400 single-shot Gemma generations plus test runs: a few GPU-hours, if the test harness works. |
| Main novelty claim | Not "graph retrieval for code" (exists: RepoGraph, LocAgent, CoSIL, OrcaLoca, KGCompass). The defensible claim is: (1) context selection posed as a **token-cost knapsack over a code graph where each node can enter at three resolutions** (full source / signature+docstring / one-line outline) under a connectivity constraint; (2) a **controlled matched-budget evaluation** that isolates the contributions of relevance, connectivity, compression, and failure signal for a small local model; (3) a reusable **budgeted-localization benchmark protocol** (recall-vs-token-budget curves) on the competition tasks. |
| Main risk | Two infrastructure unknowns, not the method: (a) whether the issue text can be embedded into the same 256-d space as node embeddings (encoder may not be provided); (b) whether per-task test execution works reliably in the time left. Both have fallbacks (below). |
| Minimum viable experiment | Localization Recall@B for {BM25, dense top-k, hybrid top-k, graph-expansion, budgeted-connected, budgeted-connected+tiers, oracle, random-connected} at B ∈ {2k, 4k, 8k, 16k} on all 129 tasks, task-level split for tuning, paired bootstrap CIs. This alone supports a credible paper even if end-to-end repair yields no significant difference. |
| Build first today | (1) Asset probe script: print schemas of graph JSON, `.npz`, task metadata; (2) gold-location extractor (patch hunks → graph nodes); (3) BM25 and dense top-k retrievers + Recall@B metric. These three give you the baseline curve by tonight. |

**Most important change from your plan:** with 5.5 days left, the 9-method × 4-budget **end-to-end** matrix (~4,600 agent runs plus test executions) is not achievable or necessary. Run the full matrix at the retrieval level and the end-to-end comparison on a reduced matrix. RL, learned pruning, and iterative refinement are cut.

---

# Deliverable 0 — Feasibility and Novelty Decision

## 0.1 Asset audit (provisional)

The table lists what the plan assumes, why it matters, and what to check. Treat every row as unconfirmed.

| Asset | Assumed form | Why it matters | Check `[VERIFY]` |
|---|---|---|---|
| Task list (129) | JSON/CSV: `instance_id`, `repo`, `base_commit`, `problem_statement`, `patch`, `test_patch`, `FAIL_TO_PASS`, `PASS_TO_PASS` (SWE-bench-like) | Query text; labels; evaluation | Exact field names; whether hints/comments are included; number of distinct repos |
| Repo snapshots | Checked-out source at `base_commit` | Source spans, token costs, patch application | Pre-patch state? Includes tests dir? |
| Code graph | JSON: nodes (id, type, name, file, start/end line), edges (type) | Expansion, connectivity, proximity | Node types (module/class/function/method/test?); edge types (calls, imports, contains, inherits?); call edges resolved or name-matched; line spans present |
| Node embeddings | `.npz`, 256-d float32, keyed by node id | Dense relevance | **Which encoder produced them, and is it available offline?** Computed on pre-patch code? Coverage (% nodes with vectors)? Normalized? |
| Graph tools | `search_similar_code`, `get_code_neighbors`, `get_code_subgraph` | May already implement query embedding | Does `search_similar_code` accept free text? If yes, it exposes the query encoder |
| Gold patches | Unified diffs | Localization labels, oracle | Only for labels/oracle |
| Test patches | Unified diffs | Evaluation only | Never in context |
| F2P / P2P | Test identifiers | Evaluation; optional "test-informed" setting | Do F2P tests exist in the pre-patch repo, or are they added by `test_patch`? |
| Execution env | Docker images / conda envs / none | End-to-end resolve rate | **Is a harness provided? Per-task runtime?** |
| Gemma checkpoint | Gemma 4 variant(s) on Kaggle Models | Agent | Size, context length, license, quantized variants, Kaggle GPU fit |

Linkage questions to answer from the samples:
1. Graph node id ↔ `.npz` key: same string, or index alignment?
2. Node span ↔ source file: can `(file, start_line, end_line)` reproduce the code text exactly?
3. Gold patch hunk ↔ node: map pre-image changed line ranges onto node spans (innermost function/method containing the line; fall back to class, then file).
4. Test ↔ function mapping: almost certainly **not provided**; must be inferred from call edges out of test nodes (static, approximate).
5. Stack traces: not a separate field in SWE-bench-style data; they appear inside some `problem_statement`s. Parse them from issue text (leakage-free).

## 0.2 Feasibility gate (the ten questions)

| # | Question | Answer |
|---|---|---|
| 1 | Implementable with provided assets only? | **Yes**, conditional on resolving query embedding. Fallbacks if the encoder is absent: (a) BM25 seeds, then use node embeddings only for node–node similarity (seed centroid as pseudo-query); (b) fit a ridge map from a small local text encoder to the 256-d node space using (node code text, node vector) pairs from the repos — no labels needed, no leakage. |
| 2 | Within Kaggle compute/time? | Retrieval experiments: yes (CPU, minutes). End-to-end: ~130 tasks × 3 methods × ~1–2 min generation ≈ 6–13 GPU-hours plus test time. Fits a Kaggle weekly GPU quota or your HPC. Estimate is unverified until the Gemma variant is known. |
| 3 | Evaluate offline on all 129? | Retrieval: yes. End-to-end: only if the test environment runs per task. Plan for partial coverage and report it. |
| 4 | Fair comparison vs top-k? | Yes, by construction: identical token budget (counted with the Gemma tokenizer), identical prompt template, same model, greedy decoding, single-shot generation. Only the context packet varies. |
| 5 | Avoid leakage? | Yes, with rules in §0.5. Two real hazards: embeddings computed on post-patch code, and F2P test names. Both are checked or isolated. |
| 6 | Generate and apply patches locally? | Yes: request search/replace edit blocks (more reliable for small models than raw unified diffs), convert to diff, `git apply --check`, then run F2P/P2P. |
| 7 | Enough results for ablations? | Yes at the retrieval level (cheap, all configs). End-to-end ablations: no — one budget, three methods. State this. |
| 8 | Three likely failure modes | (i) Graph adds nothing over hybrid top-k because gold edits are usually in the single most similar function (then connectivity spends budget on neighbours that are never edited). (ii) End-to-end differences are within noise at n=129 (resolve rates for a small local model may be ~5–20%, giving few discordant pairs). (iii) Test harness fails for a large fraction of tasks. |
| 9 | Minimum viable experiment | Retrieval-level matrix with Recall@B curves, paired bootstrap, task-level dev/test split (see Deliverable 1, Tier A). |
| 10 | Cut first if constrained | In order: learned pruning → iterative refinement → depth/seed sweeps beyond defaults → 16k budget end-to-end → no-retrieval end-to-end baseline → test-informed setting. Never cut: dev/test split, oracle and random controls, Recall@B on all tasks. |

## 0.3 Risk register

| Risk | Likelihood | Impact | Mitigation | Detect by |
|---|---|---|---|---|
| No query encoder for the 256-d space | Medium | High | Ridge-mapped encoder, or BM25-seed centroid | Day 0 asset probe |
| Embeddings built on post-patch code | Low–medium | Critical (invalidates dense results) | Recompute node text hash vs pre-patch source; if mismatch, drop dense or re-embed | Day 0 |
| Call edges unresolved / sparse | Medium | Medium | Fall back to `contains` + `imports` + same-file edges; report edge-type ablation | Day 0 degree stats |
| Graph helps nothing | Medium | Medium (narrative, not validity) | Report as a negative/conditional result; analyse which task types benefit | Day 1 gate G1 |
| Test harness unreliable | Medium–high | High for end-to-end | Report patch-apply rate + localization; run end-to-end on the runnable subset, stated explicitly | Day 2 gate G2 |
| Gemma context/latency too limited | Low–medium | Medium | Cap at 8k; quantized model; single-shot | Day 2 smoke test |
| Tuning on test tasks | Medium if rushed | High (judges will check) | Fixed dev split; freeze config before test run; log commit hash | Process |
| Word limit (3,000) | High | Low | Write results tables first; prose last | Day 4 |

## 0.4 Novelty audit

### What already exists (verified sources unless marked)

| Area | Closest work | What it does | Overlap with our idea | Gap we can claim |
|---|---|---|---|---|
| Repo-level code graph as plug-in context | **RepoGraph** (Ouyang et al., ICLR 2025, arXiv:2410.14684) | Line-level graph of definitions/references; retrieves k-hop ego-graphs around search terms; plugs into SWE-agent, AutoCodeRover, Agentless. Argues similarity-only retrieval finds "semantically similar but not genuinely related" code. | High: same motivation and same "graph context vs similarity" contrast. | RepoGraph does not enforce a token budget or choose per-node resolution; does not report matched-budget comparisons against top-k. **We must cite it as the primary precedent and not claim the motivation as new.** |
| Graph-guided localization agents | **LocAgent** (Chen et al., arXiv:2503.09089) | Heterogeneous graph (files/classes/functions; imports, invocations, inheritance); agent does multi-hop search; fine-tuned Qwen-32B. | Graph + small open model for localization. | LocAgent is an LLM-driven search policy (many calls). Ours is a deterministic, zero-LLM-call selector suited to consumer hardware. |
| LLM-driven call-graph search | **CoSIL** (arXiv:2503.22424, ASE 2025) | Builds module and function call graphs during search; pruner + reflection; Qwen2.5-Coder-32B. | Call-graph expansion with pruning. | Same as LocAgent: LLM in the loop; no explicit token knapsack. |
| Priority-queue exploration + distance pruning | **OrcaLoca** (Yu et al., ICML 2025, arXiv:2502.00350) | Priority-scheduled agent exploration; **distance-aware context pruning** keeps top-k by graph distance. | Closest to "distance-aware pruning". | Their pruning is top-k by distance inside an agent loop; ours is a cost-normalized, connected, multi-resolution selection with a hard token budget. Cite and contrast directly. |
| Knowledge graph for repair | **KGCompass** (Yang et al., arXiv:2503.21710) | KG linking issues/PRs to code entities; reports most localized bugs need multi-hop paths. | Multi-hop structure helps. | Uses repository history artifacts; ours uses only the AST graph. |
| Learned retrieval + rerank for localization | **SweRank** (Reddy et al., arXiv:2505.07849, ICLR 2026) | Bi-encoder retriever + LLM reranker; SWELoc dataset. | Strong flat-ranking baseline family. | Flat ranking; no structure or budget. Useful to cite as evidence that flat ranking is strong — our top-k baseline must be taken seriously. |
| Simple pipelines | **Agentless** (Xia et al., arXiv:2407.01489) | Hierarchical localization (file → element → line) then repair; no agent loop. | Our single-shot design follows this "pipeline, not agent" rationale. | Cite as design precedent. |
| Connected-subgraph retrieval for RAG | **G-Retriever** (He et al., arXiv:2402.07630) | Retrieval as **Prize-Collecting Steiner Tree**: connected subgraph maximizing node prizes minus edge costs. | **Very close to "budgeted connected subgraph".** | PCST penalizes edge costs, not a hard token budget; it has no multi-resolution node inclusion; applied to textual KGs, not code. **You must cite it and must not claim connected-subgraph selection as new.** Consider PCST itself as an extra baseline if time permits (`pcst_fast`). |
| LLM fault localization from a failing test | **AutoFL** (Kang, An, Yoo, FSE 2024, arXiv:2308.05487) | LLM navigates coverage-derived functions from a single failing test. | Test-driven localization. | Requires runtime coverage. Ours uses static test→code call paths only; weaker but cheaper. Cite as the dynamic counterpart. |
| Spectrum-based fault localization | Tarantula, Ochiai; Wong et al. survey (TSE 2016) `[to verify exact citations]` | Rank statements by pass/fail coverage spectra. | Conceptual source of "failure propagation". | We do **not** run tests for localization; our T term is a static proxy. Say so. |
| Budgeted selection | Budgeted maximum coverage (Khuller, Moss, Naor 1999) `[to verify]`; submodular knapsack greedy | Greedy by marginal gain per cost. | Our greedy is this heuristic. | Not a contribution. Cite as the source of the heuristic. |
| Long-context degradation | "Lost in the Middle" (Liu et al., TACL 2024, arXiv:2307.03172) `[to verify]` | Models use mid-context information poorly. | Motivates small, ordered packets. | Motivation only. |
| Benchmarks / agents | SWE-bench (arXiv:2310.06770), SWE-agent (arXiv:2405.15793), AutoCodeRover (arXiv:2404.05427) `[to verify IDs]` | Task format and agent baselines. | Task format. | Background. |
| RL for SWE | SWE-RL (arXiv:2502.18449), SWE-Gym (arXiv:2412.21139) `[to verify IDs]` | RL / trajectory training for SWE agents. | Context for "why not RL". | Background only. |

### Novelty positioning

- **Genuinely new (defensible if results support it):**
  1. Context construction as a **multi-choice knapsack with connectivity** on a code graph: each node has three candidate representations with different token costs and different evidence value; the selector upgrades/downgrades resolution rather than including/excluding nodes only.
  2. A **matched-token-budget protocol** that separates four factors (dense/lexical relevance, connectivity, multi-resolution compression, static failure signal) and reports **Recall-vs-budget curves** for a local model's context window. Existing graph papers report gains at their own (usually uncontrolled) context sizes.
  3. An **empirical answer** for small local models: at what budget does structure stop helping?
- **Adaptation of known methods:** greedy marginal-gain-per-cost selection; k-hop graph expansion; BM25 + dense hybrid; static call-path proximity; Agentless-style single-shot repair.
- **Strongest defensible claim (draft):** "Under a fixed prompt-token budget, selecting a connected, multi-resolution subgraph of the code graph retrieves more of the code that the reference fix edits than flat top-k retrieval, with the largest gap at small budgets; [end-to-end effect: TO BE FILLED]." Only keep the comparative half if Tier A results support it.
- **Claims to soften or drop:** "optimal"; "failure propagation" (rename to **static test-path proximity** unless you use runtime data); "rivals cloud models"; "first graph-based context for SWE agents"; any statement that connectivity *causes* repair success (localization recall is a mediator you measure, not a proven cause).
- **Must cite:** RepoGraph, G-Retriever, OrcaLoca, LocAgent, CoSIL, Agentless, SWE-bench, SweRank, AutoFL.

### Recommended framing for prize categories

The protocol + harness (Recall@B curves on the 129 tasks, with oracle and random controls, released as a public notebook) is a candidate for **Best New Resource / New evaluation task**, independent of whether the method wins. This hedges against failure mode (i).

## 0.5 Leakage and validity rules (binding)

| Information | Context selection (main setting) | "Test-informed" setting | Labels / analysis only |
|---|---|---|---|
| Issue title/body (incl. stack traces in it) | ✅ | ✅ | ✅ |
| Repo snapshot at base commit, graph, embeddings (pre-patch) | ✅ | ✅ | ✅ |
| F2P test **names** | ❌ | ✅ only if the test node exists in the pre-patch graph; clearly labelled | ✅ |
| `test_patch` content | ❌ | ❌ | evaluation only |
| Gold patch | ❌ | ❌ | labels, oracle |
| P2P/F2P outcomes | ❌ | ❌ | evaluation only |

Note: in SWE-bench-style data, F2P tests are frequently **added** by `test_patch`, so they will not be in the pre-patch graph. Expect the test-informed setting to cover only a subset; report its coverage. The deployable setting uses only the issue text, including any stack-trace frames it contains.

Tuning protocol: stratified (by repo) split into **dev = 30 tasks** and **test = 99 tasks**, fixed seed, list published. All hyperparameters chosen on dev; config frozen and hashed before test runs. Report dev and test separately. If there are few repos, also report a leave-one-repo-out check on retrieval metrics (cheap).

## 0.6 Go / no-go

**GO** on the main method, with the evaluation re-scoped: full matrix at the retrieval level, reduced matrix end-to-end. **Default fallback:** pure heuristic graph packing (no learned parts) — which is already the main method. **Second fallback** (if G1 fails): pivot the paper to "When does code-graph structure help budgeted context for local SWE agents? A matched-budget study," keeping the same harness and data; this is a legitimate negative-results paper plus resource.

Alternatives from your list, triaged for the time left:

| Alternative | Keep? | Reason |
|---|---|---|
| 1. Pure heuristic graph packing | **Main method** | Deterministic, CPU-only, reproducible |
| 2. Learned node pruning | **Cut** (optional Day 3 if ahead) | 129 tasks → ~100 train tasks across CV folds; small gain likely within noise; adds leakage surface |
| 3. Semantic compression (tiers) | **Keep — integrated** into the main method as multi-resolution choice; summaries = signature + docstring, no LLM summaries (too slow, nondeterministic) |
| 4. Hybrid BM25 + dense | **Keep — as seed retrieval and as a baseline** | Likely the strongest flat baseline; omitting it would be a judge objection |
| 5. Iterative refinement | **Cut** | Requires multi-turn agent runs; doubles end-to-end cost |

---

# Deliverable 1 — Final Research Plan

## 1.1 Research questions

- **RQ1 (structure).** At a fixed prompt-token budget B, does connected graph-based selection retrieve more gold-edit locations than flat BM25, dense, and hybrid top-k?
- **RQ2 (components).** How much of any gain comes from (a) graph expansion, (b) the connectivity constraint, (c) multi-resolution compression, (d) static test-path / stack-trace proximity?
- **RQ3 (budget).** How does the gap change across B ∈ {2k, 4k, 8k, 16k}?
- **RQ4 (downstream).** Does higher localization recall at fixed B translate into higher patch-apply and resolve rates for a single-shot local Gemma model?

## 1.2 Hypotheses and falsification criteria

| H | Statement | Falsified if (on the test split) |
|---|---|---|
| H1 | Full method > hybrid top-k on function-level Recall@B at B=4k and 8k | Paired bootstrap 95% CI of the difference includes 0 at both budgets |
| H2 | The advantage shrinks as B grows | Gap at 16k ≥ gap at 2k |
| H3 | Multi-resolution tiers add recall beyond connected selection with full-source-only nodes | CI of (tiers − no tiers) includes 0 |
| H4 | Static test-path/stack proximity helps on tasks where the signal exists | No gain on the subset with signal |
| H5 | Higher recall → higher resolve rate | Resolve-rate difference (McNemar / paired bootstrap) not distinguishable from 0. **Expected to be underpowered at n≈99–129; report as exploratory.** |

## 1.3 Contributions

- **Main:** Evidence-per-Token selection — a deterministic, token-budgeted, connected, multi-resolution context selector for code graphs, requiring no LLM calls.
- **Secondary:** (1) a matched-budget evaluation protocol and public notebook for budgeted localization on the 129 tasks (Recall@B, Evidence Density, AUC over budgets, oracle and random controls); (2) a component ablation; (3) an end-to-end check with a local Gemma model; (4) a failure taxonomy of where structure hurts.

## 1.4 Method specification (for implementation in Deliverable 2)

Notation: graph G=(V,E); for node v: embedding e_v, source text, three representations r ∈ {full, sig, outline} with token costs c_v^r (Gemma tokenizer; c^outline < c^sig < c^full); budget B.

**Node score (all components z-normalized per task over the candidate pool):**

s(v) = α·z(cos(q, e_v)) + α_L·z(BM25(Q, v)) + β·P(v) + γ·T(v)

- P(v) = exp(−d(v, Seeds) / τ), d = shortest undirected hop distance over allowed edge types.
- T(v) = exp(−d_call(F → v) / τ_T), F = nodes named in issue stack-trace frames (main setting) or F2P test nodes present pre-patch (test-informed setting); T(v)=0 if F=∅.
- Defaults before tuning: α=1, α_L=1, β=0.5, γ=0.5, τ=1, τ_T=2. Tune on dev only, small grid.

**Value of a representation:** value(v, r) = max(s(v), 0) · w_r with w_full=1.0, w_sig=0.45, w_outline=0.1 (tuned on dev). Upgrading v from r to r' has gain value(v,r') − value(v,r) and cost c_v^{r'} − c_v^r.

**Selection (greedy, deterministic):**
1. Pool: top-K_seed (default 5) hybrid-scored nodes → seeds; candidate pool = seeds ∪ h-hop neighbours (h ≤ 2).
2. Insert seeds at `outline` first (guarantees coverage), then repeatedly pick the action with maximal gain/cost among: add a frontier neighbour (adjacent to selected set) at some resolution, or upgrade a selected node. Ties broken by node id.
3. Stop when no action fits in remaining budget. Connectivity: selected set has ≤ K_seed components (each rooted at a seed). Ablation: unconstrained.
4. Always include file headers (path + imports outline) for files with a Tier-1 node, charged to the budget.

**Packet format:** issue → (optional) stack frames → graph outline (indented tree: file → class → function, edges as `calls:` lines) → Tier-1 full source → Tier-2 signatures/docstrings → instructions. Total counted with the actual prompt template; hard cap enforced after formatting.

**Repair (single-shot, Agentless-style):** one generation per task, greedy decoding, search/replace edit blocks, max N output tokens fixed across methods. No tool calls. This removes agent-loop variance and isolates the context effect. (Multi-turn agent use is a stated limitation.)

## 1.5 Baselines and ablations

All at identical B, same tokenizer, same template skeleton.

| ID | Method | Tier A (retrieval, all 129) | Tier B (end-to-end) |
|---|---|---|---|
| M0 | No retrieval (issue + repo file tree only) | — | optional |
| M1 | BM25 top-k (functions, full source) | ✅ | — |
| M2 | Dense top-k | ✅ | — |
| M3 | Hybrid top-k (BM25+dense) | ✅ | ✅ **main baseline** |
| M4 | Hybrid seeds + h-hop expansion, ranked, truncated at B (no connectivity, no tiers) | ✅ | — |
| M5 | Budgeted connected selection, full-source only | ✅ | — |
| M6 | M5 + multi-resolution tiers | ✅ | — |
| M7 | M6 + T (stack/test proximity) = **full method** | ✅ | ✅ |
| M8 | Random connected subgraph from same seeds, same B | ✅ | — |
| M9 | Oracle (gold-location nodes first, then M7 fill) | ✅ | ✅ (diagnostic upper bound) |
| M10 | PCST (G-Retriever style) on same scores | optional, if `pcst_fast` installs cleanly | — |

Retrieval-level sweeps (dev for choosing, test for reporting): B ∈ {2k,4k,8k,16k}; h ∈ {1,2,3}; K_seed ∈ {1,3,5,10}; tier weights; edge-type subsets (calls only / +contains / +imports).

End-to-end matrix: {M3, M7, M9} × B=8k (add 4k if time) on test split (99) or all 129 if dev-tuning used only retrieval metrics — decide and state. ≈ 300–400 generations.

## 1.6 Metrics

Gold locations L* = set of graph nodes whose pre-image span contains a changed line of the gold patch (innermost function/method; class/module fallback). New functions added by the patch are counted at file level only.

| Metric | Definition | Level |
|---|---|---|
| **Recall@B (func)** | \|L* ∩ Tier1(S)\| / \|L*\| — gold functions present with full source | primary retrieval |
| Recall@B (any tier) | gold function present at any resolution | secondary |
| File Recall@B | gold files with ≥1 selected node | secondary |
| Hit@B | 1 if ≥1 gold function in Tier 1 | task-level, for paired tests |
| Evidence density | gold-function tokens in packet / packet tokens | efficiency |
| Recall-AUC | area under Recall@B over log₂ B ∈ [2k,16k] | summary |
| Precision@Tier1 | \|L* ∩ Tier1\| / \|Tier1\| | secondary |
| Packet stats | nodes, edges, files, components, compression ratio (packet tokens / tokens of all files touched) | descriptive |
| Patch-apply rate | `git apply --check` succeeds | end-to-end |
| Resolve rate | all F2P pass and all P2P pass | **primary end-to-end** |
| F2P pass rate, P2P preservation | fractions | end-to-end |
| Prompt tokens, output tokens, latency | counted, wall-clock | cost |
| Edit locality | edited files/functions ⊆ selected? | diagnostic |
| Failure taxonomy | {not localized, localized-wrong edit, malformed edit, apply fail, test regression, timeout} | analysis |

Statistics: paired bootstrap (10,000 resamples over tasks) for metric differences; McNemar exact test for Hit@B and resolve rate; report n, discordant counts, 95% CIs. No significance claims without these. Note the power limit explicitly.

## 1.7 Timeline (5.5 days, with gates)

| Day | Date (UTC) | Work | Exit criterion |
|---|---|---|---|
| D0 | Nov 7 (today) | Asset probe; loaders; gold-location extractor; tokenizer cost function; BM25 + dense + hybrid top-k; Recall@B | Baseline Recall@B curves for M1–M3 on all 129; dev/test split frozen |
| D1 | Nov 8 | Graph expansion (M4), greedy budgeted selector (M5), tiers (M6), stack/test proximity (M7), random (M8), oracle (M9); dev tuning | **Gate G1:** M6/M7 vs M3 on dev. If no gain → pivot narrative (§0.6), keep harness |
| D2 | Nov 9 | Packet formatter + prompt; Gemma runner; edit-block parser; patch apply; test runner; smoke on 10 tasks; launch Tier B in background | **Gate G2:** harness runs ≥80% of smoke tasks. If not → end-to-end on runnable subset only, stated |
| D3 | Nov 10 | Finish Tier B; Tier A test-split runs and sweeps; bootstrap/McNemar; failure taxonomy on ~30 sampled tasks; figures | All numbers in the paper exist in logs |
| D4 | Nov 11 | Write paper from tables; public notebook (retrieval reproduction runs on CPU in one click) | Draft ≤3,000 words |
| D5 | Nov 12 | Judge-risk review; claim pruning; reference check; **submit by 18:00 UTC** | Submitted with 6h buffer |

## 1.8 Minimum viable paper vs full target

| | Minimum viable (must have by D3) | Full target |
|---|---|---|
| Retrieval | M1–M9 at 4 budgets, all 129 tasks, dev/test, bootstrap CIs | + sweeps, edge-type ablation, PCST baseline, per-repo breakdown |
| End-to-end | M3 vs M7 at 8k on runnable subset; patch-apply + resolve rates | + M9 oracle, 4k budget, M0 |
| Analysis | Failure taxonomy on 30 tasks; where structure hurts | + qualitative packet examples, correlation of recall and resolve (labelled as correlational) |
| Resource | Public notebook reproducing Tier A from cached artifacts | + packaged library with CLI |

## 1.9 What I need from you next

To turn the provisional audit into a confirmed one, send:
1. Dataset documentation and the directory listing of the competition data.
2. One task record (all fields), one graph JSON (or the first ~50 nodes and edges), and `np.load(...).files` plus shapes and dtypes for one `.npz`.
3. The starter notebook, especially how `search_similar_code` gets a query vector, and how tests are executed.
4. Which Gemma 4 checkpoint(s) you intend to use and your hardware (Kaggle GPU type or HPC GPU).

Once I have these, I will confirm or revise every `[VERIFY]` row, then move to Deliverables 2–3.

---

## Source log

| Source | URL | Establishes | Effect on method | Novelty impact |
|---|---|---|---|---|
| RepoGraph (ICLR 2025) | https://arxiv.org/abs/2410.14684 | Repo-level code graph as plug-in context for SWE agents; ego-graph retrieval | Primary precedent; RepoGraph-style k-hop retrieval ≈ our M4 | **Weakens** motivation novelty; budget + resolution remain open |
| LocAgent | https://arxiv.org/abs/2503.09089 | Heterogeneous code graph + agent multi-hop search; fine-tuned 32B open model | Edge-type design reference | Weakens "graph for localization"; supports "deterministic, no-LLM selector" as a distinct niche |
| CoSIL (ASE 2025) | https://arxiv.org/abs/2503.22424 | LLM-built call graphs with pruning for function-level localization | Call-graph expansion precedent | Weakens "call-graph expansion" novelty |
| OrcaLoca (ICML 2025) | https://arxiv.org/abs/2502.00350 · https://proceedings.mlr.press/v267/yu25x.html | Distance-aware context pruning within agent exploration | Closest pruning precedent; contrast explicitly | Weakens "distance-based pruning"; budget/resolution remain open |
| KGCompass | https://arxiv.org/abs/2503.21710 | KG with issues/PRs + code entities; multi-hop paths frequent among localized bugs | Supports structural hypothesis | Neutral |
| SweRank (ICLR 2026) | https://arxiv.org/abs/2505.07849 | Strong flat retrieve-and-rerank localization | Our top-k baseline must be strong (hybrid) | Raises the bar |
| Agentless | https://arxiv.org/abs/2407.01489 | Simple localize-then-repair pipeline competitive with agents | Justifies single-shot design | Neutral |
| G-Retriever | https://arxiv.org/abs/2402.07630 | Connected subgraph retrieval via PCST for graph RAG | Candidate extra baseline (M10) | **Weakens** "connected subgraph selection" novelty; must cite |
| AutoFL (FSE 2024) | https://arxiv.org/abs/2308.05487 | LLM fault localization from one failing test using coverage | Dynamic counterpart to our static T term | Forces renaming "failure propagation" → static proximity |
| Competition info (third-party; not official) | https://internshala.com/competitions/gemma-4-developer-agent-paper-track-2026/ · https://aiwiki.ai/wiki/gemma_4_developer_agent_competition | Deadline (Nov 13 05:29 IST = Nov 12 23:59 UTC), 3,000 words, 256-d vectors, directed code graphs | Consistent with your brief | Verify against the official Kaggle page |

Unverified references (to verify before citing): SWE-bench (arXiv:2310.06770), SWE-agent (arXiv:2405.15793), AutoCodeRover (arXiv:2404.05427), Lost in the Middle (arXiv:2307.03172), SWE-RL (arXiv:2502.18449), SWE-Gym (arXiv:2412.21139), Khuller–Moss–Naor 1999 (budgeted maximum coverage), Wong et al. 2016 fault-localization survey (IEEE TSE), BM25 (Robertson & Zaragoza 2009).

---

# Revision 1 (2026-11-07) — supersedes earlier sections where they conflict

Inputs: official competition text (now confirmed: deadline 2026-11-12 23:59 UTC; judging panel listed) and an external review of v0.

## R1.1 Judging panel → what to emphasise

The panel (per the official citation block) includes graph-ML researchers (Perozzi, Galkin, Rózemberczki, Markowitz, Li) and a software-testing researcher (Hemmati). Consequences:

- **Graph serialization is a judged-relevant variable.** Perozzi's group showed that how a graph is written as text changes LLM graph-reasoning accuracy substantially ("Talk like a Graph", Fatemi, Halcrow, Perozzi, ICLR 2024, arXiv:2310.04560; GraphToken, arXiv:2402.05862). Our packet's "graph outline" is exactly such an encoding. Add a small ablation: outline encoding ∈ {none, edge list, indented containment tree + `calls:` lines} at fixed B. Cite both papers.
- **Fault-localization rigor will be checked.** Keep SBFL/AutoFL citations, the static-vs-dynamic distinction, and the leakage gate prominent.

## R1.2 New middle tier: model-in-the-loop localization (Tier A.5)

Gap in v0: Tier A measures whether evidence is *in* the packet; Tier B measures repair. Nothing measures whether a small model can *use* the evidence, and Tier B depends on the riskiest infrastructure (test execution).

Tier A.5: give Gemma the packet and ask for a ranked list of ≤5 functions to edit (JSON, ~100 output tokens). Score Acc@1/Acc@5 against gold functions. No patch, no tests.

- Cost: 129 tasks × ~6 configs ≈ 800 short generations; much cheaper than Tier B.
- Isolates "evidence present but not used" — the graph-encoding ablation (R1.1) runs here.
- Survives a failed test harness (Gate G2).

## R1.3 Headline figure: loss waterfall

For each method, attribute every task to the first stage where it fails:
retrieval miss (Tier A) → model mis-localizes despite evidence (Tier A.5) → malformed/unappliable edit → tests fail → resolved.
This decomposes the oracle gap into where a local agent loses, which is more informative than a single resolve rate and robust to small n.

## R1.4 Metric definitions (replaces §1.6 primary rows)

- Gold functions L* from pre-image changed lines (innermost enclosing function/method). Selection never reads L*.
- **Hit@B**: ≥1 gold function in Tier 1 (full source). **Recall@B**: fraction of gold functions in Tier 1. Report both; Hit@B is the paired-test unit.
- Any-tier variants and File Recall@B are diagnostics. File recall is never primary (a selector can inflate it with large files).
- **Evidence density** = gold-function tokens / packet tokens.
- Dropped from the external review: "connected-context recall" — trivially equal to Recall@B when connectivity is enforced.

## R1.5 Connectivity definition (replaces §1.4 step 3)

Seed-rooted connectivity: every selected node is reachable from some seed through selected nodes in the undirected graph over allowed edge types. Default allows up to K_seed components (one per seed). Ablation: single component (forces bridging paths; tests whether bugs spanning distant code are hurt).
Reason for not defaulting to one component: multi-file fixes whose locations are far apart in the graph would force budget onto bridging nodes.

## R1.6 Tuning protocol (refines §0.5)

- Tier A: 5-fold task-level CV, stratified by repo; every task gets an out-of-fold score, so all 129 count. Small grid only.
- Tiers A.5/B: one config chosen on the 30-task dev split, frozen and hashed; report on the remaining 99 (also all 129, labelled).

## R1.7 Gates (quantitative; replaces §1.7 gates)

| Gate | When | Pass condition | Action on fail |
|---|---|---|---|
| G0 Leakage | before any reported number | selector inputs audited: no gold paths/diff/function names/lines; embeddings match pre-patch source (hash check on sample) | fix before proceeding |
| G1 Retrieval | end Nov 8 | out-of-fold ΔRecall@B (M7 − M3) > 0 at ≥3 of 4 budgets and paired-bootstrap 95% CI of Recall-AUC difference excludes 0 | pivot to "when does structure help" study |
| G2 Harness | end Nov 9 | test runner completes on ≥80% of 10 smoke tasks | Tier B on runnable subset, coverage disclosed; lean on A.5 + waterfall |

A fixed "2–3 pp on 30 tasks" threshold is not used: 1 task = 3.3 pp, so it is noise.

## R1.8 Tier B matrix (replaces §1.5 Tier B column)

{M3 hybrid top-k, M7 full, M9 oracle} × B=8k on all runnable tasks (≈390 runs). Add M4 graph-expansion-only only if the first three finish by Nov 10 noon UTC.

## R1.9 Resource framing (Best New Resource)

Release, as a Kaggle dataset + notebook: per task × budget × method, the packet, its token count, selected node ids, Tier-A labels, and Tier-A.5 Gemma outputs. The official text names "a dataset that leverages the provided embeddings for evaluation or fine-tuning of Gemma 4" as an example resource; these packets with gold localization labels are directly usable as localization SFT data for the main competition. Do not fine-tune within this deadline; state it as a use.

## R1.10 Claims removed from the original Gemini proposal

"Mathematically guaranteed" token count (it is enforced, not guaranteed-optimal); "3D lattice"; "local 9B rivals cloud"; "30–40% VRAM reduction" (VRAM is dominated by weights at these context sizes; report prompt tokens, latency, and measured peak VRAM only if measured).

---

# Revision 2 (2026-11-07) — constraints from the main-competition rules

Source: official rules of "Google - The Gemma 4 Developer Agent Competition" (featured track), pasted by the user. The paper track's own rules page has not been seen; assume these rules apply to shared Competition Data unless the paper-track page says otherwise `[VERIFY]`.

| Rule | Consequence for this project |
|---|---|
| 2.4.b Data security: do not publish or redistribute Competition Data to non-participants | **Revises R1.9.** Do not upload packets containing source code, issue text, or gold patches as a standalone public dataset. Release instead: a public Kaggle notebook that attaches the competition dataset as input and regenerates packets, plus a derived table of task ids, node ids, scores, token counts, labels, and metrics. Ask on the competition forum whether derived datasets are allowed before publishing anything more. |
| 3.6.a/b Public code must be shared on Kaggle (forum or notebooks for this competition); deemed OSI-licensed | Primary code location = public Kaggle notebook linked from the Writeup. Any GitHub mirror is secondary. Add an Apache-2.0 header. No private sharing of code outside your Kaggle team. |
| 3.6.c Open-source dependencies must be OSI-licensed with no commercial-use limits | Dependency whitelist: numpy, scipy, networkx, rank_bm25, unidiff, tiktoken-free tokenizer from the Gemma checkpoint. Check `pcst_fast` licence before using M10. If the ridge-mapped query encoder fallback is needed, use an Apache/MIT text encoder, not CC-BY-NC. |
| 2.6 External data/models must be publicly available at minimal cost | No paid APIs in the pipeline. Gemma via Kaggle Models. |
| 3.4.b No hand labelling or human prediction of validation/test records | Failure taxonomy hand-labels apply only to the 129 public tasks, used for analysis. Never hand-label or inspect hidden leaderboard tasks. Keep the pipeline fully automatic. |
| 3.18.a Competition Data contains private and public test sets | The paper reports only on the 129 public tasks. State this. Never claim leaderboard generalization from them. |
| 2.1 Team ≤ 5; 3.5 one Kaggle account per person | Writeup co-authors must be team members on Kaggle. |
| 2.2 One main-competition submission per day | Affects only the main track. If you enter it, the selector can be reused there; that is optional and separate from the paper. |
| 2.7 Data use permitted for academic research | Reporting derived statistics and short code excerpts in the paper is fine. Keep code excerpts in the Writeup short. |

---

# Revision 3 (2026-11-10) — confirmed asset audit from the competition zip

Source: `C:\Users\Adarsh\Desktop\gemma4\gemma-4-developer-agent.zip` (21.9 GB). Extracted (small parts only) to `C:\Users\Adarsh\Desktop\gemma4\data\`: `HARNESS_README.md`, `tasks.jsonl`, `graphs/`, `embeddings/`, `sample_submission/` configs, `docker/`, `sandbox/`. Not extracted: `snapshots/*.tgz` (bulk of the 21.9 GB) and 124 wheels.

## R3.1 What exists

| Asset | Confirmed form |
|---|---|
| `tasks.jsonl` | 129 rows. Fields: `instance_id, repo, base_commit, patch, test_patch, problem_statement, hints_text, created_at`. **No `FAIL_TO_PASS` / `PASS_TO_PASS` fields** — the harness derives required tests from `test_patch`. `hints_text` empty for all 129. |
| Repos | fastapi/fastapi 67, Textualize/rich 48, psf/requests 13, encode/httpx 1. Created 2023-07 → 2026-06. |
| Problem statements | **PR titles/descriptions, not bug reports** (e.g. "✨ Add support for Server Sent Events"; 9 contain PR templates). Median 418 chars. Only **1/129** contains a traceback. Many tasks are features/refactors, not bugs. |
| Gold patches | 91/129 touch one file; max 26 files. 11 add a new file. All touch ≥1 `.py`. |
| Graphs `graphs/<repo>_<commit>.json` | 127 files (two commits shared by two tasks each). NetworkX node-link JSON, `directed: true, multigraph: true`. Node fields: **only `id`, `name`, `text`** (dotted symbol path + source text; class bodies stubbed with `...`). **No file path, no line numbers, no node type field.** Edge fields: `source, target, type, key`; **the only edge type is `calls`** (it also encodes class→method containment). Size: fastapi ~3.3–4.6k nodes / ~2.4k edges (incl. `docs_src` tutorial modules); requests ~0.7k / 1.8k; rich ~2.0k / 5.6k. |
| Embeddings `embeddings/<repo>_<commit>.npz` | One array per node id (`np.load(...).files` = node ids), shape `(256,)`, float32, **not unit-normalized** (norm ≈ 3.7). 100% node coverage on sampled repos. Identical text → identical vector (text-derived). **Encoder unknown and not provided.** |
| Graph tools (harness) | `search_similar_code(query)` resolves `query` to an existing **symbol name** and returns its embedding neighbours; multi-word text falls back to lexical token overlap. There is **no free-text query encoder** in the harness either. |
| Harness | `swegemma` (+ `adk-submission`, `adk-eval-core`) — **documented in `HARNESS_README.md` but not included in the zip**. Docker image `swebench-sandbox`, network off, 4 GiB / 2 vCPU per container; resolved = pytest exit 0 and all tests from `test_patch` pass. |
| Models | Competition: `gemma-4-{31b,12b,e4b,e2b}-it-qat-w4a16-ct` on 4×L4 (96 GB), vLLM `max_model_len=32768`. ADK compaction at 14,336 tokens; tool outputs capped at 5,000 chars; `read_file` 150 lines. |
| This laptop | Python 3.10, numpy 1.26, networkx 3.4. **No Docker, no NVIDIA GPU.** |

## R3.2 Checks run

| Check | Result | Consequence |
|---|---|---|
| Leakage: are graphs pre-fix? | Removed (`-`) gold lines found in node text: 77.5% (mean over 99 tasks with removed lines). Genuinely new (`+`) lines found: 4.6% (119 tasks). Only 2 tasks >50% (`rich_4075`, `fastapi_14301`; likely moved code — inspect). | Graphs/embeddings reflect the base commit. **Gate G0 passes** for graph inputs. |
| Gold-location mapping (no line numbers available) | Gold nodes = nodes whose text contains a removed gold line (most specific id) ∪ functions named in hunk headers. Non-empty for **119/129**; the 10 empty are pure additions (new files/functions). Gold nodes per task: 1 → 29 tasks, 2 → 38, 3–5 → 24, 6+ → 28. | Function-level Recall@B is computable for 119 tasks; report the 10 as "addition-only" separately (file-level only). |
| How often the PR text names the answer | A gold symbol name appears in the problem statement in **25/129** tasks; a gold file stem in 33/129. | Retrieval is non-trivial; lexical baselines will not saturate. |

## R3.3 Plan changes forced by the audit

1. **Drop the test/failure term T entirely.** No F2P fields, no hints, 1/129 tracebacks; the only test signal is `test_patch`, which is evaluation-only (leakage). The method is relevance + connectivity + multi-resolution compression. Remove "failure"/"test-aware" from the title.
2. **Dense retrieval needs a pseudo-query.** No query encoder exists. Deployable options, all leakage-free: (a) **identifier anchoring** — extract identifiers/backticked names/dotted paths from the PR text, resolve to graph nodes (same 4-tier resolution as the harness), use their embeddings; (b) **BM25-seed centroid** — mean embedding of the BM25 top-m nodes; (c) ridge map from an open text encoder (only if (a)/(b) are weak; licence must be Apache/MIT). This is now a sub-question of the paper: how to use code-only embeddings when the query is natural language.
3. **"Connected" uses the single `calls` edge type**, undirected. Edge-type ablation is impossible; replace it with a direction ablation (callers-only / callees-only / both).
4. **Token costs** come from node `text` (function source). Tier definitions: full = node text; signature = first line(s) up to `:` + docstring; outline = dotted id only. Class nodes are already stubs.
5. **Retrieval corpus filter**: exclude `tests.*` nodes and (for fastapi) `docs_src` tutorial nodes from the candidate pool by default, and report the effect. Gold patches for fastapi docs tasks may touch `docs_src` — check before filtering (`[VERIFY]` on day 1).
6. **Tier B depends on infrastructure you do not have locally** (harness not in zip; no Docker/GPU here). Tier B is cut unless the harness already runs for you on Kaggle or HPC. Tier A.5 (Gemma localization probe) needs only a GPU with a Gemma 4 checkpoint (Kaggle T4×2 or HPC), no Docker.
7. **Generalization claim** must be scoped to four Python web/CLI libraries (fastapi-heavy: 52% of tasks). Report per-repo results.

## R3.4 Revised timeline (~65 h left at 2026-11-10 07:00 UTC)

| When (UTC) | Work | Exit |
|---|---|---|
| Nov 10 | Tier A pipeline on CPU: loaders, gold mapping, BM25, identifier-anchored dense, BM25-centroid dense, hybrid, graph expansion, budgeted connected selector, tiers, random, oracle; Recall@B at 2k/4k/8k/16k; 5-fold CV | Tier A tables + recall-vs-budget figure |
| Nov 11 | Tier A ablations + bootstrap CIs; Tier A.5 on GPU (one Gemma 4 model, 3 packet types × 129 tasks, short JSON outputs) | Waterfall (retrieval → model localization) |
| Nov 12 by 18:00 | Write paper (≤3,000 words), public Kaggle notebook reproducing Tier A from competition data, submit | Submitted with 6 h buffer |

---

# Revision 4 (2026-11-10) — Tier A outcome; Gate G1 failed → pivot

Full tables: `results/tier_a/summary.md`; figure: `results/tier_a/fig_recall_vs_budget.png`; diagnostics: `results/tier_a/diagnostics.csv`.

- Out-of-fold Recall-AUC (120 tasks with gold): BM25 0.489, Hybrid top-k 0.480, Hybrid+tiers 0.477, Graph expansion 0.460, Connected+tiers (EpT) 0.452, Connected full-only 0.394, Random connected 0.318, Dense (pseudo-query) 0.16–0.17, Oracle 0.937.
- EpT vs Hybrid top-k: ΔAUC −0.028 [−0.069, +0.012]. Connected full-only: −0.086 [−0.131, −0.043]. Tiers recover most of the connectivity loss and give the best file-level coverage (flat tiers File@4k 0.871 vs 0.790).
- The provided embeddings are weak for natural-language PR text (no query encoder): dense top-k recall is a third of BM25's.
- Diagnostics: 23% of pool nodes have no `calls` edges. When BM25 misses at 8k (32 tasks), the nearest gold node is 1–2 hops from the packet in 17 of them — the graph has headroom, but text-only relevance cannot rank the right neighbour.
- Oracle gap ≈ 0.45 AUC: localization, not packing, is the bottleneck.

G1 condition (EpT > hybrid at ≥3 budgets with CI excluding 0) is not met. Per the pre-registered plan the paper pivots to: "When does code-graph structure help budgeted context for local SWE agents? A matched-budget study on the Gemma 4 graphs", with the harness + protocol as the resource contribution.
