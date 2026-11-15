# Evidence-per-Token (EpT) — Tier A reproduction

Budgeted, connected, multi-resolution context selection over the Gemma 4 Developer Agent code graphs.
Tier A measures whether the code a reference fix edits ends up in a token-budgeted context packet. It uses CPU only and makes no LLM calls.

## Setup

- Python ≥ 3.10, `numpy`, `matplotlib`, `pytest` (tests only). No other dependencies.
- Competition data directory containing `tasks.jsonl`, `graphs/`, `embeddings/` (from the competition dataset; not redistributed here).

## Run

```bash
python -m evtok.run_tier_a --data <DATA_DIR> --out results/tier_a --workers 8   # ~3 min on 12 cores
python -m evtok.analyze --grid results/tier_a/grid.pkl --out results/tier_a         # CV selection + stats
python -m evtok.figures --csv results/tier_a/oof_per_task.csv --out results/tier_a
python -m pytest -q tests
```

## What uses what

| Step | Inputs | Never uses |
|---|---|---|
| Relevance (BM25, identifier anchors, dense pseudo-query) | `problem_statement`, graph node ids/text, embeddings | `patch`, `test_patch` |
| Selection (all selectors) | relevance scores, graph edges, token costs, budget | `patch`, `test_patch` |
| Gold labels, oracle, metrics | `patch` (evaluation only) | — |

Hyper-parameters are chosen by 5-fold task-level CV (stratified by repo, seed 0); every reported number is out-of-fold.

## Known approximations

- Token costs use `len(text)/3.6` characters per token. Calibrate against the Gemma 4 tokenizer before quoting absolute budgets.
- Gold nodes are found by matching removed/context lines of the reference patch to node source text (the graphs have no file paths or line numbers). 9 of 129 tasks only add new code and have no gold function nodes; they are reported at file level only.
- Node→file membership is inferred from dotted ids and the patched file path.

Code licence: Apache-2.0.
