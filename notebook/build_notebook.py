"""Build the public Kaggle notebook from the repository sources (single source of truth).

Usage: python notebook/build_notebook.py  ->  notebook/ept-tier-a-reproduction.ipynb
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULES = ["__init__", "textproc", "data", "gold", "retrieval", "selectors", "metrics", "run_tier_a", "analyze", "diagnostics", "figures", "extensions", "export_labels", "rl_select"]


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(keepends=True)}


def read(p):
    with open(os.path.join(ROOT, p), encoding="utf-8") as f:
        return f.read()


cells = [
    md("""
# Evidence-per-Token: Tier A reproduction

This notebook reproduces every number and figure in the paper *When does code-graph structure help a token-budgeted coding agent? A matched-budget study of context selection on the Gemma 4 Developer Agent graphs*.

**What it does.** It evaluates ten context selectors (BM25, dense, hybrid, graph expansion, connected, connected + multi-level packing, random and oracle controls) on all 129 public tasks at budgets of 2k, 4k, 8k and 16k estimated tokens. It reports function-level recall of the code edited by the reference fix. Every configuration is tuned by 5-fold task-level cross-validation, and every reported number is out-of-fold.

**Inputs.** Attach the competition dataset (*Google - The Gemma 4 Developer Agent Competition*) as an input. The notebook reads only `tasks.jsonl`, `graphs/` and `embeddings/`, and copies no competition data into its outputs except derived metrics.

**Leakage boundary.** Selectors read only `problem_statement`, the graphs and the embeddings. `patch` is used only for gold labels, the oracle and metrics. `test_patch` is never read.

**Runtime.** CPU only, no GPU or internet needed. About 45 to 55 minutes on a 4-core Kaggle CPU session: grid about 4 min, diagnostics about 4 min, robustness extensions about 6 min, RL (three seeds) about 30 min.

Code licence: Apache-2.0.
"""),
    code("""
import glob, os, subprocess, sys

cands = glob.glob("/kaggle/input/**/tasks.jsonl", recursive=True)
DATA = os.path.dirname(cands[0]) if cands else os.environ.get("EPT_DATA", "data")
assert os.path.exists(os.path.join(DATA, "tasks.jsonl")), f"tasks.jsonl not found under {DATA}; attach the competition dataset"
assert os.path.isdir(os.path.join(DATA, "graphs")) and os.path.isdir(os.path.join(DATA, "embeddings"))
WORKERS = max(1, (os.cpu_count() or 2))
print("data:", DATA, "| workers:", WORKERS)
os.makedirs("evtok", exist_ok=True)
os.makedirs("tests", exist_ok=True)

def run(*args):
    # Run an evtok module and stream its output live, so long steps show progress.
    import time
    t0 = time.time()
    print("running:", " ".join(args), flush=True)
    p = subprocess.Popen([sys.executable, "-u", *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="", flush=True)
    rc = p.wait()
    print(f"[exit code {rc}, {time.time() - t0:.0f}s]", flush=True)
    if rc != 0:
        raise RuntimeError(f"step failed with exit code {rc}; see the output above")
"""),
    md("""
## 1. Source code

Each cell below writes one module of the `evtok` package. The same files are in the paper's code release.

| Module | Role |
|---|---|
| `textproc` | identifier-aware tokenization, token-cost estimate, signature extraction |
| `data` | load tasks, graphs (node-link JSON), embeddings (`.npz` keyed by node id) |
| `gold` | map the reference patch to graph nodes (**evaluation only**) |
| `retrieval` | BM25, identifier anchors, dense pseudo-query, hybrid score (problem statement only) |
| `selectors` | all selectors; each guarantees packet cost ≤ budget |
| `metrics` | Recall@B, Hit@B, any-tier recall, file hit, components |
| `run_tier_a` | evaluate every configuration on every task |
| `analyze` | 5-fold CV selection, paired bootstrap, McNemar, tables |
| `diagnostics` | graph-distance headroom analysis |
| `figures` | recall-vs-budget figure |
| `extensions` | learned ranker, budget-split rescue, Steiner-connected top-k (Section 6.3) |
| `export_labels` | regenerate localization labels (evaluation set) |
| `rl_select` | REINFORCE context selection with token penalties (Section 6.4) |
"""),
]
for m in MODULES:
    cells.append(code(f"%%writefile evtok/{m}.py\n" + read(f"evtok/{m}.py")))
cells += [
    code("%%writefile tests/test_core.py\n" + read("tests/test_core.py")),
    md("## 2. Unit tests\n\nBudget bound, connectivity, test-node exclusion, patch parsing, tokenization."),
    code("""
run("-m", "pytest", "-q", "tests")
"""),
    md("## 3. Evaluate every configuration on every task\n\nWrites `results/tier_a/grid.pkl`. No gold information is used to build packets."),
    code("""
run("-m", "evtok.run_tier_a", "--data", DATA, "--out", "results/tier_a", "--workers", str(WORKERS))
"""),
    md("## 4. Cross-validated selection and statistics\n\nWrites `summary.md`, `summary.csv`, `oof_per_task.csv` and `cv_choices.json`."),
    code("""
run("-m", "evtok.analyze", "--grid", "results/tier_a/grid.pkl", "--out", "results/tier_a")
from IPython.display import Markdown, display
display(Markdown(open("results/tier_a/summary.md", encoding="utf-8").read()))
"""),
    md("## 5. Structural headroom diagnostics\n\nHow far are gold functions from what BM25 retrieved? Writes `diagnostics.csv`."),
    code("""
run("-m", "evtok.diagnostics", "--data", DATA, "--out", "results/tier_a")
"""),
    md("## 6. Subgroup analysis by number of gold functions (exploratory, post hoc)"),
    code("""
import csv, collections
import numpy as np
rows = list(csv.DictReader(open("results/tier_a/oof_per_task.csv", encoding="utf-8")))
d = collections.defaultdict(lambda: collections.defaultdict(list)); ng = {}
for r in rows:
    if r["n_gold"] == "0":
        continue
    d[r["method"]][r["instance_id"]].append(float(r["recall"])); ng[r["instance_id"]] = int(r["n_gold"])
def ci(x):
    x = np.array(x); rng = np.random.default_rng(0)
    m = x[rng.integers(0, len(x), (10000, len(x)))].mean(1)
    return f"{x.mean():+.3f} [{np.percentile(m, 2.5):+.3f}, {np.percentile(m, 97.5):+.3f}]"
for lo, hi in ((1, 1), (2, 3), (4, 999)):
    s = [i for i in ng if lo <= ng[i] <= hi]
    for m in ("Connected + tiers (EpT)", "Hybrid + tiers (no graph)", "Graph expansion (full only)"):
        print(f"gold {lo}-{hi if hi < 999 else '+'} (n={len(s)})  {m} - BM25: {ci([np.mean(d[m][i]) - np.mean(d['BM25 top-k'][i]) for i in s])}")
"""),
    md("## 7. Figure 1"),
    code("""
run("-m", "evtok.figures", "--csv", "results/tier_a/oof_per_task.csv", "--out", "results/tier_a")
from IPython.display import Image
Image("results/tier_a/fig_recall_vs_budget.png")
"""),
    md("""## 8. Robustness: learned ranker, budget-split rescue, Steiner-connected top-k (paper Section 6.3)

Same 5 folds. The ranker (logistic regression and gradient-boosted trees, scikit-learn) is trained per fold on training tasks only and scored on held-out tasks."""),
    code("""
from IPython.display import Markdown, display
run("-m", "evtok.extensions", "--data", DATA, "--out", "results/tier_a_ext", "--workers", str(WORKERS))
display(Markdown(open("results/tier_a_ext/summary.md", encoding="utf-8").read()))
"""),
    md("""## 9. RL with token penalties for context selection (paper Section 6.4)

REINFORCE policy that adds functions one at a time or stops. Reward = gold captured / |gold| − λ · tokens/1000, with λ ∈ {0.02, 0.05, 0.1}. Trained per fold on training tasks only. Each held-out task is compared with BM25 given the same number of tokens the policy chose to spend. Training is repeated with three seeds to show how stable the result is; the first table is seed 0. About 30 minutes on 4 CPU cores."""),
    code("""
from IPython.display import Markdown, display
run("-m", "evtok.rl_select", "--data", DATA, "--out", "results/rl", "--workers", str(WORKERS), "--epochs", "40", "--seeds", "3")
display(Markdown(open("results/rl/summary.md", encoding="utf-8").read()))
"""),
    md("## 10. Localization labels (evaluation set)\n\nRegenerates gold function labels from the attached dataset. The file is deleted afterwards because competition data may not be redistributed; run this cell yourself to get it."),
    code("""
run("-m", "evtok.export_labels", "--data", DATA, "--out", "/tmp/labels.jsonl")
import json
rows = [json.loads(l) for l in open("/tmp/labels.jsonl", encoding="utf-8")]
print(sum(not x["addition_only"] for x in rows), "of", len(rows), "tasks have function-level labels")
os.remove("/tmp/labels.jsonl")
"""),
    md("""
## Notes

- Token costs are `len(text) / 3.6`, an estimate shared by all methods. Absolute budgets are approximate.
- 9 of 129 tasks only add new code and have no gold function nodes. They are excluded from function-level metrics.
- The last cell deletes `grid.pkl`. It is large and contains gold node ids derived from the reference patches. The CSV and Markdown outputs are enough to check every number in the paper.
"""),
    code("""
os.remove("results/tier_a/grid.pkl")
print(sorted(os.listdir("results/tier_a")))
"""),
    md(read("docs/research_log.md")),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = os.path.join(ROOT, "notebook", "ept-tier-a-reproduction.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("wrote", out, len(cells), "cells")
