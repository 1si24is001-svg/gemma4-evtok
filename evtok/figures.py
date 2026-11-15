"""Paper figures from Tier A out-of-fold results (static PNG for the Kaggle Writeup).

Usage: python -m evtok.figures --csv results/tier_a/oof_per_task.csv --out results/tier_a
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Reference categorical palette (light), fixed order; controls in neutral gray.
SERIES = [
    ("BM25 top-k", "#2a78d6", "-", "o"),
    ("Dense top-k (anchor query)", "#eb6834", "-", "s"),
    ("Hybrid top-k", "#1baf7a", "-", "D"),
    ("Hybrid + tiers (no graph)", "#eda100", "-", "^"),
    ("Connected + tiers (EpT)", "#6250d6", "-", "v"),
    ("Random connected (control)", "#8a8984", ":", "x"),
    ("Oracle (gold first)", "#52514e", "--", "+"),
]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load(path):
    d = defaultdict(lambda: defaultdict(list))
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if int(r["n_gold"]) == 0:
                continue
            v = float(r["recall"])
            if not math.isnan(v):
                d[r["method"]][int(r["budget"])].append(v)
    return d


def boot(v, n=4000, seed=0):
    v = np.asarray(v)
    rng = np.random.default_rng(seed)
    m = v[rng.integers(0, v.size, (n, v.size))].mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/tier_a/oof_per_task.csv")
    ap.add_argument("--out", default="results/tier_a")
    a = ap.parse_args()
    d = load(a.csv)
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=200, facecolor=SURF)
    ax.set_facecolor(SURF)
    for name, col, ls, mk in SERIES:
        if name not in d:
            continue
        B = sorted(d[name])
        y = [np.mean(d[name][b]) for b in B]
        lo, hi = zip(*[boot(d[name][b]) for b in B])
        ax.fill_between(B, lo, hi, color=col, alpha=0.10, linewidth=0)
        ax.plot(B, y, ls, color=col, lw=2, marker=mk, ms=6, label=name, markeredgecolor=SURF, markeredgewidth=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks([2000, 4000, 8000, 16000], ["2k", "4k", "8k", "16k"])
    ax.minorticks_off()
    ax.set_xlabel("Context budget B (estimated tokens)")
    ax.set_ylabel("Function Recall@B (gold functions in full source)")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK)
    ax.set_title("Recall vs. context budget (out-of-fold, 120 tasks, 95% bootstrap bands)", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    p = os.path.join(a.out, "fig_recall_vs_budget.png")
    fig.savefig(p, facecolor=SURF)
    print("wrote", p)


if __name__ == "__main__":
    main()
