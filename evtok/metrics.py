"""Localization metrics for a packet against gold nodes (evaluation only)."""
from __future__ import annotations

import numpy as np

from .data import RepoGraph


def packet_metrics(g: RepoGraph, pk: dict[int, int], gold: dict) -> dict:
    G = gold["nodes"]
    full = {v for v, t in pk.items() if t == 2}
    anyt = set(pk)
    cost = sum(int(g.cost[v, t]) for v, t in pk.items())
    m = {
        "cost": cost,
        "n_nodes": len(pk),
        "n_full": len(full),
        "file_hit": float(bool(full & gold["file_nodes"])) if gold["file_nodes"] else float("nan"),
        "file_hit_any": float(bool(anyt & gold["file_nodes"])) if gold["file_nodes"] else float("nan"),
    }
    if G:
        hit_full = G & full
        m["recall"] = len(hit_full) / len(G)
        m["recall_any"] = len(G & anyt) / len(G)
        m["hit"] = float(bool(hit_full))
        m["hit_any"] = float(bool(G & anyt))
        m["density"] = (sum(int(g.cost[v, 2]) for v in hit_full) / cost) if cost else 0.0
    else:
        for k in ("recall", "recall_any", "hit", "hit_any", "density"):
            m[k] = float("nan")
    # connectivity diagnostics
    sel = set(pk)
    comps = 0
    seen: set[int] = set()
    for v in sel:
        if v in seen:
            continue
        comps += 1
        stack = [v]
        seen.add(v)
        while stack:
            x = stack.pop()
            for n in g.nbrs[x]:
                n = int(n)
                if n in sel and n not in seen:
                    seen.add(n)
                    stack.append(n)
    m["components"] = comps
    return m


def recall_auc(values_by_budget: list[float]) -> float:
    """Mean over log-spaced budgets (equal weight per doubling)."""
    v = [x for x in values_by_budget if not np.isnan(x)]
    return float(np.mean(v)) if v else float("nan")
