"""Leakage-free relevance signals from the problem statement only.

- BM25 over identifier-split node id + source text.
- Identifier anchoring: names mentioned in the PR text resolved to graph nodes
  (same resolution order as the harness: exact -> dotted suffix -> last component).
- Dense pseudo-query: mean embedding of anchor nodes (or of BM25 top-m when no anchor resolves),
  scored by cosine similarity. No text encoder exists for the 256-d node space.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from .data import RepoGraph
from .textproc import STOP, clean_problem, ident_tokens


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.dl = np.array([len(d) for d in docs], dtype=np.float32)
        self.avgdl = float(self.dl.mean()) if self.n else 1.0
        self.tf: list[Counter] = [Counter(d) for d in docs]
        df: Counter = Counter()
        for c in self.tf:
            df.update(c.keys())
        self.idf = {t: math.log(1 + (self.n - v + 0.5) / (v + 0.5)) for t, v in df.items()}
        self.post: dict[str, list[tuple[int, int]]] = {}
        for i, c in enumerate(self.tf):
            for t, v in c.items():
                self.post.setdefault(t, []).append((i, v))

    def score(self, q: list[str]) -> np.ndarray:
        s = np.zeros(self.n, dtype=np.float32)
        norm = self.k1 * (1 - self.b + self.b * self.dl / self.avgdl)
        for t, qtf in Counter(q).items():
            if t not in self.idf:
                continue
            idf = self.idf[t]
            for i, v in self.post[t]:
                s[i] += idf * v * (self.k1 + 1) / (v + norm[i])
        return s


def bm25_for_graph(g: RepoGraph) -> BM25:
    if "bm25" not in g.extra:
        docs = []
        for nid, t in zip(g.ids, g.text):
            idt = ident_tokens(nid.replace(".", " "))
            docs.append(idt + idt + ident_tokens(t[:20000]))
        g.extra["bm25"] = BM25(docs)
    return g.extra["bm25"]


_BT = re.compile(r"`([^`\n]{2,80})`")
_DOTTED = re.compile(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\b")
_CAMEL = re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+\b")
_SNAKE = re.compile(r"\b[a-z][a-z0-9]*_[a-z0-9_]+\b")
_CALL = re.compile(r"\b([A-Za-z_]\w{2,})\(")


def extract_identifiers(ps: str) -> list[str]:
    ps = clean_problem(ps)
    cands: list[str] = []
    for m in _BT.findall(ps):
        cands.extend(re.findall(r"[A-Za-z_][\w.]*", m))
    cands += _DOTTED.findall(ps) + _CAMEL.findall(ps) + _SNAKE.findall(ps) + _CALL.findall(ps)
    out, seen = [], set()
    for c in cands:
        c = c.strip(".")
        if len(c) < 3 or c.lower() in STOP or c in seen:
            continue
        if re.fullmatch(r"[a-z]+\.(py|md|txt|toml|cfg|yaml|yml|json|html)", c):
            continue
        seen.add(c)
        out.append(c)
    return out


def resolve_anchors(g: RepoGraph, names: list[str], max_matches: int = 8) -> dict[int, float]:
    if "by_last" not in g.extra:
        by_last: dict[str, list[int]] = {}
        for i, nid in enumerate(g.ids):
            by_last.setdefault(nid.rsplit(".", 1)[-1], []).append(i)
        g.extra["by_last"] = by_last
    by_last = g.extra["by_last"]
    w: dict[int, float] = {}
    for c in names:
        if c.endswith(".py"):
            c = c[:-3].replace("/", ".")
        m: list[int] = []
        if c in g.idx:
            m = [g.idx[c]]
        else:
            last = c.rsplit(".", 1)[-1]
            cand = by_last.get(last, [])
            if "." in c:
                m = [i for i in cand if g.ids[i].endswith("." + c)]
            if not m:
                m = list(cand)
            if not m and "." in c:
                # module path mention, e.g. fastapi.routing -> all nodes under it are too many; skip
                m = []
        m = [i for i in m if g.pool[i]]
        if 0 < len(m) <= max_matches:
            for i in m:
                w[i] = max(w.get(i, 0.0), 1.0 / len(m))
    return w


def zscore(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    v = x[mask]
    sd = float(v.std()) if v.size else 0.0
    z = (x - float(v.mean() if v.size else 0.0)) / (sd if sd > 1e-9 else 1.0)
    z[~mask] = -1e9
    return z


def task_signals(g: RepoGraph, problem: str, m_centroid: int = 5) -> dict:
    """All per-task relevance signals, computed from the problem statement only."""
    q = ident_tokens(clean_problem(problem))
    bm = bm25_for_graph(g).score(q)
    bm[~g.pool] = 0.0
    names = extract_identifiers(problem)
    anchors = resolve_anchors(g, names)
    anc = np.zeros(len(g.ids), dtype=np.float32)
    for i, w in anchors.items():
        anc[i] = w

    def dense_from(nodes: list[int]) -> np.ndarray:
        nodes = [i for i in nodes if g.has_emb[i]]
        if not nodes:
            return np.zeros(len(g.ids), dtype=np.float32)
        qv = g.emb[nodes].mean(axis=0)
        n = float(np.linalg.norm(qv))
        return (g.emb @ (qv / n)) if n > 0 else np.zeros(len(g.ids), dtype=np.float32)

    order = np.argsort(-bm, kind="stable")
    top_bm = [int(i) for i in order[:m_centroid] if bm[i] > 0]
    dense_bm = dense_from(top_bm)
    dense_anchor = dense_from(list(anchors.keys())) if anchors else dense_bm
    return {
        "bm25": bm,
        "anchor": anc,
        "dense_anchor": dense_anchor,
        "dense_bm": dense_bm,
        "n_anchor": len(anchors),
        "names": names,
    }


def hybrid_score(g: RepoGraph, sig: dict, w_dense: float, w_anchor: float, dense_src: str = "anchor") -> np.ndarray:
    mask = g.pool
    s = zscore(np.log1p(sig["bm25"]), mask)
    if w_dense:
        d = sig["dense_anchor"] if dense_src == "anchor" else sig["dense_bm"]
        s = s + w_dense * zscore(d, mask)
    if w_anchor:
        s = s + w_anchor * sig["anchor"] * 3.0
    s[~mask] = -1e9
    return s
