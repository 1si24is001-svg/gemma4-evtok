"""Loading tasks, code graphs (node-link JSON) and node embeddings (.npz keyed by node id)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

from .textproc import est_tokens, signature_text

TIERS = ("outline", "sig", "full")


def load_tasks(data_dir: str) -> list[dict]:
    with open(os.path.join(data_dir, "tasks.jsonl"), encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def graph_key(task: dict) -> str:
    return task["repo"].split("/")[1] + "_" + task["base_commit"]


def is_test_node(node_id: str) -> bool:
    parts = node_id.split(".")
    return any(p in ("tests", "test", "testing", "conftest") or p.startswith("test_") for p in parts)


@dataclass
class RepoGraph:
    key: str
    ids: list[str]
    text: list[str]
    idx: dict[str, int]
    nbrs: list[np.ndarray]  # undirected neighbours (dedup, no self loops)
    emb: np.ndarray  # (n, d), L2-normalized; zero rows for missing vectors
    has_emb: np.ndarray  # bool (n,)
    cost: np.ndarray  # (n, 3) token cost per tier: outline, sig, full
    pool: np.ndarray  # bool (n,): eligible for context (non-test)
    extra: dict = field(default_factory=dict)
    succ: list = field(default_factory=list)  # directed `calls` successors (used only for packet rendering)


def load_graph(data_dir: str, key: str) -> RepoGraph:
    with open(os.path.join(data_dir, "graphs", key + ".json"), encoding="utf-8") as f:
        g = json.load(f)
    ids = [n["id"] for n in g["nodes"]]
    text = [n.get("text") or "" for n in g["nodes"]]
    idx = {k: i for i, k in enumerate(ids)}
    adj: list[set[int]] = [set() for _ in ids]
    succ: list[set[int]] = [set() for _ in ids]
    for e in g["edges"]:
        s, t = idx.get(e["source"]), idx.get(e["target"])
        if s is None or t is None or s == t:
            continue
        adj[s].add(t)
        adj[t].add(s)
        succ[s].add(t)
    nbrs = [np.fromiter(sorted(a), dtype=np.int64, count=len(a)) for a in adj]

    emb_path = os.path.join(data_dir, "embeddings", key + ".npz")
    d = 256
    emb = np.zeros((len(ids), d), dtype=np.float32)
    has = np.zeros(len(ids), dtype=bool)
    if os.path.exists(emb_path):
        z = np.load(emb_path, allow_pickle=False)
        for k in z.files:
            i = idx.get(k)
            if i is not None:
                v = z[k].astype(np.float32).reshape(-1)
                nv = float(np.linalg.norm(v))
                if nv > 0 and v.shape[0] == d:
                    emb[i] = v / nv
                    has[i] = True

    cost = np.zeros((len(ids), 3), dtype=np.int64)
    for i, (nid, t) in enumerate(zip(ids, text)):
        head = est_tokens(nid) + 3  # "### <id>" header line
        cost[i, 0] = head
        cost[i, 1] = head + est_tokens(signature_text(t))
        cost[i, 2] = head + est_tokens(t)
    cost[:, 1] = np.maximum(cost[:, 1], cost[:, 0] + 1)
    cost[:, 2] = np.maximum(cost[:, 2], cost[:, 1])
    pool = np.array([not is_test_node(n) for n in ids], dtype=bool)
    return RepoGraph(key, ids, text, idx, nbrs, emb, has, cost, pool, succ=succ)
