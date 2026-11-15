"""Regenerate function-level localization labels for the public tasks (a Gemma 4 localization eval set).

Usage: python -m evtok.export_labels --data <DATA_DIR> --out labels.jsonl

Each line: {"instance_id", "repo", "base_commit", "gold_nodes": [dotted ids], "gold_files": [paths],
"addition_only": bool, "n_graph_nodes": int}. Labels are derived from the reference patch by textual
matching to graph nodes (see evtok.gold). Competition rules forbid redistributing competition data, so
this script is the distributable artefact: anyone with the dataset can rebuild the labels exactly.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

from .data import graph_key, load_graph, load_tasks
from .gold import gold_nodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="labels.jsonl")
    a = ap.parse_args()
    groups = defaultdict(list)
    for t in load_tasks(a.data):
        groups[graph_key(t)].append(t)
    n = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for key in sorted(groups):
            g = load_graph(a.data, key)
            for t in groups[key]:
                gd = gold_nodes(g, t["patch"])
                f.write(json.dumps({
                    "instance_id": t["instance_id"], "repo": t["repo"], "base_commit": t["base_commit"],
                    "gold_nodes": sorted(g.ids[v] for v in gd["nodes"]), "gold_files": gd["files"],
                    "addition_only": gd["addition_only"], "n_graph_nodes": len(g.ids),
                }) + "\n")
                n += 1
    print(f"wrote {n} label rows to {a.out}")


if __name__ == "__main__":
    main()
