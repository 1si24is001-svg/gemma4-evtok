"""Map a reference (gold) patch onto graph nodes. Used ONLY for evaluation labels and the oracle.

Graph nodes carry no file paths or line numbers, so mapping is textual:
  1. removed ('-') lines of a hunk -> nodes whose source contains that line (most specific id);
  2. hunks with no removed lines -> context lines adjacent to the added block;
  3. fallback -> function named in the hunk header (git's nearest preceding def).
Candidates are restricted to nodes whose dotted id matches the patched file's module path when possible.
"""
from __future__ import annotations

import re
from collections import defaultdict

from .data import RepoGraph
from .textproc import norm_line

_HDR = re.compile(r"^@@[^@]*@@\s*(?:async\s+)?(?:def|class)\s+(\w+)")
TRIVIAL = {"return", "pass", "else:", "try:", "finally:", ")", "]", "}", "),", "return None", "continue", "break"}


def parse_patch(patch: str) -> list[dict]:
    """Return [{'path': str, 'new_file': bool, 'hunks': [{'header': str, 'lines': [(op, text)]}]}]."""
    files: list[dict] = []
    cur = None
    hunk = None
    for line in patch.split("\n"):
        if line.startswith("--- "):
            cur = {"path": None, "new_file": line.startswith("--- /dev/null"), "hunks": []}
            files.append(cur)
            hunk = None
        elif line.startswith("+++ ") and cur is not None:
            p = line[4:].strip()
            cur["path"] = p[2:] if p.startswith("b/") else p
        elif line.startswith("@@") and cur is not None:
            hunk = {"header": line, "lines": []}
            cur["hunks"].append(hunk)
            if line.startswith("@@ -0,0 "):
                cur["new_file"] = True
        elif hunk is not None and line[:1] in ("+", "-", " "):
            hunk["lines"].append((line[:1], line[1:]))
    return [f for f in files if f["path"]]


def module_prefixes(path: str) -> list[str]:
    if not path.endswith(".py"):
        return []
    dotted = path[:-3].replace("/", ".")
    out = [dotted]
    parts = dotted.split(".")
    if parts[0] in ("src", "docs_src", "lib"):
        out.append(".".join(parts[1:]))
    if dotted.endswith(".__init__"):
        out = [p[: -len(".__init__")] for p in out]
    return [p for p in out if p]


def nodes_in_file(g: RepoGraph, path: str) -> set[int]:
    prefs = module_prefixes(path)
    is_init = path.endswith("__init__.py")
    out = set()
    for i, nid in enumerate(g.ids):
        for p in prefs:
            if nid == p or nid.startswith(p + "."):
                if is_init:
                    rest = nid[len(p) + 1:]
                    # __init__ members are p.name or p.Class.method; exclude submodules (lowercase.x.y)
                    if rest.count(".") >= 2:
                        continue
                out.add(i)
                break
    return out


def _line_index(g: RepoGraph) -> dict[str, set[int]]:
    if "line_index" not in g.extra:
        idx: dict[str, set[int]] = defaultdict(set)
        for i, t in enumerate(g.text):
            for l in t.split("\n"):
                n = norm_line(l)
                if len(n) >= 8 and n not in TRIVIAL:
                    idx[n].add(i)
        g.extra["line_index"] = idx
    return g.extra["line_index"]


def _most_specific(g: RepoGraph, cands: set[int]) -> set[int]:
    if not cands:
        return set()
    # prefer the deepest ids (method over class over module-level)
    best = max(g.ids[c].count(".") for c in cands)
    deep = {c for c in cands if g.ids[c].count(".") == best}
    # if one candidate's id is a prefix of another, keep the longer
    return {c for c in deep if not any(o != c and g.ids[o].startswith(g.ids[c] + ".") for o in deep)}


def gold_nodes(g: RepoGraph, patch: str) -> dict:
    """Return {'nodes': set[int], 'files': list[str], 'file_nodes': set[int], 'addition_only': bool}."""
    li = _line_index(g)
    gold: set[int] = set()
    file_nodes: set[int] = set()
    files = parse_patch(patch)
    for f in files:
        in_file = nodes_in_file(g, f["path"])
        file_nodes |= in_file
        if f["new_file"]:
            continue
        for h in f["hunks"]:
            hits: set[int] = set()
            lines = h["lines"]
            removed = [t for op, t in lines if op == "-"]
            probe = list(removed)
            if not removed:
                for k, (op, t) in enumerate(lines):
                    if op == "+":
                        if k > 0 and lines[k - 1][0] == " ":
                            probe.append(lines[k - 1][1])
                        if k + 1 < len(lines) and lines[k + 1][0] == " ":
                            probe.append(lines[k + 1][1])
            for t in probe:
                n = norm_line(t)
                if len(n) < 8 or n in TRIVIAL:
                    continue
                c = li.get(n, set())
                if in_file:
                    c = c & in_file
                hits |= _most_specific(g, c)
            if not hits:
                m = _HDR.match(h["header"])
                if m:
                    name = m.group(1)
                    pool = in_file if in_file else range(len(g.ids))
                    c = {i for i in pool if g.ids[i].rsplit(".", 1)[-1] == name}
                    if len(c) <= 3:
                        hits |= c
            gold |= hits
    return {
        "nodes": gold,
        "files": [f["path"] for f in files],
        "file_nodes": file_nodes,
        "addition_only": not gold,
    }
