#!/usr/bin/env python3
"""Rank functions and classes in a Python repository against a change request (BM25, stdlib only).

Usage:
    python localize.py --query "text of the change request" [--root /workspace] [--k 15]
    python localize.py --query-file /tmp/request.txt

Prints one line per hit: score  path:line  qualified.name  (signature)
Test files and directories are skipped. Licensed under Apache-2.0.
"""
import argparse
import ast
import math
import os
import re
import sys
from collections import Counter

STOP = set(
    """a an the and or of to in on for with by is are be was were this that it its as at from
    if else elif not no but we you they he she i do does did can could should would will may
    have has had into than then there here which who what when where how why all any some
    self cls none true false return def class import pass raise ok use using used new
    pr pull request description discussion please link fix fixes add adds added https http www com github""".split()
)
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
SKIP_DIRS = {".git", "tests", "test", "testing", "node_modules", "build", "dist", "__pycache__", ".tox", ".venv", "venv"}


def tokens(s):
    out = []
    for w in _WORD.findall(s):
        lw = w.lower()
        parts = [p.lower() for part in w.split("_") if part for p in _CAMEL.findall(part)]
        if len(parts) > 1 and lw not in STOP:
            out.append(lw)
        out.extend(p for p in parts if p not in STOP and len(p) > 1)
        if len(parts) <= 1 and lw not in STOP and len(lw) > 1 and lw not in parts:
            out.append(lw)
    return out


def units(root):
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in SKIP_DIRS and not d.startswith(".")]
        for fn in fns:
            if not fn.endswith(".py") or fn.startswith("test_") or fn.endswith("_test.py") or fn == "conftest.py":
                continue
            path = os.path.join(dp, fn)
            rel = os.path.relpath(path, root).replace(os.sep, "/")
            try:
                src = open(path, encoding="utf-8", errors="replace").read()
                tree = ast.parse(src)
            except (SyntaxError, ValueError, OSError):
                continue
            lines = src.split("\n")
            mod = rel[:-3].replace("/", ".")

            def walk(node, prefix):
                for ch in ast.iter_child_nodes(node):
                    if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        name = f"{prefix}.{ch.name}"
                        end = getattr(ch, "end_lineno", ch.lineno)
                        body = "\n".join(lines[ch.lineno - 1:end])
                        sig = lines[ch.lineno - 1].strip()
                        if isinstance(ch, ast.ClassDef):
                            body = "\n".join(lines[ch.lineno - 1:min(end, ch.lineno + 15)])
                        yield rel, ch.lineno, name, sig, body
                        yield from walk(ch, name)

            yield from walk(tree, mod)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="")
    ap.add_argument("--query-file", default="")
    ap.add_argument("--root", default="/workspace")
    ap.add_argument("--k", type=int, default=15)
    a = ap.parse_args()
    q = a.query
    if a.query_file:
        q += "\n" + open(a.query_file, encoding="utf-8", errors="replace").read()
    q = re.sub(r"<!--.*?-->", " ", q, flags=re.S)
    q = re.sub(r"https?://\S+", " ", q)
    qt = Counter(tokens(q))
    if not qt:
        print("empty query", file=sys.stderr)
        return 2
    U = list(units(a.root))
    docs = []
    for rel, ln, name, sig, body in U:
        idt = tokens(name.replace(".", " "))
        docs.append(Counter(idt + idt + tokens(body[:20000])))
    n = len(docs)
    if n == 0:
        print("no python units found under", a.root)
        return 1
    dl = [sum(d.values()) for d in docs]
    avg = sum(dl) / n
    df = Counter()
    for d in docs:
        df.update(d.keys())
    k1, b = 1.2, 0.75
    scores = []
    for i, d in enumerate(docs):
        s = 0.0
        norm = k1 * (1 - b + b * dl[i] / avg)
        for t in qt:
            v = d.get(t)
            if v:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * v * (k1 + 1) / (v + norm)
        scores.append(s)
    order = sorted(range(n), key=lambda i: -scores[i])[: a.k]
    for i in order:
        rel, ln, name, sig, _ = U[i]
        print(f"{scores[i]:7.2f}  {rel}:{ln}  {name}  ({sig[:100]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
