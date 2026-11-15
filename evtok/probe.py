"""Tier A.5: does Gemma 4 use the evidence in a packet? A localization probe.

For each task and packet method, render the packet as text, ask the model to name the functions
that must be edited (JSON), and score the answer against gold nodes.

Usage (on a machine with an OpenAI-compatible server, e.g. vLLM on 127.0.0.1:8000):
    python -m evtok.probe --data <DATA_DIR> --out results/probe --api http://127.0.0.1:8000/v1 --model <served name>
    python -m evtok.probe ... --mock        # offline smoke test, no model

Leakage boundary: prompts contain only the problem statement and packet text built from graphs and
embeddings. Gold nodes are used only to score answers.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import threading
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import selectors as S
from .data import graph_key, load_graph, load_tasks
from .gold import gold_nodes
from .retrieval import hybrid_score, task_signals
from .run_tier_a import TAU, TEMP, TIERW
from .textproc import clean_problem, signature_text

# Fixed configurations: the settings chosen most often by Tier A cross-validation.
METHODS = {
    "BM25 top-k": dict(kind="flat", h=(0.0, 0.0, "anchor")),
    "Hybrid + tiers (no graph)": dict(kind="flatT", h=(0.5, 1.0, "bm"), rho=0.5, tw="a"),
    "Connected + tiers (EpT)": dict(kind="connT", h=(0.0, 0.0, "anchor"), beta=1.0, hh=2, k=10, rho=0.5, tw="a"),
    "Oracle (gold first)": dict(kind="oracle", h=(0.0, 0.0, "anchor")),
    # Graph-serialization ablation: one EpT packet built at 0.9B, rendered three ways.
    "EpT 0.9B, no graph text": dict(kind="connT", h=(0.0, 0.0, "anchor"), beta=1.0, hh=2, k=10, rho=0.5, tw="a", scale=0.9, enc="none"),
    "EpT 0.9B + edge list": dict(kind="connT", h=(0.0, 0.0, "anchor"), beta=1.0, hh=2, k=10, rho=0.5, tw="a", scale=0.9, enc="edges"),
    "EpT 0.9B + per-entry adjacency": dict(kind="connT", h=(0.0, 0.0, "anchor"), beta=1.0, hh=2, k=10, rho=0.5, tw="a", scale=0.9, enc="adj"),
}
DEFAULT_METHODS = ["BM25 top-k", "Hybrid + tiers (no graph)", "Connected + tiers (EpT)", "Oracle (gold first)"]

SYSTEM = (
    "You localize code changes. You are given a change request for a Python repository and a context "
    "packet: functions and classes from the repository, each under a header line '### <dotted.id>'. "
    "Some entries show full source, some only a signature, some only the id. "
    "Name the functions or classes that must be edited to implement the request, most likely first, "
    "at most 5, using ids exactly as written in the packet. "
    'Reply with JSON only: {"functions": ["dotted.id", ...]}'
)


def build_packet(g, sig, gold, method: str, B: int) -> dict[int, int]:
    cfg = METHODS[method]
    B = int(B * cfg.get("scale", 1.0))
    s = hybrid_score(g, sig, *cfg["h"])
    order = np.argsort(-s, kind="stable")
    if cfg["kind"] == "flat":
        return S.flat_full(g, order, B)
    if cfg["kind"] == "oracle":
        return S.oracle(g, gold["nodes"], S.flat_full(g, order, B), B)
    top = [int(v) for v in order[:400] if s[v] > -1e8]
    if cfg["kind"] == "flatT":
        return S.greedy_tiers(g, S.utility(s, None, 0.0, TAU, TEMP), B, top, rho=cfg["rho"], expand=False, wt=TIERW[cfg["tw"]])
    seeds = top[: cfg["k"]]
    dist = S.bfs_dist(g, seeds, cfg["hh"])
    u = S.utility(s, dist, cfg["beta"], TAU, TEMP)
    return S.greedy_tiers(g, u, B, seeds, rho=cfg["rho"], expand=True, region=dist, wt=TIERW[cfg["tw"]])


def render(g, pk: dict[int, int], s_order: dict[int, int], enc: str = "none") -> str:
    """Full-source entries first, then signatures, then ids; each group in relevance order.

    enc: "none" (no graph text), "edges" (a call-edge list among packet entries at the end),
    "adj" (a 'calls: ... | called by: ...' line under each entry header).
    """
    sel = set(pk)
    calls = {v: [w for w in sorted(g.succ[v]) if w in sel] for v in sel} if g.succ else {v: [] for v in sel}
    called_by: dict[int, list[int]] = {v: [] for v in sel}
    for v, ws in calls.items():
        for w in ws:
            called_by[w].append(v)
    parts = []
    for tier in (2, 1, 0):
        nodes = sorted((v for v, t in pk.items() if t == tier), key=lambda v: s_order.get(v, 10**9))
        for v in nodes:
            head = f"### {g.ids[v]}" + {2: "", 1: " (signature only)", 0: " (id only)"}[tier]
            if enc == "adj" and (calls[v] or called_by[v]):
                head += "\ncalls: " + (", ".join(g.ids[w] for w in calls[v]) or "-") + " | called by: " + (", ".join(g.ids[w] for w in called_by[v]) or "-")
            if tier == 2:
                parts.append(f"{head}\n```python\n{g.text[v]}\n```")
            elif tier == 1:
                parts.append(f"{head}\n```python\n{signature_text(g.text[v])}\n```")
            else:
                parts.append(head)
    if enc == "edges":
        edges = [f"- {g.ids[v]} -> {g.ids[w]}" for v in sorted(sel, key=lambda x: s_order.get(x, 10**9)) for w in calls[v]]
        parts.append("## Call edges among packet entries (caller -> callee)\n" + ("\n".join(edges) if edges else "(none)"))
    return "\n\n".join(parts)


def make_prompt(task: dict, packet_text: str) -> list[dict]:
    user = (
        f"Repository: {task['repo']}\n\n## Change request\n{clean_problem(task['problem_statement']).strip()[:6000]}\n\n"
        f"## Context packet\n{packet_text}\n\nReply with JSON only."
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def parse_answer(text: str) -> list[str]:
    text = re.sub(r"<\|channel>.*?<channel\|>", "", text or "", flags=re.S)
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            v = json.loads(m.group(0)).get("functions", [])
            return [str(x).strip().strip("`") for x in v if isinstance(x, (str, int))][:5]
        except (json.JSONDecodeError, AttributeError):
            pass
    return re.findall(r"[A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)+", text)[:5]


def resolve(g, names: list[str]) -> list[int]:
    out = []
    for n in names:
        n = n.split("(")[0].strip()
        if n in g.idx:
            out.append(g.idx[n])
            continue
        c = [i for i, nid in enumerate(g.ids) if nid.endswith("." + n)]
        if len(c) == 1:
            out.append(c[0])
    return out


def call_openai(api: str, model: str, messages: list[dict], max_tokens: int, timeout: int = 600, no_think_kwarg: bool = True) -> dict:
    body = {"model": model, "messages": messages, "temperature": 0.0, "max_tokens": max_tokens, "seed": 0}
    if no_think_kwarg:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(
        api.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + os.environ.get("LOCAL_API_KEY", "EMPTY")},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.loads(r.read())
    except urllib.error.HTTPError as e:  # surface the server's message, not just the status code
        detail = e.read().decode("utf-8", "replace")[:500]
        if e.code == 400 and no_think_kwarg and "chat_template_kwargs" in detail:
            return call_openai(api, model, messages, max_tokens, timeout, no_think_kwarg=False)
        raise RuntimeError(f"HTTP {e.code}: {detail}") from None
    msg = resp["choices"][0]["message"]
    return {
        # some servers put the answer in reasoning_content when a reasoning parser is active
        "text": msg.get("content") or msg.get("reasoning_content") or msg.get("reasoning") or "",
        "prompt_tokens": resp.get("usage", {}).get("prompt_tokens"),
        "completion_tokens": resp.get("usage", {}).get("completion_tokens"),
        "latency": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/probe")
    ap.add_argument("--api", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="")
    ap.add_argument("--budget", type=int, default=8000)
    ap.add_argument("--methods", default=",".join(DEFAULT_METHODS), help="comma-separated; 'all' runs every method incl. the serialization ablation")
    ap.add_argument("--parallel", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--mock", action="store_true", help="no model: answer with the first full-source id")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.methods == "all":
        methods = list(METHODS)
    elif a.methods == "ablation":  # the three graph-serialization renderings (their names contain commas)
        methods = [m for m in METHODS if m not in DEFAULT_METHODS]
    else:
        methods = [m for m in a.methods.split(",") if m]
    tasks = load_tasks(a.data)
    if a.limit:
        tasks = tasks[: a.limit]
    groups = defaultdict(list)
    for t in tasks:
        groups[graph_key(t)].append(t)

    jobs = []  # (task, method, messages, meta)
    graphs = {}
    for key in sorted(groups):
        g = load_graph(a.data, key)
        for t in groups[key]:
            gold = gold_nodes(g, t["patch"])
            if not gold["nodes"]:
                continue
            sig = task_signals(g, t["problem_statement"])
            order = np.argsort(-hybrid_score(g, sig, 0.0, 0.0, "anchor"), kind="stable")
            rank = {int(v): r for r, v in enumerate(order[:5000])}
            for m in methods:
                pk = build_packet(g, sig, gold, m, a.budget)
                text = render(g, pk, rank, METHODS[m].get("enc", "none"))
                full = {v for v, tt in pk.items() if tt == 2}
                jobs.append((t, m, make_prompt(t, text), {
                    "graph": key, "gold": sorted(gold["nodes"]), "in_full": bool(full & gold["nodes"]),
                    "in_any": bool(set(pk) & gold["nodes"]), "est_cost": S.packet_cost(g, pk), "chars": len(text),
                    "first_full": g.ids[min(full, key=lambda v: rank.get(v, 10**9))] if full else "",
                }))
        g.extra.clear()  # drop BM25 index and line index; keep ids/text for scoring
        graphs[key] = g
    print(f"{len(jobs)} prompts built", flush=True)

    if not a.mock:  # fail fast with a readable message if the server or model name is wrong
        try:
            pre = call_openai(a.api, a.model, jobs[0][2], a.max_tokens, timeout=900)
        except Exception as e:  # noqa: BLE001
            raise SystemExit(f"PREFLIGHT FAILED, nothing was run: {e!r}")
        print(f"preflight ok: {pre['prompt_tokens']} prompt tokens, {pre['latency']:.1f}s, answer: {pre['text'][:200]!r}", flush=True)

    done, lock = [0], threading.Lock()

    def run(job):
        t, m, msgs, meta = job
        if a.mock:
            return {"text": json.dumps({"functions": [meta["first_full"]]}), "prompt_tokens": None, "completion_tokens": None, "latency": 0.0}
        err, out = "", None
        for attempt in range(3):
            try:
                out = call_openai(a.api, a.model, msgs, a.max_tokens)
                break
            except Exception as e:  # noqa: BLE001
                err = repr(e)
                time.sleep(5 * (attempt + 1))
        with lock:
            done[0] += 1
            if done[0] % 50 == 0:
                print(f"  {done[0]}/{len(jobs)} responses, {time.time() - t0:.0f}s", flush=True)
        return out or {"text": "", "error": err, "prompt_tokens": None, "completion_tokens": None, "latency": None}

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=1 if a.mock else a.parallel) as ex:
        outs = list(ex.map(run, jobs))
    print(f"generation done in {time.time() - t0:.0f}s", flush=True)

    # Real tokenizer cost of the packet alone: prompt tokens minus the same prompt with an empty packet.
    base_tok = {}
    if not a.mock:
        seen = {}
        for t, m, msgs, meta in jobs:
            seen.setdefault(t["instance_id"], t)

        def base(t):
            try:
                return t["instance_id"], call_openai(a.api, a.model, make_prompt(t, ""), 1)["prompt_tokens"]
            except Exception:  # noqa: BLE001
                return t["instance_id"], None

        with ThreadPoolExecutor(max_workers=a.parallel) as ex:
            base_tok = dict(ex.map(base, seen.values()))
        print(f"baseline prompts done in {time.time() - t0:.0f}s", flush=True)

    rows = []
    for (t, m, msgs, meta), o in zip(jobs, outs):
        g = graphs[meta["graph"]]
        pred = resolve(g, parse_answer(o["text"]))
        gold = set(meta["gold"])
        rows.append({
            "instance_id": t["instance_id"], "repo": t["repo"], "method": m, "n_gold": len(gold),
            "gold_in_full": int(meta["in_full"]), "gold_in_any": int(meta["in_any"]),
            "acc1": int(bool(pred) and pred[0] in gold), "acc5": int(bool(set(pred[:5]) & gold)),
            "n_pred": len(pred), "parse_ok": int(bool(pred)),
            "est_cost": meta["est_cost"], "chars": meta["chars"],
            "prompt_tokens": o.get("prompt_tokens"), "completion_tokens": o.get("completion_tokens"),
            "packet_tokens": (o["prompt_tokens"] - base_tok[t["instance_id"]]) if (o.get("prompt_tokens") and base_tok.get(t["instance_id"])) else None,
            "latency": o.get("latency"), "error": o.get("error", ""),
            "answer": (o["text"] or "")[:400].replace("\n", " "),
        })
    with open(os.path.join(a.out, "probe_per_task.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # summary + waterfall
    lines = [f"# Tier A.5 localization probe (budget {a.budget}, model {a.model or 'MOCK'})\n",
             "| Method | n | Gold in packet (full) | Gold in packet (any tier) | Acc@1 | Acc@5 | Acc@5 given gold in full | Parse OK |",
             "|---|---|---|---|---|---|---|---|"]
    for m in methods:
        r = [x for x in rows if x["method"] == m]
        gf = [x for x in r if x["gold_in_full"]]
        lines.append(f"| {m} | {len(r)} | {np.mean([x['gold_in_full'] for x in r]):.3f} | {np.mean([x['gold_in_any'] for x in r]):.3f} | "
                     f"{np.mean([x['acc1'] for x in r]):.3f} | {np.mean([x['acc5'] for x in r]):.3f} | "
                     f"{(np.mean([x['acc5'] for x in gf]) if gf else float('nan')):.3f} | {np.mean([x['parse_ok'] for x in r]):.3f} |")
    pt = [(x["est_cost"], x["packet_tokens"]) for x in rows if x["packet_tokens"] and x["est_cost"]]
    if pt:
        est, real = np.array(pt, dtype=float).T
        ratio = real / est
        lines.append(f"\nToken calibration (Gemma tokenizer, packet only; n={len(pt)}): real / estimated tokens: median {np.median(ratio):.3f}, "
                     f"10th-90th percentile {np.percentile(ratio, 10):.3f}-{np.percentile(ratio, 90):.3f}. "
                     f"Median real packet size {np.median(real):.0f} tokens for a budget of {a.budget} estimated tokens; "
                     f"{np.mean(real > a.budget):.1%} of packets exceed the budget in real tokens. "
                     "Rendering adds code fences and tier labels that the estimate does not count.")
    lat = [x["latency"] for x in rows if x["latency"]]
    if lat:
        ct = [x["completion_tokens"] for x in rows if x["completion_tokens"]]
        lines.append(f"\nLatency per request (parallel {a.parallel}): median {np.median(lat):.1f}s, 90th percentile {np.percentile(lat, 90):.1f}s; "
                     f"median completion {np.median(ct):.0f} tokens. Requests with errors: {sum(bool(x['error']) for x in rows)} of {len(rows)}.")
    lines.append("\n## Waterfall (share of tasks)\n")
    lines.append("| Method | missed by retrieval | in packet, model misses | model hits (Acc@5) |")
    lines.append("|---|---|---|---|")
    for m in methods:
        r = [x for x in rows if x["method"] == m]
        miss = np.mean([1 - x["gold_in_any"] for x in r])
        used = np.mean([x["gold_in_any"] * (1 - x["acc5"]) for x in r])
        hit = np.mean([x["gold_in_any"] * x["acc5"] for x in r])
        lines.append(f"| {m} | {miss:.3f} | {used:.3f} | {hit:.3f} |")
    open(os.path.join(a.out, "probe_summary.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
