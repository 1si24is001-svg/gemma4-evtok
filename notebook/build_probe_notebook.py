"""Build the Kaggle GPU notebook for the Tier A.5 Gemma 4 localization probe.

Usage: python notebook/build_probe_notebook.py -> notebook/ept-gemma4-localization-probe.ipynb
Server start-up follows the official competition starter notebook (wheelhouse + adk_submission.VllmServer).
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULES = ["__init__", "textproc", "data", "gold", "retrieval", "selectors", "metrics", "run_tier_a", "probe"]


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(keepends=True)}


def read(p):
    with open(os.path.join(ROOT, p), encoding="utf-8") as f:
        return f.read()


cells = [
    md("""
# Evidence-per-Token, Tier A.5: does Gemma 4 use the evidence?

Companion to the CPU reproduction notebook. For each of the 120 labelled tasks, we build a context packet with four selectors at an 8k-token budget: BM25 top-k, multi-level packing without a graph, connected + multi-level (EpT), and an oracle with the gold functions first. We give Gemma 4 the change request and the packet and ask it to name up to 5 functions to edit, as JSON. We then score the answer against the functions the reference fix edits. No patch is generated and no tests run.

**Inputs to attach:**
1. The competition dataset (*Google - The Gemma 4 Developer Agent Competition*).
2. The wheelhouse dataset `metric/gemma-4-developer-agent-wheelhouse`, the same one the official starter uses.
3. The model *google/gemma-4*, variant `gemma-4-e4b-it-qat-w4a16-ct` (and optionally `gemma-4-12b-it-qat-w4a16-ct`).

**Environment.** The wheelhouse is built for Python 3.12. Run this inside a copy of the official starter notebook (*Getting Started - Gemma 4 Developer Agent* → Copy & Edit → File → Import Notebook), which keeps the matching environment and inputs. The first cell stops with instructions if the environment is wrong.

**Settings:** Accelerator GPU L4 x4 or GPU T4 x2 (the notebook adapts), Internet off.

**Leakage boundary.** Prompts contain only the problem statement and graph-derived packet text. Gold labels are used only for scoring.
"""),
    code("""
import glob, importlib, json, logging, os, re, shutil, subprocess, sys, time, urllib.request
from pathlib import Path

os.environ['VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS'] = '1'
os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['LITELLM_LOCAL_MODEL_COST_MAP'] = 'True'
os.environ['LITELLM_TELEMETRY'] = 'False'
os.environ['OPENAI_API_KEY'] = os.environ.get('OPENAI_API_KEY', 'EMPTY')
os.environ['LOCAL_API_KEY'] = os.environ.get('LOCAL_API_KEY', 'EMPTY')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['OTEL_SDK_DISABLED'] = 'true'

# --- environment checks: stop early with instructions instead of failing deep inside vLLM ---
cands = glob.glob('/kaggle/input/**/gemma-4-developer-agent-wheelhouse', recursive=True)
WHEELHOUSE_DIR = Path(cands[0]) if cands else Path('/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse')
all_wheels = [w.name for w in WHEELHOUSE_DIR.glob('*.whl')]
assert all_wheels, ('Wheelhouse not attached. Add Input -> Datasets -> "Gemma 4 Developer Agent Wheelhouse" '
                    '(metric/gemma-4-developer-agent-wheelhouse), then run again.')
py_tag = f'cp{sys.version_info.major}{sys.version_info.minor}'
wheel_tags = sorted(set(re.findall(r'-(cp3\\d+)-', ' '.join(all_wheels))))
print('Python', sys.version.split()[0], '| wheelhouse built for:', wheel_tags or 'any')
if wheel_tags and py_tag not in wheel_tags:
    raise SystemExit(
        f'WRONG ENVIRONMENT: this session runs Python {sys.version_info.major}.{sys.version_info.minor} ({py_tag}) '
        f'but the wheelhouse is built for {wheel_tags}. vLLM cannot load here.\\n'
        'Fix: open the official starter notebook "Getting Started - Gemma 4 Developer Agent", click Copy & Edit, '
        'then in that copy use File -> Import Notebook to load this file. That copy keeps the right environment.')
r = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], capture_output=True, text=True)
print('GPUs:', r.stdout.strip().replace('\\n', ' | ') or 'NONE')
assert r.stdout.strip(), 'No GPU. Settings -> Accelerator -> GPU L4 x4 (or GPU T4 x2), then run again.'

# --- wheelhouse install (from the official starter notebook) ---
shutil.rmtree('/tmp/wheelhouse', ignore_errors=True)
for pth_pattern in ('/usr/local/lib/python*/dist-packages/*cutlass*.pth', '/usr/local/lib/python*/site-packages/*cutlass*.pth'):
    for pth in glob.glob(pth_pattern):
        try:
            os.unlink(pth)
        except OSError:
            pass
tmp_whl = Path('/tmp/wheelhouse'); tmp_whl.mkdir(parents=True, exist_ok=True)
for w in WHEELHOUSE_DIR.glob('*.whl'):
    if 'cutlass' in w.name.lower():
        continue
    target = tmp_whl / (w.name.replace('cu128', '+cu128') if ('cu128' in w.name and '+' not in w.name) else w.name)
    if not target.exists():
        try:
            os.symlink(w, target)
        except OSError:
            shutil.copyfile(w, target)
wheels = sorted(str(w) for w in tmp_whl.glob('*.whl'))
print(f'Installing {len(wheels)} wheels from {WHEELHOUSE_DIR}...')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '--force-reinstall', *wheels], check=True)
importlib.invalidate_caches()
print('Wheelhouse installation complete.')
"""),
    code("""
# --- data and model ---
cands = glob.glob('/kaggle/input/**/tasks.jsonl', recursive=True)
DATA = os.path.dirname(cands[0])
PREFERRED = ['gemma-4-e4b-it-qat-w4a16-ct', 'gemma-4-12b-it-qat-w4a16-ct', 'gemma-4-31b-it-qat-w4a16-ct', 'gemma-4-e2b-it-qat-w4a16-ct']  # first one found is used
configs = [Path(c) for c in glob.glob('/kaggle/input/**/config.json', recursive=True) if 'gemma' in c.lower()]
print('Gemma model folders found:', sorted({str(c.parent) for c in configs}) or 'NONE')
MODEL_NAME = MODEL_PATH = None
for name in PREFERRED:
    hits = sorted(c.parent for c in configs if name in str(c))
    if hits:
        MODEL_NAME, MODEL_PATH = name, hits[-1]
        break
assert MODEL_PATH is not None, 'No Gemma 4 model attached. Add Input -> Models -> google/gemma-4, variant gemma-4-e4b-it-qat-w4a16-ct'
print('data:', DATA, '| model:', MODEL_NAME, '|', MODEL_PATH)
"""),
    code("""
# --- start vLLM (configuration from the official starter notebook) ---
import torch
from adk_submission import VllmConfig, VllmServer, discover_adapters
from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS

EXTRA = {'gemma-4-12b-it-qat-w4a16-ct': ('--hf-overrides', '{"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"}')}
empty = Path('/tmp/no_adapters'); empty.mkdir(exist_ok=True)
adapters = discover_adapters(str(empty), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
tp_size = 4 if gpu_count >= 4 else (2 if gpu_count >= 2 else 1)
if 'server_instance' in globals() and server_instance is not None:
    server_instance.stop()
vllm_cfg = VllmConfig(
    model=str(MODEL_PATH), port=8000, host='127.0.0.1',
    tool_call_parser='gemma4', reasoning_parser='gemma4', max_model_len=32768,
    dtype='bfloat16' if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else 'auto',
    gpu_memory_utilization=0.90, enable_auto_tool_choice=True, enable_lora=False, max_loras=1, max_lora_rank=128,
    scheduler_reserve_full_isl=False, tensor_parallel_size=tp_size, startup_timeout=60 * 20,
    extra_args=EXTRA.get(MODEL_NAME, ()),
)
server_instance = VllmServer(vllm_cfg, adapter_manifest=adapters)
server_instance.start()
API = 'http://127.0.0.1:8000/v1'
with urllib.request.urlopen(urllib.request.Request(API + '/models', headers={'Authorization': 'Bearer EMPTY'})) as r:
    SERVED = json.loads(r.read())['data'][0]['id']
PARALLEL = str(4 * tp_size)   # 16 concurrent requests on L4 x4, 8 on T4 x2, 4 on one GPU
print('vLLM up, tp =', tp_size, '| served model id:', SERVED, '| parallel requests:', PARALLEL)
"""),
    md("## Source code (same files as the CPU notebook, plus `probe`)"),
]
cells.append(code("os.makedirs('evtok', exist_ok=True)"))
for m in MODULES:
    cells.append(code(f"%%writefile evtok/{m}.py\n" + read(f"evtok/{m}.py")))
cells += [
    md("## Smoke test (6 tasks, one method)\n\nStops the notebook with a readable error if the model does not answer, so the long run below never starts on a broken setup."),
    code("""
def run(*args):
    # Run an evtok module and stream its output live, so long steps show progress.
    t0 = time.time()
    print('running:', ' '.join(args), flush=True)
    p = subprocess.Popen([sys.executable, '-u', *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end='', flush=True)
    rc = p.wait()
    print(f'[exit code {rc}, {time.time() - t0:.0f}s]', flush=True)
    if rc != 0:
        raise RuntimeError(f'step failed with exit code {rc}; see the output above')

run('-m', 'evtok.probe', '--data', DATA, '--out', 'results/probe_smoke', '--api', API, '--model', SERVED,
    '--limit', '6', '--methods', 'BM25 top-k', '--parallel', '4')
import csv
smoke = list(csv.DictReader(open('results/probe_smoke/probe_per_task.csv', encoding='utf-8')))
for row in smoke:
    print(row['instance_id'], 'acc5', row['acc5'], 'tokens', row['prompt_tokens'], row['error'][:200], '|', row['answer'][:200])
assert not any(row['error'] for row in smoke), 'the model returned errors; fix this before the full run'
assert any(row['parse_ok'] == '1' for row in smoke), 'no answer could be parsed as function ids; check the answers printed above'
"""),
    md("""## Main probe: 120 tasks × 4 packets at 8k

BM25 top-k, tiers without graph, EpT, and the oracle. On T4 x2 this is the slow step (it can take over an hour); progress prints every 50 responses."""),
    code("""
OUT = f'results/probe_{MODEL_NAME}'
run('-m', 'evtok.probe', '--data', DATA, '--out', OUT, '--api', API, '--model', SERVED, '--parallel', PARALLEL)
"""),
    md("""## Graph-serialization ablation: 120 tasks × 3 renderings

One EpT packet built at 0.9 × budget and rendered with no graph text, with a call-edge list, or with per-entry `calls / called by` lines. A failure here does not discard the main probe above."""),
    code("""
OUT_ABL = f'results/probe_serialization_{MODEL_NAME}'
ABLATION_OK = True
try:
    run('-m', 'evtok.probe', '--data', DATA, '--out', OUT_ABL, '--api', API, '--model', SERVED, '--parallel', PARALLEL,
        '--methods', 'ablation')
except Exception as e:
    ABLATION_OK = False
    print('serialization ablation did not finish:', e)
"""),
    md("## No-LLM baseline: answer with the top-ranked full-source function"),
    code("""
run('-m', 'evtok.probe', '--data', DATA, '--out', 'results/probe_rank1', '--mock')
"""),
    md("## Paired comparisons (bootstrap 95% CI, exact McNemar on Acc@5)"),
    code("""
import csv, math
import numpy as np
from collections import defaultdict
rows = list(csv.DictReader(open(f'{OUT}/probe_per_task.csv', encoding='utf-8')))
if ABLATION_OK:
    rows += list(csv.DictReader(open(f'{OUT_ABL}/probe_per_task.csv', encoding='utf-8')))
acc = defaultdict(dict)
for r_ in rows:
    acc[r_['method']][r_['instance_id']] = int(r_['acc5'])
ids = sorted(acc['BM25 top-k'])
def mcn(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n)
rng = np.random.default_rng(0)
def compare(m, base):
    d = np.array([acc[m][i] - acc[base][i] for i in ids], dtype=float)
    bs = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
    b = int(sum(acc[m][i] == 1 and acc[base][i] == 0 for i in ids)); c = int(sum(acc[m][i] == 0 and acc[base][i] == 1 for i in ids))
    print(f'{m} - {base}: dAcc@5 {d.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]  wins/losses {b}/{c}  McNemar p={mcn(b, c):.3f}')
for m in ['Hybrid + tiers (no graph)', 'Connected + tiers (EpT)', 'Oracle (gold first)']:
    compare(m, 'BM25 top-k')
for m in ['EpT 0.9B + edge list', 'EpT 0.9B + per-entry adjacency']:
    if ABLATION_OK:
        compare(m, 'EpT 0.9B, no graph text')
"""),
    md("## Results to copy into the paper\n\nEverything between the two marker lines."),
    code("""
try:
    server_instance.stop()
except Exception as e:
    print('server stop:', e)
print('=' * 20, 'COPY FROM HERE', '=' * 20)
print('model:', MODEL_NAME, '| GPUs:', gpu_count, '| tensor parallel:', tp_size)
print(open(f'{OUT}/probe_summary.md', encoding='utf-8').read())
print('--- graph-serialization ablation ---')
print(open(f'{OUT_ABL}/probe_summary.md', encoding='utf-8').read() if ABLATION_OK else 'not completed')
print('--- no-LLM rank-1 baseline ---')
print(open('results/probe_rank1/probe_summary.md', encoding='utf-8').read())
print('=' * 20, 'COPY TO HERE', '=' * 20)
"""),
    md("""
## Notes

- Decoding: temperature 0, seed 0, thinking disabled, max 512 output tokens. The same prompt template is used for every method, and only the packet differs.
- `packet_tokens` in `probe_per_task.csv` is the real Gemma tokenizer count of the packet alone (prompt tokens minus the same prompt with an empty packet). The summary reports its ratio to the `len/3.6` estimate used for budgets.
- Fixed selector settings are the ones Tier A cross-validation chose most often. They are listed in `evtok/probe.py`.
"""),
]

nb = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = os.path.join(ROOT, "notebook", "ept-gemma4-localization-probe.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("wrote", out, len(cells), "cells")
