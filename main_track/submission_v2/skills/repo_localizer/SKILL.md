---
name: repo_localizer
description: Ranks the functions and classes in /workspace that best match a change request (BM25 over identifiers and source, tests excluded). Use it once at the start to find where to edit.
---

# repo_localizer

Finds the code a change request is most likely about. It ranks every function and class in the repository by text match against the request and prints `score  path:line  qualified.name  (signature)` for the top hits.

## How to use

Run the script `scripts/localize.py` with the key words of the request: identifiers, error messages, option names and distinctive nouns.

Arguments:
- `--query "<keywords>"` (required): for example `--query "netrc auth get_netrc_auth raise_errors"`
- `--k 15`: number of results (optional)
- `--root /workspace`: repository root (optional, default `/workspace`)

Then open the top 2 or 3 files at the listed line numbers with `read_file`.

## If the script cannot run

Do not retry more than once. Use text search instead:
`cd /workspace && git grep -n -E "word1|word2|word3" -- '*.py' ':!tests' ':!test' | head -60`
