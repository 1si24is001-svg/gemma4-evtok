"""Text utilities: identifier-aware tokenization, query cleaning, token-cost estimates, tier rendering."""
from __future__ import annotations

import re

# Token cost estimate. The Gemma tokenizer is not available offline on the dev machine;
# characters-per-token is calibrated later against the real tokenizer (see README).
CHARS_PER_TOKEN = 3.6

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")

STOP = set(
    """a an the and or of to in on for with by is are be was were this that it its as at from
    if else elif not no but we you they he she i do does did can could should would will may
    have has had into than then there here which who what when where how why all any some
    self cls none true false return def class import pass raise none ok use using used new
    pr pull request description discussion please link fix fixes add adds added https http www com github
    """.split()
)


def ident_tokens(s: str) -> list[str]:
    """Split text into lowercase tokens; identifiers are also split on snake_case and camelCase.

    The full identifier is kept as an extra token so exact-name matches score higher.
    """
    out: list[str] = []
    for w in _WORD.findall(s):
        lw = w.lower()
        parts = [p.lower() for part in w.split("_") if part for p in _CAMEL.findall(part)]
        if len(parts) > 1 and lw not in STOP:
            out.append(lw)
        out.extend(p for p in parts if p not in STOP and len(p) > 1)
        if len(parts) <= 1 and lw not in STOP and len(lw) > 1 and lw not in parts:
            out.append(lw)
    return out


_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
_URL = re.compile(r"https?://\S+")


def clean_problem(ps: str) -> str:
    """Remove PR-template comments and URLs from a problem statement."""
    ps = _HTML_COMMENT.sub(" ", ps)
    ps = _URL.sub(" ", ps)
    return ps


def est_tokens(s: str) -> int:
    return int(len(s) / CHARS_PER_TOKEN) + 1


def signature_text(text: str) -> str:
    """Return the definition header plus the first docstring paragraph (if any)."""
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    # header: decorators + def/class line(s) until a line ending with ':'
    seen_def = False
    while i < len(lines):
        out.append(lines[i])
        seen_def = seen_def or re.match(r"\s*(async\s+def|def|class)\b", lines[i]) is not None
        if seen_def and lines[i].rstrip().endswith(":"):
            i += 1
            break
        i += 1
        if i > 20:
            break
    # docstring
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i < len(lines) and lines[i].strip()[:3] in ('"""', "'''"):
        q = lines[i].strip()[:3]
        doc = [lines[i]]
        if not (lines[i].strip().count(q) >= 2 and len(lines[i].strip()) > 3):
            i += 1
            while i < len(lines) and len(doc) < 8:
                doc.append(lines[i])
                if q in lines[i] or not lines[i].strip():
                    break
                i += 1
        out.extend(doc)
    return "\n".join(out)


def norm_line(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()
