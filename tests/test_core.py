"""Minimal unit tests: budget enforcement, connectivity, patch parsing, tokenization."""
import numpy as np

from evtok import selectors as S
from evtok.data import RepoGraph
from evtok.gold import parse_patch, module_prefixes
from evtok.textproc import ident_tokens, signature_text


def toy_graph():
    ids = ["pkg.a", "pkg.b", "pkg.c", "pkg.d", "tests.test_x"]
    text = ["def a():\n    return b()", "def b():\n    return c()", "def c():\n    pass", "def d():\n    pass", "def test_x():\n    a()"]
    nbrs = [np.array([1, 4]), np.array([0, 2]), np.array([1]), np.array([], dtype=np.int64), np.array([0])]
    cost = np.array([[3, 6, 20], [3, 6, 20], [3, 6, 20], [3, 6, 20], [3, 6, 20]])
    pool = np.array([True, True, True, True, False])
    return RepoGraph("toy", ids, text, {k: i for i, k in enumerate(ids)}, nbrs, np.zeros((5, 4)), np.zeros(5, bool), cost, pool)


def test_budget_never_exceeded():
    g = toy_graph()
    s = np.array([3.0, 2.0, 1.0, 2.5, -1e9])
    for B in (5, 10, 25, 47, 100):
        for expand in (False, True):
            pk = S.greedy_tiers(g, S.utility(s, None, 0, 1, 1), B, [0, 3] if not expand else [0], expand=expand)
            assert S.packet_cost(g, pk) <= B
        assert S.packet_cost(g, S.flat_full(g, np.argsort(-s), B)) <= B


def test_connected_only_reaches_neighbours_and_skips_tests():
    g = toy_graph()
    s = np.array([3.0, 2.0, 1.0, 2.5, 5.0])
    pk = S.greedy_tiers(g, S.utility(s, None, 0, 1, 1), 1000, [0], expand=True)
    assert 3 not in pk  # disconnected from seed
    assert 4 not in pk  # test node excluded from pool
    assert set(pk) == {0, 1, 2}


def test_parse_patch_and_prefixes():
    p = "--- a/pkg/mod.py\n+++ b/pkg/mod.py\n@@ -1,2 +1,2 @@ def f():\n-    x = 1\n+    x = 2\n"
    f = parse_patch(p)
    assert f[0]["path"] == "pkg/mod.py" and f[0]["hunks"][0]["lines"][0] == ("-", "    x = 1")
    assert module_prefixes("docs_src/a/b.py") == ["docs_src.a.b", "a.b"]
    assert module_prefixes("rich/__init__.py") == ["rich"]


def test_tokens_and_signature():
    assert "parse" in ident_tokens("parseHeader") and "parse_header" in ident_tokens("parse_header")
    sig = signature_text('def f(a,\n      b):\n    """Doc."""\n    return a')
    assert sig.startswith("def f(a,") and '"""Doc."""' in sig and "return" not in sig
