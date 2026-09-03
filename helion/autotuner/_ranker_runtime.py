"""
Runtime support for the experimental ``ranker`` heuristic backend.

Generated heuristic files embed tree ensembles as flat arrays plus a
gzip-compressed config pool; this module decodes them and scores candidates.
Kept separate from the generated file so the interpreter is unit-tested once
instead of re-emitted per kernel.

Experimental: removable together with ``ranker_backend.py``.
"""

from __future__ import annotations

import base64
import gzip
import json
from typing import Any
from typing import Sequence

# A tree is (feature, threshold, left, right, value) as flat tuples, mirroring
# sklearn's ``tree_`` arrays. feature[i] < 0 marks a leaf.
Tree = tuple[
    Sequence[int], Sequence[float], Sequence[int], Sequence[int], Sequence[float]
]


def decode_pool(blob: str) -> list[dict[str, Any]]:
    """Decode a gzip+base64 config pool into config dicts."""
    raw = gzip.decompress(base64.b64decode(blob)).decode()
    return [json.loads(line) for line in raw.splitlines() if line]


def decode_matrix(blob: str) -> list[list[float]]:
    """Decode a gzip+base64 integer matrix (the pool's config features).

    The matrix is 1400+ rows x ~86 columns of small integers. Written as a
    Python literal it costs ~600 KB -- 19x more than this encoding -- and
    dominates the artifact, so it ships compressed like the pool itself.
    """
    return json.loads(gzip.decompress(base64.b64decode(blob)).decode())


def _tree_value(tree: Tree, x: Sequence[float]) -> float:
    feature, threshold, left, right, value = tree
    node = 0
    while feature[node] >= 0:
        node = left[node] if x[feature[node]] <= threshold[node] else right[node]
    return value[node]


def forest_mean(trees: Sequence[Tree], x: Sequence[float]) -> float:
    """Mean leaf value over ``trees`` (sklearn RandomForest aggregation)."""
    return sum(_tree_value(t, x) for t in trees) / len(trees)


def boosted_sum(trees: Sequence[Tree], baseline: float, x: Sequence[float]) -> float:
    """Baseline plus the sum of leaf values (sklearn HistGradientBoosting)."""
    return baseline + sum(_tree_value(t, x) for t in trees)


def rank_candidates(
    shape_features: Sequence[float],
    config_features: Sequence[Sequence[float]],
    clf_trees: Sequence[Tree],
    perf_trees: Sequence[Tree],
    perf_baseline: float,
    n: int,
    min_p_success: float = 0.9,
) -> list[int]:
    """Indices of the ``n`` best candidates: fastest among those likely to launch.

    Gates on predicted success probability, then orders the whole surviving set
    by predicted relative slowdown. Gating on a threshold rather than taking the
    top ``n`` by robustness matters: the safest configs are systematically the
    smallest/slowest, so a robustness-first top-``n`` would hand the speed model
    only the slowest candidates and always return the smallest config.

    When fewer than ``n`` candidates clear the gate, the rest are appended
    most-robust-first rather than returning a short list: the caller is already
    handling a launch failure, so a risky candidate beats no candidate.
    """
    shape = list(shape_features)
    rows = [shape + list(cf) for cf in config_features]
    p_success = [forest_mean(clf_trees, r) for r in rows]

    safe = {i for i, p in enumerate(p_success) if p >= min_p_success}
    if safe and perf_trees:
        order = sorted(
            safe, key=lambda i: boosted_sum(perf_trees, perf_baseline, rows[i])
        )
    else:
        order = sorted(safe, key=lambda i: -p_success[i])
    if len(order) < n:
        order += sorted(set(range(len(rows))) - safe, key=lambda i: -p_success[i])
    return order[:n]
