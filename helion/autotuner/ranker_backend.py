"""
Ranker Backend for AOT Autotuning (experimental)
================================================

Extends the decision tree backend with a robustness classifier and a
performance model trained on collect-phase measurements. The generated file
keeps ``key_*``/``autotune_*`` byte-identical to ``decision_tree`` and adds
``fallbacks_<kernel>(*args)``, which returns configs to try when the primary
config fails to launch.

Experimental: selected via ``--backend ranker``; delete this file plus the
registry entry to remove.
"""

from __future__ import annotations

import base64
import gzip
import json
import logging
import math
from typing import TYPE_CHECKING
from typing import Any

import numpy as np

from .decision_tree_backend import DecisionTreeBackend
from .heuristic_generator import HeuristicBackendResult
from .heuristic_generator import feature_to_var_name
from .heuristic_generator import generate_feature_extraction_code

if TYPE_CHECKING:
    from ..runtime.config import Config
    from .heuristic_generator import ShapeConfigData

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.ensemble import RandomForestClassifier

    HAS_ML_DEPS = True
except ImportError as e:  # pragma: no cover - sklearn is a hard dependency
    HAS_ML_DEPS = False
    _IMPORT_ERROR = e

log: logging.Logger = logging.getLogger(__name__)

_UNKNOWN = "__unknown__"
_MISSING = "__missing__"

# Hyperparameters are measured, not defaults: an unset max_depth grows a
# ~3.4MB artifact for no accuracy gain, and unbounded n_jobs oversubscribes
# shared build machines.
_CLF_KWARGS: dict[str, Any] = {
    "n_estimators": 10,
    "max_depth": 6,
    "n_jobs": 1,
    "random_state": 0,
}
_PERF_KWARGS: dict[str, Any] = {
    "max_iter": 50,
    "max_leaf_nodes": 15,
    "random_state": 0,
}
# The models learn over config *features*, which generalize across shapes even
# when no config identity repeats (converged searches like LFBO measure almost
# every config at exactly one shape). What must hold is that there is more than
# one shape to condition on and both launch outcomes are represented.
_MIN_SHAPES = 2
_MAX_POOL_CONFIGS = 4096


def _flatten(prefix: str, value: object, out: dict[str, object]) -> None:
    if isinstance(value, list):
        for i, v in enumerate(value):
            _flatten(f"{prefix}_{i}", v, out)
    else:
        out[prefix] = value


class _ConfigEncoder:
    """Flattens config dicts to a fixed float vector (one-hot for categoricals)."""

    def fit(self, configs: list[dict[str, Any]]) -> _ConfigEncoder:
        rows = [self._flat(c) for c in configs]
        self._columns = sorted({k for r in rows for k in r})
        self._is_numeric = {
            c: all(isinstance(r.get(c, 0), (int, float)) for r in rows)
            for c in self._columns
        }
        self._categories: dict[str, list[str]] = {}
        for c in self._columns:
            if not self._is_numeric[c]:
                values = sorted({str(r.get(c, _MISSING)) for r in rows})
                self._categories[c] = [*values, _UNKNOWN]
        self.dim = sum(
            1 if self._is_numeric[c] else len(self._categories[c])
            for c in self._columns
        )
        return self

    @staticmethod
    def _flat(config: dict[str, Any]) -> dict[str, object]:
        flat: dict[str, object] = {}
        for k, v in config.items():
            _flatten(k, v, flat)
        return flat

    def transform(self, config: dict[str, Any]) -> list[float]:
        flat = self._flat(config)
        out: list[float] = []
        for c in self._columns:
            if self._is_numeric[c]:
                value = flat.get(c)
                out.append(float(value) if isinstance(value, (int, float)) else 0.0)
            else:
                cats = self._categories[c]
                value = str(flat.get(c, _MISSING))
                idx = cats.index(value) if value in cats else len(cats) - 1
                out.extend(1.0 if i == idx else 0.0 for i in range(len(cats)))
        return out

    def to_literal(self) -> str:
        return repr(
            {
                "columns": self._columns,
                "is_numeric": self._is_numeric,
                "categories": self._categories,
            }
        )


def _gzip_b64(raw: bytes) -> str:
    """gzip+base64 a payload for embedding as a Python string literal."""
    return base64.b64encode(gzip.compress(raw, 9)).decode()


def _rf_trees(model: Any) -> list[Any]:
    """Export a RandomForestClassifier as flat per-tree arrays (P(class=1))."""
    trees = []
    for est in model.estimators_:
        t = est.tree_
        counts = t.value  # (n_nodes, 1, n_classes)
        if counts.shape[2] == 1:
            values = [1.0 if model.classes_[0] else 0.0] * counts.shape[0]
        else:
            values = [float(c[0][1] / s) if (s := c[0].sum()) else 0.0 for c in counts]
        trees.append(
            (
                t.feature.tolist(),
                t.threshold.tolist(),  # never rounded: see _hgb_trees
                t.children_left.tolist(),
                t.children_right.tolist(),
                [round(v, 6) for v in values],
            )
        )
    return trees


def _hgb_trees(model: Any) -> tuple[list[Any], float]:
    """Export a HistGradientBoostingRegressor as flat per-tree arrays.

    Uses the private ``_predictors`` attribute -- the only access sklearn
    offers. Guarded by a codegen test that compares against ``predict``.

    Two preconditions the interpreter in ``_ranker_runtime`` relies on, both
    enforced here rather than assumed:

    * No categorical splits. ``_PERF_KWARGS`` passes no ``categorical_features``,
      so every ``is_categorical`` is 0 and ``bitset_idx`` is unused; the
      interpreter routes purely on ``<=``.
    * No missing values. sklearn routes NaN by the learned
      ``missing_go_to_left``, which is not exported, so a NaN feature would
      diverge. The fit is asserted finite instead.
    """
    trees = []
    for stage in model._predictors:
        nodes = stage[0].nodes
        feature, threshold, left, right, value = [], [], [], [], []
        for nd in nodes:
            leaf = bool(nd["is_leaf"])
            if not leaf and nd["is_categorical"]:
                raise ValueError("categorical splits are not exportable")
            feature.append(-1 if leaf else int(nd["feature_idx"]))
            # Thresholds must keep full precision. Both sklearn tree flavours
            # can place a split exactly on an observed value, so rounding one
            # down flips every sample sitting on that boundary to the wrong
            # leaf -- a wrong prediction, not a small numeric error.
            threshold.append(0.0 if leaf else float(nd["num_threshold"]))
            left.append(int(nd["left"]))
            right.append(int(nd["right"]))
            value.append(round(float(nd["value"]), 6) if leaf else 0.0)
        trees.append((feature, threshold, left, right, value))
    return trees, round(float(model._baseline_prediction.ravel()[0]), 6)


class RankerBackend(DecisionTreeBackend):
    """Decision tree plus an ensemble that ranks fallback configs."""

    name: str = "ranker"

    def generate_heuristic(
        self,
        kernel_name: str,
        data: ShapeConfigData,
        selected_configs: list[Config],
        feature_names: list[str],
    ) -> HeuristicBackendResult:
        result = super().generate_heuristic(
            kernel_name, data, selected_configs, feature_names
        )
        rows = getattr(data, "collect_rows", None)
        if not rows:
            log.warning(
                "Ranker backend: no collect-phase rows available; emitting a "
                "plain decision-tree heuristic. Re-run collect with "
                "HELION_COLLECT_ALL_MEASUREMENTS=1 and HELION_AUTOTUNER=RandomSearch."
            )
            return result
        if not HAS_ML_DEPS:  # pragma: no cover
            log.warning(f"Ranker backend: sklearn unavailable ({_IMPORT_ERROR})")
            return result

        try:
            extra = self._generate_ranker_code(kernel_name, rows, feature_names)
        except Exception as e:
            log.warning(f"Ranker backend: failed to train ranker ({e}); skipping")
            return result

        if extra is None:
            return result
        return HeuristicBackendResult(
            generated_code=result.generated_code + extra,
            model_accuracy=result.model_accuracy,
            feature_names=result.feature_names,
        )

    def _generate_ranker_code(
        self,
        kernel_name: str,
        rows: list[dict[str, Any]],
        feature_names: list[str],
    ) -> str | None:
        # Only features the extraction codegen actually assigns can appear in
        # the emitted shape vector; anything else would be a NameError at call
        # time. Filtering here keeps training and inference on one list.
        extract = generate_feature_extraction_code(feature_names)
        feature_names = [
            f for f in feature_names if f"{feature_to_var_name(f)} = " in extract
        ]
        if not feature_names:
            log.warning("Ranker backend: no extractable shape features")
            return None
        extract = generate_feature_extraction_code(feature_names)

        pool, encoder, X, ok, slowdown = self._prepare_training_data(
            rows, feature_names
        )
        if pool is None:
            return None
        assert encoder is not None

        clf = RandomForestClassifier(**_CLF_KWARGS).fit(X, ok)
        clf_trees = _rf_trees(clf)
        n_nodes = sum(len(t[0]) for t in clf_trees)
        log.info(
            f"Ranker backend: classifier {n_nodes} nodes, "
            f"{int(ok.sum())}/{len(ok)} successful rows"
        )

        fit_mask = np.isfinite(slowdown)
        perf_trees: list[tuple[Any, ...]] = []
        perf_baseline = 0.0
        if fit_mask.sum() >= 10:
            perf = HistGradientBoostingRegressor(**_PERF_KWARGS).fit(
                X[fit_mask], slowdown[fit_mask]
            )
            perf_trees, perf_baseline = _hgb_trees(perf)

        pool_feats = [encoder.transform(c) for c in pool]
        blob = _gzip_b64(
            "\n".join(json.dumps(c, sort_keys=True) for c in pool).encode()
        )
        # The encoder emits categorical indices and power-of-two sizes, so the
        # matrix is integral; store it as ints to keep the blob small, and fall
        # back to floats if a future feature breaks that assumption. The test
        # must run before the cast: int() raises on NaN/inf, which a user config
        # kwarg can carry all the way through the encoder.
        integral = all(
            math.isfinite(v) and float(v).is_integer()
            for row in pool_feats
            for v in row
        )
        matrix: list[list[float]] | list[list[int]] = (
            [[int(v) for v in row] for row in pool_feats] if integral else pool_feats
        )
        feats_blob = _gzip_b64(json.dumps(matrix, separators=(",", ":")).encode())

        return self._emit(
            kernel_name=kernel_name,
            feature_names=feature_names,
            extract=extract,
            blob=blob,
            feats_blob=feats_blob,
            clf_trees=clf_trees,
            perf_trees=perf_trees,
            perf_baseline=perf_baseline,
        )

    def _prepare_training_data(
        self, rows: list[dict[str, Any]], feature_names: list[str]
    ) -> tuple[
        list[dict[str, Any]] | None,
        _ConfigEncoder | None,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Build (pool, encoder, X, success_labels, log-relative-slowdown).

        Trains from CSV rows, not the dense timing matrix: ``load_measurements``
        fills unmeasured pairs with ``inf``, which is indistinguishable from a
        genuine failure and would label a sparse collect matrix ~all-failed.
        """
        empty = np.empty(0)
        n_shapes = len({r["shape_hash"] for r in rows})
        if n_shapes < _MIN_SHAPES:
            log.warning(
                f"Ranker backend: collect data covers {n_shapes} shape(s); "
                "need at least two to condition on shape."
            )
            return None, None, empty, empty, empty
        if len({bool(r["ok"]) for r in rows}) < 2:
            log.warning(
                "Ranker backend: collect data has a single launch outcome; "
                "nothing for the robustness classifier to separate."
            )
            return None, None, empty, empty, empty

        # Best timing per shape, for the scale-free performance target.
        best: dict[str, float] = {}
        for r in rows:
            t = r["timing_ms"]
            if np.isfinite(t) and t > 0:
                best[r["shape_hash"]] = min(best.get(r["shape_hash"], np.inf), t)

        pool_map: dict[str, dict[str, Any]] = {}
        succ: dict[str, int] = {}
        seen: dict[str, int] = {}
        for r in rows:
            pool_map.setdefault(r["config_hash"], r["config"])
            seen[r["config_hash"]] = seen.get(r["config_hash"], 0) + 1
            if r["ok"]:
                succ[r["config_hash"]] = succ.get(r["config_hash"], 0) + 1

        # Keep configs that ever worked; cap by success rate so a large pool
        # cannot silently inflate the artifact.
        keep = [h for h in pool_map if succ.get(h, 0) > 0]
        if not keep:
            log.warning("Ranker backend: no successful configs in collect data")
            return None, None, empty, empty, empty
        keep.sort(key=lambda h: -succ.get(h, 0) / seen[h])
        keep = keep[:_MAX_POOL_CONFIGS]
        pool = [pool_map[h] for h in keep]

        encoder = _ConfigEncoder().fit(list(pool_map.values()))
        X = np.array(
            [
                [r["shape_features"].get(f, 0) for f in feature_names]
                + encoder.transform(r["config"])
                for r in rows
            ],
            dtype=np.float64,
        )
        if not np.isfinite(X).all():
            # The exported trees route purely on `<=`, which cannot reproduce
            # sklearn's learned NaN direction. See _hgb_trees.
            log.warning("Ranker backend: non-finite shape/config features")
            return None, None, empty, empty, empty
        ok = np.array([bool(r["ok"]) for r in rows])
        slowdown = np.array(
            [
                np.log(r["timing_ms"] / best[r["shape_hash"]])
                if (
                    r["ok"]
                    and np.isfinite(r["timing_ms"])
                    and best.get(r["shape_hash"], 0) > 0
                )
                else np.inf
                for r in rows
            ],
            dtype=np.float64,
        )
        log.info(
            f"Ranker backend: {len(rows)} rows, {len(pool_map)} configs "
            f"({len(pool)} kept), {n_shapes} shapes, "
            f"{100 * (1 - ok.mean()):.1f}% launch failures"
        )
        return pool, encoder, X, ok, slowdown

    def _emit(
        self,
        *,
        kernel_name: str,
        feature_names: list[str],
        extract: str,
        blob: str,
        feats_blob: str,
        clf_trees: list[tuple[Any, ...]],
        perf_trees: list[tuple[Any, ...]],
        perf_baseline: float,
    ) -> str:
        shape_vec = ", ".join(feature_to_var_name(f) for f in feature_names)
        return f'''

# --- experimental ranker fallback -------------------------------------------
_RANKER_POOL = "{blob}"
_RANKER_POOL_FEATS = "{feats_blob}"
_RANKER_CLF = {clf_trees!r}
_RANKER_PERF = {perf_trees!r}
_RANKER_PERF_BASELINE = {perf_baseline!r}
_RANKER_CONFIGS = None
_RANKER_FEATS = None


def fallbacks_{kernel_name}(*args, n: int = 5) -> list:
    """Configs to try, in ensemble order, if the primary config fails to launch."""
    global _RANKER_CONFIGS, _RANKER_FEATS
    from helion.autotuner._ranker_runtime import decode_matrix
    from helion.autotuner._ranker_runtime import decode_pool
    from helion.autotuner._ranker_runtime import rank_candidates

    if _RANKER_CONFIGS is None:
        _RANKER_CONFIGS = decode_pool(_RANKER_POOL)
        _RANKER_FEATS = decode_matrix(_RANKER_POOL_FEATS)
{extract}
    order = rank_candidates(
        [{shape_vec}],
        _RANKER_FEATS,
        _RANKER_CLF,
        _RANKER_PERF,
        _RANKER_PERF_BASELINE,
        n,
    )
    return [_RANKER_CONFIGS[i] for i in order]
'''
