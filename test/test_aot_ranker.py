from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import torch

from helion._testing import TestCase
from helion.autotuner._ranker_runtime import boosted_sum
from helion.autotuner._ranker_runtime import decode_matrix
from helion.autotuner._ranker_runtime import decode_pool
from helion.autotuner._ranker_runtime import forest_mean
from helion.autotuner._ranker_runtime import rank_candidates
from helion.autotuner.heuristic_generator import ShapeConfigData
from helion.autotuner.heuristic_generator import feature_to_var_name
from helion.autotuner.heuristic_generator import get_backend
from helion.autotuner.heuristic_generator import load_collect_rows
from helion.autotuner.logger import match_launch_resource_error
from helion.autotuner.ranker_backend import _hgb_trees
from helion.autotuner.ranker_backend import _rf_trees
from helion.runtime.config import Config

_BLOCKS = [16, 32, 64, 128, 256]
_WARPS = [1, 2, 4, 8]
_SHAPE_DIMS = [1, 8, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
_FEATURES = ["arg0_dim0", "arg0_dim1"]
# A config fails to launch when block * warps * rows exceeds this budget; the
# classifier has to learn this interaction rather than "bigger is riskier".
_BUDGET = 4_000_000


def _fails(config: dict, dim0: int) -> bool:
    return config["block_sizes"][0] * config["num_warps"] * dim0 > _BUDGET


def _synthetic_rows(seed: int = 0, sparsity: float = 0.45) -> list[dict]:
    """Collect-style rows: sparse, with real launch failures and a speed signal."""
    rng = np.random.default_rng(seed)
    pool = [
        {"block_sizes": [b], "num_warps": w, "num_stages": s}
        for b in _BLOCKS
        for w in _WARPS
        for s in (1, 2, 3)
    ]
    rows = []
    for si, dim0 in enumerate(_SHAPE_DIMS):
        features = {"arg0_dim0": dim0, "arg0_dim1": 64}
        for ci, config in enumerate(pool):
            if rng.random() < sparsity:
                continue  # collect data is sparse: not every pair is measured
            failed = _fails(config, dim0)
            # Larger blocks are faster, so the optimum is the largest safe block.
            timing = (
                float("inf")
                if failed
                else 0.05 + 2.0 / config["block_sizes"][0] + rng.random() * 0.002
            )
            rows.append(
                {
                    "shape_hash": f"s{si}",
                    "config_hash": f"c{ci}",
                    "config": config,
                    "shape_features": features,
                    "timing_ms": timing,
                    "ok": not failed,
                }
            )
    return rows


class _FakeKernel:
    name = "k"

    def __init__(self) -> None:
        from helion.runtime.settings import Settings

        # set_config records the winning config in `counters`, which formats the
        # kernel decorator and so needs real settings.
        self.settings = Settings()


def _make_data(rows: list[dict] | None) -> tuple[ShapeConfigData, list[Config]]:
    configs = [Config(block_sizes=[b], num_warps=1, num_stages=1) for b in _BLOCKS[:4]]
    n_shapes = len(_SHAPE_DIMS)
    return (
        ShapeConfigData(
            kernel_name="k",
            shape_features=[{"arg0_dim0": d, "arg0_dim1": 64} for d in _SHAPE_DIMS],
            timings=np.tile(
                np.array([4.0, 3.0, 2.0, 1.0]),
                (n_shapes, 1),
            ),
            configs=configs,
            shape_hashes=[f"s{i}" for i in range(n_shapes)],
            config_hashes=[f"c{i}" for i in range(len(configs))],
            selected_config_indices=list(range(len(configs))),
            collect_rows=rows,
        ),
        configs,
    )


def _generate(rows: list[dict] | None, backend: str = "ranker") -> str:
    data, configs = _make_data(rows)
    return (
        get_backend(backend)
        .generate_heuristic("k", data, configs, _FEATURES)
        .generated_code
    )


def _load(code: str, name: str):
    path = Path(tempfile.mkdtemp()) / f"{name}.py"
    path.write_text(code)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


class TestRankerRuntime(TestCase):
    """The inlined interpreter must reproduce sklearn exactly."""

    def test_forest_matches_sklearn(self) -> None:
        from sklearn.ensemble import RandomForestClassifier

        rng = np.random.default_rng(0)
        X = rng.normal(size=(300, 10))
        y = (X[:, 0] + X[:, 3] > 0).astype(int)
        model = RandomForestClassifier(
            n_estimators=10, max_depth=6, n_jobs=1, random_state=0
        ).fit(X, y)
        trees = _rf_trees(model)
        mine = np.array([forest_mean(trees, row) for row in X.tolist()])
        # Leaf values are rounded to 6dp to shrink the artifact; thresholds are
        # exact, so the routing is identical and only the value is approximate.
        self.assertLess(np.abs(mine - model.predict_proba(X)[:, 1]).max(), 1e-5)

    def test_boosted_matches_sklearn(self) -> None:
        from sklearn.ensemble import HistGradientBoostingRegressor

        rng = np.random.default_rng(0)
        X = rng.normal(size=(300, 10))
        y = X[:, 1] * 2 + rng.normal(scale=0.1, size=300)
        model = HistGradientBoostingRegressor(
            max_iter=50, max_leaf_nodes=15, random_state=0
        ).fit(X, y)
        trees, baseline = _hgb_trees(model)
        mine = np.array([boosted_sum(trees, baseline, row) for row in X.tolist()])
        # 6dp leaf rounding, summed over max_iter stages.
        self.assertLess(np.abs(mine - model.predict(X)).max(), 1e-4)

    def test_thresholds_are_not_rounded(self) -> None:
        """A rounded threshold sends samples sitting exactly on a split the wrong way.

        Both exporters must keep full precision: HistGradientBoosting places
        splits on observed feature values, so rounding one down flips every
        sample at that boundary into the other leaf.
        """
        from sklearn.ensemble import HistGradientBoostingRegressor

        rng = np.random.default_rng(0)
        X = rng.normal(size=(300, 10))
        y = X[:, 1] * 2 + rng.normal(scale=0.1, size=300)
        model = HistGradientBoostingRegressor(
            max_iter=50, max_leaf_nodes=15, random_state=0
        ).fit(X, y)
        trees, _ = _hgb_trees(model)
        raw = {
            float(nd["num_threshold"])
            for stage in model._predictors
            for nd in stage[0].nodes
            if not nd["is_leaf"]
        }
        exported = {
            t for tree in trees for f, t in zip(tree[0], tree[1], strict=True) if f >= 0
        }
        self.assertTrue(exported <= raw, "thresholds were altered during export")

    def test_gate_prefers_fast_among_safe(self) -> None:
        """Robustness gates; speed orders. A safest-first ranking would fail this."""
        # One-node tree: every candidate is safe.
        clf = [([-1], [0.0], [-1], [-1], [1.0])]
        # Root splits config feature (index 1) at 1.5; leaf values are predicted
        # slowdown, so the candidate landing right of the split is fastest.
        perf = [
            ([1, -1, -1], [1.5, 0.0, 0.0], [1, -1, -1], [2, -1, -1], [0.0, 5.0, 0.0])
        ]
        order = rank_candidates([0.0], [[0.0], [1.0], [2.0]], clf, perf, 0.0, 3)
        self.assertEqual(order[0], 2)

    def test_gate_falls_back_when_nothing_safe(self) -> None:
        """Nothing clears the gate: order by descending P(success), least-risky first."""
        # Splits config feature (index 1) at 1.5, so the lower candidates score
        # 0.5 and the higher one 0.1 -- all below the 0.9 gate.
        risky = (
            [1, -1, -1],
            [1.5, 0.0, 0.0],
            [1, -1, -1],
            [2, -1, -1],
            [0.0, 0.5, 0.1],
        )
        order = rank_candidates([0.0], [[0.0], [1.0], [2.0]], [risky], [], 0.0, 2)
        self.assertEqual(order, [0, 1])

    def test_short_safe_set_is_topped_up(self) -> None:
        """A caller handling a launch failure wants n candidates, not one.

        With a hard gate, an extreme shape can leave a single survivor; the rest
        must still be offered, most-robust-first.
        """
        # Only the candidate right of the split clears 0.9.
        clf = [
            ([1, -1, -1], [1.5, 0.0, 0.0], [1, -1, -1], [2, -1, -1], [0.0, 0.4, 1.0])
        ]
        order = rank_candidates([0.0], [[0.0], [1.0], [2.0]], clf, [], 0.0, 3)
        self.assertEqual(len(order), 3)
        self.assertEqual(order[0], 2, "the safe candidate must still come first")


class TestRankerCodegen(TestCase):
    def test_key_and_autotune_identical_to_decision_tree(self) -> None:
        """The inherited default path must be byte-identical."""
        rows = _synthetic_rows()
        ranker = _generate(rows)
        baseline = _generate(rows, backend="decision_tree")
        marker = "\n\n# --- experimental ranker fallback"
        self.assertIn(marker, ranker)
        self.assertEqual(
            ranker[: ranker.index(marker)].rstrip("\n"), baseline.rstrip("\n")
        )

    def test_fallbacks_avoid_launch_failures(self) -> None:
        module, _ = _load(_generate(_synthetic_rows()), "gen_ok")
        for dim0 in (8, 1024, 16384, 32768):
            configs = module.fallbacks_k(torch.empty((dim0, 64), device="meta"), n=5)
            self.assertTrue(configs)
            for config in configs:
                self.assertFalse(
                    _fails(config, dim0), f"dim0={dim0} picked failing {config}"
                )

    def test_fallbacks_prefer_largest_safe_block(self) -> None:
        """Speed must survive the robustness gate, and adapt to the shape."""
        module, _ = _load(_generate(_synthetic_rows()), "gen_speed")
        small = module.fallbacks_k(torch.empty((8, 64), device="meta"), n=1)
        large = module.fallbacks_k(torch.empty((32768, 64), device="meta"), n=1)
        self.assertEqual(small[0]["block_sizes"][0], 256)
        self.assertLess(large[0]["block_sizes"][0], 256)

    def test_pool_roundtrips(self) -> None:
        module, _ = _load(_generate(_synthetic_rows()), "gen_pool")
        pool = decode_pool(module._RANKER_POOL)
        feats = decode_matrix(module._RANKER_POOL_FEATS)
        # One encoded feature row per pool config, or ranking indexes the wrong
        # config -- both blobs are decoded independently at runtime.
        self.assertEqual(len(pool), len(feats))
        self.assertEqual(len({len(r) for r in feats}), 1, "ragged feature matrix")
        self.assertTrue(all("block_sizes" in c for c in pool))

    def test_feature_matrix_is_compressed_not_a_literal(self) -> None:
        """A plain literal matrix cost 601 KB of a 721 KB artifact on real data."""
        module, _ = _load(_generate(_synthetic_rows()), "gen_feats")
        self.assertIsInstance(module._RANKER_POOL_FEATS, str)
        self.assertLess(
            len(module._RANKER_POOL_FEATS),
            len(repr(decode_matrix(module._RANKER_POOL_FEATS))),
        )

    def test_artifact_size_and_node_count_bounded(self) -> None:
        """An unset max_depth silently produces a multi-MB artifact.

        Budget checked on real paged_attention collect data (2430 configs,
        1417 kept): 151.8 KB. Every embedded bulk payload must be compressed --
        emitting the config feature matrix as a plain literal instead cost
        601 KB on its own and pushed the artifact to 721 KB.
        """
        code = _generate(_synthetic_rows())
        module, _ = _load(code, "gen_size")
        self.assertLess(len(code.encode()), 256 * 1024)
        self.assertLessEqual(sum(len(t[0]) for t in module._RANKER_CLF), 1500)
        # No bulk payload may be emitted as an uncompressed Python literal.
        for name in ("_RANKER_POOL", "_RANKER_POOL_FEATS"):
            self.assertIsInstance(getattr(module, name), str, f"{name} not a blob")

    def test_shape_feature_order_matches_training(self) -> None:
        """A silent train/inference feature-order mismatch would be undetectable.

        The emitted shape vector must list features in the same order the model
        was trained on, so permuting the declared features must change the
        ranking rather than being silently absorbed.
        """
        rows = _synthetic_rows()
        code = _generate(rows)
        # The emitted call passes shape features positionally, in feature order.
        vec = code.split("order = rank_candidates(")[1].split("\n")[1]
        self.assertEqual(
            vec.strip().rstrip(","),
            "[" + ", ".join(feature_to_var_name(f) for f in _FEATURES) + "]",
        )

    def test_missing_shape_feature_degrades(self) -> None:
        """A shape lacking a selected feature must default to 0, not raise."""
        module, _ = _load(_generate(_synthetic_rows()), "gen_missing")
        # A 1-D tensor has no dim1, so arg0_dim1 cannot be extracted.
        configs = module.fallbacks_k(torch.empty((1024,), device="meta"), n=3)
        self.assertTrue(configs)

    def test_no_collect_rows_emits_plain_heuristic(self) -> None:
        code = _generate(None)
        self.assertNotIn("fallbacks_k", code)
        self.assertEqual(code, _generate(None, backend="decision_tree"))

    def test_converged_data_is_rejected(self) -> None:
        """LFBO-style data (~1 shape per config) cannot train the classifier."""
        rows = [
            {
                "shape_hash": f"s{i}",
                "config_hash": f"c{i}",
                "config": {"block_sizes": [16], "num_warps": 4, "num_stages": 1},
                "shape_features": {"arg0_dim0": 1024, "arg0_dim1": 64},
                "timing_ms": 0.1,
                "ok": True,
            }
            for i in range(40)
        ]
        self.assertNotIn("fallbacks_k", _generate(rows))


class TestCollectRowLoader(TestCase):
    def _write(self, text: str) -> Path:
        path = Path(tempfile.mkdtemp()) / "measurements_collect.csv"
        path.write_text(text)
        return path

    def test_infers_ok_from_timing_without_status_column(self) -> None:
        path = self._write(
            "kernel_name,shape_hash,config_hash,config,shape_features,timing_ms\n"
            'k,s0,c0,"{""num_warps"": 4}","{""arg0_dim0"": 8}",0.5\n'
            'k,s0,c1,"{""num_warps"": 8}","{""arg0_dim0"": 8}",inf\n'
        )
        rows = load_collect_rows(path)["k"]
        self.assertEqual([r["ok"] for r in rows], [True, False])

    def test_status_column_filters_non_launches(self) -> None:
        """`filtered` never launched, so it is not a failure; `deduplicated` is a success."""
        path = self._write(
            "kernel_name,shape_hash,config_hash,config,shape_features,timing_ms,status\n"
            'k,s0,c0,"{}","{""arg0_dim0"": 8}",0.5,ok\n'
            'k,s0,c1,"{}","{""arg0_dim0"": 8}",inf,filtered\n'
            'k,s0,c2,"{}","{""arg0_dim0"": 8}",0.4,deduplicated\n'
            'k,s0,c3,"{}","{""arg0_dim0"": 8}",inf,error\n'
        )
        rows = load_collect_rows(path)["k"]
        self.assertEqual([r["config_hash"] for r in rows], ["c0", "c2", "c3"])
        self.assertEqual([r["ok"] for r in rows], [True, True, False])

    def test_missing_file_is_empty(self) -> None:
        self.assertEqual(load_collect_rows(Path("/nonexistent/x.csv")), {})


class TestLaunchErrorMatching(TestCase):
    def test_matches_launch_resource_errors(self) -> None:
        for message in (
            "out of resource: shared memory",
            "too many resources requested for launch",
            "exceeds triton maximum tensor numel",
            "too many blocks in cooperative launch",
        ):
            self.assertTrue(match_launch_resource_error(RuntimeError(message)))

    def test_ignores_unrelated_errors(self) -> None:
        self.assertFalse(match_launch_resource_error(RuntimeError("unrelated")))
        self.assertFalse(
            match_launch_resource_error(RuntimeError("illegal memory access"))
        )


class TestFallbackGating(TestCase):
    """The env var alone decides whether the retry wrapper is installed."""

    def test_disabled_without_env_var(self) -> None:
        from helion.runtime.kernel import BoundKernel

        bound = object.__new__(BoundKernel)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HELION_AOT_RANKER_FALLBACK", None)
            self.assertEqual(BoundKernel._ranker_fallback_configs(bound, Config()), [])

    def test_retries_until_one_succeeds(self) -> None:
        from helion.runtime.kernel import BoundKernel

        good = Config(block_sizes=[16], num_warps=1)
        bad = Config(block_sizes=[256], num_warps=8)
        attempts: list[Config] = []

        def compile_config(_self: object, config: Config):
            attempts.append(config)

            def run(*args: object) -> str:
                if config != good:
                    raise RuntimeError("out of resource: shared memory")
                return "ok"

            return run

        bound = object.__new__(BoundKernel)
        bound.kernel = _FakeKernel()
        with patch.object(BoundKernel, "compile_config", compile_config):
            primary = compile_config(bound, bad)
            wrapped = BoundKernel._run_with_fallbacks(bound, primary, [bad, good])
            self.assertEqual(wrapped(), "ok")
        # primary, then each fallback in order until one works
        self.assertEqual(attempts, [bad, bad, good])
        self.assertEqual(bound._config, good)

    def test_retry_capability_survives_a_win(self) -> None:
        """A BoundKernel is shared across shapes, so it must stay wrapped.

        Promoting the bare compiled callable into ``_run`` would silently drop
        the untried candidates after the first successful fallback.
        """
        from helion.runtime.kernel import BoundKernel

        good = Config(block_sizes=[16], num_warps=1)
        mid = Config(block_sizes=[64], num_warps=2)
        bad = Config(block_sizes=[256], num_warps=8)
        allowed = {mid}

        def compile_config(_self: object, config: Config):
            def run(*args: object) -> str:
                if config not in allowed:
                    raise RuntimeError("out of resource: shared memory")
                return repr(config)

            return run

        bound = object.__new__(BoundKernel)
        bound.kernel = _FakeKernel()
        with patch.object(BoundKernel, "compile_config", compile_config):
            primary = compile_config(bound, bad)
            wrapped = BoundKernel._run_with_fallbacks(bound, primary, [mid, good])
            self.assertEqual(wrapped(), repr(mid))
            # mid now works and was promoted. Make it start failing, as a bigger
            # shape hitting the same BoundKernel would: `good` must still be
            # reachable through the promoted callable.
            allowed.clear()
            allowed.add(good)
            self.assertEqual(bound._run(), repr(good))
            self.assertEqual(bound._config, good)

    def test_non_launch_error_propagates(self) -> None:
        from helion.runtime.kernel import BoundKernel

        def primary(*args: object) -> str:
            raise RuntimeError("unrelated failure")

        bound = object.__new__(BoundKernel)
        bound.kernel = _FakeKernel()
        wrapped = BoundKernel._run_with_fallbacks(
            bound, primary, [Config(block_sizes=[16])]
        )
        with self.assertRaisesRegex(RuntimeError, "unrelated failure"):
            wrapped()

    def test_exhausting_candidates_reraises(self) -> None:
        from helion.runtime.kernel import BoundKernel

        def always_fail(*args: object) -> str:
            raise RuntimeError("out of resource: shared memory")

        bound = object.__new__(BoundKernel)
        bound.kernel = _FakeKernel()
        with patch.object(BoundKernel, "compile_config", lambda self, c: always_fail):
            wrapped = BoundKernel._run_with_fallbacks(
                bound, always_fail, [Config(block_sizes=[16])]
            )
            with self.assertRaisesRegex(RuntimeError, "out of resource"):
                wrapped()


class TestMeasurementRoundTrip(TestCase):
    """The writer and the loader must agree on the status column.

    Without this the loader's status branch is dead on every CSV the pipeline
    actually produces, and `filtered` configs -- which never launched -- become
    negative training examples for launch robustness.
    """

    def _write_rows(self, results: list[tuple[Config, float, str]]) -> Path:
        from helion.autotuner.aot_cache import AOTAutotuneCache
        from helion.autotuner.aot_cache import ShapeKey

        cache = object.__new__(AOTAutotuneCache)
        path = Path(tempfile.mkdtemp()) / "measurements_collect.csv"
        key = ShapeKey(kernel_name="k", specialization_key=(8,), hardware_id="hw")
        with patch.object(
            type(cache), "_measurements_file", property(lambda self: path)
        ):
            for config, timing, status in results:
                cache._save_measurement(
                    kernel_name="k",
                    shape_key=key,
                    config=config,
                    timing_ms=timing,
                    shape_features={"arg0_dim0": 8},
                    status=status,
                )
        return path

    def test_writer_emits_status_and_loader_reads_it(self) -> None:
        path = self._write_rows(
            [
                (Config(num_warps=4), 0.5, "ok"),
                (Config(num_warps=8), float("inf"), "filtered"),
                (Config(num_warps=2), float("inf"), "error"),
                (Config(num_warps=1), 0.4, "deduplicated"),
                (Config(num_stages=3), float("inf"), "timeout"),
            ]
        )
        with open(path) as f:
            self.assertIn("status", f.readline())
        rows = load_collect_rows(path)["k"]
        # `filtered` and `timeout` never launched, so they carry no robustness
        # signal and are dropped rather than labelled as failures.
        self.assertEqual([r["ok"] for r in rows], [True, False, True])

    def test_default_status_keeps_rows_labelled(self) -> None:
        """The measure path calls _save_measurement without a status."""
        path = self._write_rows([(Config(num_warps=4), 0.5, "ok")])
        self.assertEqual(load_collect_rows(path)["k"][0]["ok"], True)


class TestFallbackKeying(TestCase):
    """One BoundKernel is shared by many shapes, so the key must be per-shape."""

    def test_uses_last_resolved_shape_not_first(self) -> None:
        from helion.autotuner.aot_cache import AOTAutotuneCache
        from helion.runtime.kernel import BoundKernel

        small = [Config(block_sizes=[16])]
        large = [Config(block_sizes=[256])]

        def _fn() -> None:
            return None

        src, name = _fn.__code__.co_filename, "k"
        try:
            AOTAutotuneCache._fallback_configs = {
                (src, name, "shapeA"): large,
                (src, name, "shapeB"): small,
            }
            AOTAutotuneCache._last_fallback_key = (src, name, "shapeB")
            bound = object.__new__(BoundKernel)
            bound.kernel = _FakeKernel()
            bound.kernel.fn = _fn
            with patch.dict(os.environ, {"HELION_AOT_RANKER_FALLBACK": "5"}):
                got = BoundKernel._ranker_fallback_configs(bound, Config())
            # shapeB resolved last, so its ranking must win over shapeA's, which
            # a scan over the dict by kernel name would have returned first.
            self.assertEqual(got, small)
        finally:
            AOTAutotuneCache.clear_caches()

    def test_ignores_entry_for_another_kernel(self) -> None:
        from helion.autotuner.aot_cache import AOTAutotuneCache
        from helion.runtime.kernel import BoundKernel

        try:
            key = ("/other/file.py", "other", "s0")
            AOTAutotuneCache._fallback_configs = {key: [Config(block_sizes=[16])]}
            AOTAutotuneCache._last_fallback_key = key
            bound = object.__new__(BoundKernel)
            bound.kernel = _FakeKernel()
            bound.kernel.fn = lambda: None
            with patch.dict(os.environ, {"HELION_AOT_RANKER_FALLBACK": "5"}):
                self.assertEqual(
                    BoundKernel._ranker_fallback_configs(bound, Config()), []
                )
        finally:
            AOTAutotuneCache.clear_caches()

    def test_clear_caches_resets_slot(self) -> None:
        from helion.autotuner.aot_cache import AOTAutotuneCache

        AOTAutotuneCache._fallback_configs[("f", "k", "s")] = [Config()]
        AOTAutotuneCache._last_fallback_key = ("f", "k", "s")
        AOTAutotuneCache.clear_caches()
        self.assertEqual(AOTAutotuneCache._fallback_configs, {})
        self.assertIsNone(AOTAutotuneCache._last_fallback_key)
