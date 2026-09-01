#!/usr/bin/env python3
# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tests for SpyreAsyncCompile persistent kernel caching.

Each test runs inside ``torch._inductor.utils.fresh_cache()``, which redirects
all Inductor / Spyre cache I/O to a fresh temporary directory.  The developer's
real cache is never read or modified by these tests.

Run with:
    python -m pytest tests/test_kernel_caching.py -v
    python -m pytest tests/test_kernel_caching.py -v -k test_cache_hit
"""

import os
import unittest
import warnings

import torch
import torch_spyre  # noqa: F401 — side-effects: registers Spyre backend

from torch._inductor.utils import fresh_cache

from torch_spyre.execution.kernel_cache import (
    allocate_compile_dir,
    commit_compile_dir,
    get_cache_root_dir,
    get_cache_stats,
    get_cached_kernel_dir,
)

DEVICE = torch.device("spyre")


def _simple_fn(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=-1)


def _make_input(shape=(64, 512), dtype=torch.float16):
    return torch.rand(*shape, dtype=dtype).to(DEVICE)


class TestCacheMissOnColdStart(unittest.TestCase):
    def test_cache_is_empty_before_first_compile(self):
        """Cache must be empty before any torch.compile() is called."""
        with fresh_cache():
            torch._dynamo.reset()
            stats = get_cache_stats()
            self.assertEqual(stats["total_cached_kernels"], 0)

    def test_first_compile_populates_cache(self):
        """After one torch.compile() run, at least one kernel should be cached."""
        with fresh_cache():
            torch._dynamo.reset()
            compiled = torch.compile(_simple_fn)
            compiled(_make_input())

            stats = get_cache_stats()
            self.assertGreater(
                stats["total_cached_kernels"],
                0,
                "Expected at least one kernel in cache after first compile",
            )


class TestCacheArtifactCompleteness(unittest.TestCase):
    REQUIRED = [
        "bundle.mlir",
        os.path.join("spyreCodeDir", "init_binary.bin"),
        os.path.join("spyreCodeDir", "spyrecode.json"),
    ]

    def test_all_required_artifacts_present(self):
        """Every cached kernel directory must contain all required artifacts."""
        with fresh_cache():
            torch._dynamo.reset()
            torch.compile(_simple_fn)(_make_input())

            cache_root = get_cache_root_dir()
            cached_entries = [
                d
                for d in os.listdir(cache_root)
                if os.path.isdir(os.path.join(cache_root, d))
            ]
            self.assertGreater(len(cached_entries), 0, "No cached entries found")

            for entry in cached_entries:
                entry_dir = os.path.join(cache_root, entry)
                for artifact in self.REQUIRED:
                    self.assertTrue(
                        os.path.isfile(os.path.join(entry_dir, artifact)),
                        f"Missing artifact '{artifact}' in cache entry '{entry}'",
                    )

                has_sdsc = any(
                    f.startswith("sdsc_") and f.endswith(".json")
                    for f in os.listdir(entry_dir)
                )
                self.assertTrue(
                    has_sdsc,
                    f"No sdsc_N.json files found in cache entry '{entry}'",
                )


class TestPartialCacheEntryTreatedAsMiss(unittest.TestCase):
    def test_partial_write_does_not_produce_cache_hit(self):
        """A directory missing spyreCodeDir/init_binary.bin must be a cache miss."""
        with fresh_cache():
            cache_root = get_cache_root_dir()
            fake_key = "c" + "a" * 63
            fake_dir = os.path.join(cache_root, fake_key)
            os.makedirs(os.path.join(fake_dir, "spyreCodeDir"), exist_ok=True)

            with open(os.path.join(fake_dir, "bundle.mlir"), "w") as f:
                f.write("fake bundle")
            with open(os.path.join(fake_dir, "sdsc_0.json"), "w") as f:
                f.write("{}")
            with open(
                os.path.join(fake_dir, "spyreCodeDir", "spyrecode.json"), "w"
            ) as f:
                f.write("{}")
            # init_binary.bin intentionally missing

            result = get_cached_kernel_dir(fake_key)
            self.assertIsNone(
                result,
                "Expected cache miss for partial entry missing init_binary.bin",
            )


class TestCacheDisabledViaConfig(unittest.TestCase):
    def test_cache_disabled_leaves_cache_empty(self):
        """With spyre_kernel_cache=False, the kernel cache must remain empty."""
        import torch_spyre._inductor.config as spyre_config

        with fresh_cache():
            torch._dynamo.reset()
            original = spyre_config.spyre_kernel_cache
            spyre_config.spyre_kernel_cache = False
            try:
                torch.compile(_simple_fn)(_make_input())
                self.assertEqual(
                    get_cache_stats()["total_cached_kernels"],
                    0,
                    "Expected empty cache when spyre_kernel_cache=False",
                )
            finally:
                spyre_config.spyre_kernel_cache = original


class TestForceDisableCaches(unittest.TestCase):
    def test_force_disable_caches_leaves_cache_empty(self):
        """torch._inductor.config.force_disable_caches must bypass the Spyre cache."""
        with fresh_cache():
            torch._dynamo.reset()
            with torch._inductor.config.patch({"force_disable_caches": True}):
                torch.compile(_simple_fn)(_make_input())

            self.assertEqual(
                get_cache_stats()["total_cached_kernels"],
                0,
                "Expected empty cache when force_disable_caches=True",
            )


class TestDifferentOpsProduceDifferentKeys(unittest.TestCase):
    def test_softmax_and_relu_have_different_cache_entries(self):
        """Two different ops must not share a cache entry."""
        with fresh_cache():
            torch._dynamo.reset()
            x = _make_input()

            torch.compile(lambda a: torch.softmax(a, dim=-1))(x)
            count_after_softmax = get_cache_stats()["total_cached_kernels"]

            torch._dynamo.reset()
            torch.compile(lambda a: torch.relu(a))(x)
            count_after_relu = get_cache_stats()["total_cached_kernels"]

            self.assertGreater(
                count_after_relu,
                count_after_softmax,
                "Expected a new cache entry for relu vs softmax",
            )


class TestCompileConfigAffectsCacheKey(unittest.TestCase):
    """A config knob that changes compiled output must change the cache key.

    Regression guard for two distinct ways this silently broke: memoising the
    config snapshot (so ``config.patch()`` never reached the key), and simply
    omitting a knob that ``generate_bundle`` branches on. Both produce a stale
    cache HIT that returns a kernel compiled under a different configuration --
    wrong numerics, no error.
    """

    # Knobs that generate_bundle / the SDSC emitter branch on, and so must be
    # part of the cache key. Each maps to a value differing from the default.
    EMISSION_KNOBS = {
        "frontend_pool_allocation": True,
        "sencores": 4,
        "hbm_pool_planning": False,
        "lx_planning": False,
        "enable_reduction_tiling": False,
        "core_id_k_fast_emission": False,
        "ignore_span_overflow_hints": False,
    }

    def test_each_emission_knob_changes_the_key(self):
        from torch_spyre._inductor import config as spyre_config
        from torch_spyre.execution.kernel_cache import _get_compile_config

        baseline = _get_compile_config()
        for knob, value in self.EMISSION_KNOBS.items():
            with self.subTest(knob=knob):
                self.assertNotEqual(
                    getattr(spyre_config, knob),
                    value,
                    f"{knob} test value equals the default; pick a different one",
                )
                with spyre_config.patch({knob: value}):
                    patched = _get_compile_config()
                self.assertNotEqual(
                    baseline,
                    patched,
                    f"config.{knob} does not affect the cache key: a kernel "
                    f"compiled with {knob}={value} would be served from a cache "
                    f"entry compiled with the default value",
                )

    def test_config_snapshot_is_not_memoised(self):
        """_get_compile_config must re-read config on every call, not cache it."""
        from torch_spyre._inductor import config as spyre_config
        from torch_spyre.execution.kernel_cache import _get_compile_config

        # Prime any accidental memoisation with the default config first.
        _get_compile_config()
        with spyre_config.patch({"sencores": 4}):
            under_patch = _get_compile_config()
        after = _get_compile_config()

        self.assertIn('"sencores": 4', under_patch)
        self.assertNotIn('"sencores": 4', after)


class TestSameOpReusesCacheEntry(unittest.TestCase):
    def test_same_op_compiled_twice_uses_same_cache_entry(self):
        """Compiling the same op twice must not create duplicate cache entries."""
        with fresh_cache():
            torch._dynamo.reset()
            x = _make_input()

            torch.compile(lambda a: torch.softmax(a, dim=-1))(x)
            count_first = get_cache_stats()["total_cached_kernels"]

            torch._dynamo.reset()
            torch.compile(lambda a: torch.softmax(a, dim=-1))(x)
            count_second = get_cache_stats()["total_cached_kernels"]

            self.assertEqual(
                count_first,
                count_second,
                "Expected no new cache entries when compiling the same op twice",
            )


class TestClearCache(unittest.TestCase):
    def test_clear_cache_removes_all_entries(self):
        """clear_cache() must leave total_cached_kernels == 0."""
        from torch_spyre.execution.kernel_cache import clear_cache

        with fresh_cache():
            torch._dynamo.reset()
            torch.compile(_simple_fn)(_make_input())

            self.assertGreater(get_cache_stats()["total_cached_kernels"], 0)

            clear_cache()
            stats = get_cache_stats()
            self.assertEqual(stats["total_cached_kernels"], 0)
            self.assertAlmostEqual(stats["cache_size_mb"], 0.0, places=1)


class TestAtomicCommit(unittest.TestCase):
    def test_concurrent_commit_same_key_does_not_corrupt(self):
        """Four threads compiling the same key concurrently must leave exactly one valid entry."""
        import threading

        fake_key = "c" + "b" * 63

        with fresh_cache():
            errors = []

            def do_compile():
                try:
                    # Each thread gets its own allocated tmp dir with the same key.
                    tmp_dir = allocate_compile_dir(fake_key)
                    # Populate it with the minimal required artifacts.
                    os.makedirs(os.path.join(tmp_dir, "spyreCodeDir"), exist_ok=True)
                    for name in ["bundle.mlir", "sdsc_0.json"]:
                        with open(os.path.join(tmp_dir, name), "w") as f:
                            f.write("content")
                    for name in ["init_binary.bin", "spyrecode.json"]:
                        with open(
                            os.path.join(tmp_dir, "spyreCodeDir", name), "wb"
                        ) as f:
                            f.write(b"content")
                    commit_compile_dir(tmp_dir, fake_key)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=do_compile) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            self.assertEqual(errors, [], f"Concurrent commit raised errors: {errors}")

            result = get_cached_kernel_dir(fake_key)
            self.assertIsNotNone(
                result, "Expected a valid cache entry after concurrent commit"
            )


class TestNoDiskIOOnCacheHit(unittest.TestCase):
    def test_generate_bundle_skipped_on_cache_hit(self):
        """On a kernel cache hit generate_bundle must not be called."""
        from unittest.mock import patch

        with fresh_cache():
            torch._dynamo.reset()
            # Populate the cache on first run.
            torch.compile(_simple_fn)(_make_input())

            # Second compile (after dynamo reset) must hit the cache.
            torch._dynamo.reset()
            with patch(
                "torch_spyre.execution.async_compile.generate_bundle"
            ) as mock_gen:
                torch.compile(_simple_fn)(_make_input())

            mock_gen.assert_not_called()


class TestAOTAutogradCacheWarning(unittest.TestCase):
    def test_warning_emitted_when_enable_autograd_cache_true(self):
        """A UserWarning must be raised when enable_autograd_cache=True on a Spyre graph."""
        import torch._functorch.config as _ftn_config

        with fresh_cache():
            torch._dynamo.reset()
            original = getattr(_ftn_config, "enable_autograd_cache", False)
            _ftn_config.enable_autograd_cache = True
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    torch.compile(_simple_fn)(_make_input())

                spyre_warnings = [
                    str(w.message)
                    for w in caught
                    if "AOTAutogradCache" in str(w.message)
                ]
                self.assertTrue(
                    len(spyre_warnings) > 0,
                    "Expected a warning about AOTAutogradCache being unsupported for Spyre",
                )
            finally:
                _ftn_config.enable_autograd_cache = original

    def test_no_warning_when_enable_autograd_cache_false(self):
        """No warning must be emitted when enable_autograd_cache=False (the default)."""
        import torch._functorch.config as _ftn_config

        with fresh_cache():
            torch._dynamo.reset()
            original = getattr(_ftn_config, "enable_autograd_cache", False)
            _ftn_config.enable_autograd_cache = False
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    torch.compile(_simple_fn)(_make_input())

                spyre_warnings = [
                    str(w.message)
                    for w in caught
                    if "AOTAutogradCache" in str(w.message)
                ]
                self.assertEqual(
                    spyre_warnings,
                    [],
                    "Expected no AOTAutogradCache warnings when it is disabled",
                )
            finally:
                _ftn_config.enable_autograd_cache = original


if __name__ == "__main__":
    unittest.main(verbosity=2)
