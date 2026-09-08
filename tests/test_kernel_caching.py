#!/usr/bin/env python3
# Copyright 2026 The Torch-Spyre Authors.
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
import torch
import torch_spyre  # noqa: F401 — side-effects: registers Spyre backend

import torch_spyre._inductor.config as spyre_config
from torch._inductor.utils import fresh_cache

from torch_spyre.execution.kernel_cache import (
    _FAILED_DIR_NAME,
    _move_to_failed_dir,
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
        with fresh_cache(), spyre_config.patch({"spyre_kernel_cache": True}):
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
        with fresh_cache(), spyre_config.patch({"spyre_kernel_cache": True}):
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
        with fresh_cache(), spyre_config.patch({"spyre_kernel_cache": False}):
            torch._dynamo.reset()
            torch.compile(_simple_fn)(_make_input())
            self.assertEqual(
                get_cache_stats()["total_cached_kernels"],
                0,
                "Expected empty cache when spyre_kernel_cache=False",
            )


class TestForceDisableCaches(unittest.TestCase):
    def test_force_disable_caches_leaves_cache_empty(self):
        """torch._inductor.config.force_disable_caches must bypass the Spyre cache.

        Deliberately does not patch spyre_kernel_cache: it relies on the default
        being on, so force_disable_caches is the only thing suppressing the cache
        here. If the default were ever flipped back off this assertion would pass
        for the wrong reason, so pair any such change with an explicit patch.
        """
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
        with fresh_cache(), spyre_config.patch({"spyre_kernel_cache": True}):
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


class TestSameOpReusesCacheEntry(unittest.TestCase):
    def test_same_op_compiled_twice_uses_same_cache_entry(self):
        """Compiling the same op twice must not create duplicate cache entries.

        Deliberately does not patch spyre_kernel_cache: it relies on the default
        being on, so a first entry is actually written and the second compile has
        something to collide with. With caching off both counts would be 0 and the
        assertion would hold vacuously -- so pair a default flip with an explicit
        patch here.
        """
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

        with fresh_cache(), spyre_config.patch({"spyre_kernel_cache": True}):
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

        with fresh_cache(), spyre_config.patch({"spyre_kernel_cache": True}):
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


def _populate_compile_dir(compile_dir: str) -> None:
    """Write the minimal artifact set that makes a dir a valid cache entry."""
    os.makedirs(os.path.join(compile_dir, "spyreCodeDir"), exist_ok=True)
    for name in ["bundle.mlir", "sdsc_0.json"]:
        with open(os.path.join(compile_dir, name), "w") as f:
            f.write("content")
    for name in ["init_binary.bin", "spyrecode.json"]:
        with open(os.path.join(compile_dir, "spyreCodeDir", name), "wb") as f:
            f.write(b"content")


class TestMoveToFailedDir(unittest.TestCase):
    """A failed compile must be retained under failed/ for manual debugging."""

    def test_failed_dir_is_moved_and_contents_preserved(self):
        """The dir must leave the cache root and keep its artifacts intact."""
        with fresh_cache():
            cache_root = get_cache_root_dir()
            tmp_dir = allocate_compile_dir("d" + "a" * 63)
            _populate_compile_dir(tmp_dir)

            _move_to_failed_dir(tmp_dir)

            self.assertFalse(
                os.path.exists(tmp_dir), "Failed compile dir must not stay in place"
            )
            dest = os.path.join(cache_root, _FAILED_DIR_NAME, os.path.basename(tmp_dir))
            self.assertTrue(os.path.isdir(dest), f"Expected failed dir at {dest}")
            self.assertTrue(
                os.path.isfile(os.path.join(dest, "bundle.mlir")),
                "Artifacts must survive the move so dxp_standalone -d can rerun",
            )

    def test_failed_dirs_do_not_collide(self):
        """Two failures for the same key must not overwrite each other."""
        with fresh_cache():
            cache_root = get_cache_root_dir()
            key = "d" + "b" * 63

            for _ in range(2):
                tmp_dir = allocate_compile_dir(key)
                _populate_compile_dir(tmp_dir)
                _move_to_failed_dir(tmp_dir)

            failed_root = os.path.join(cache_root, _FAILED_DIR_NAME)
            self.assertEqual(
                len(os.listdir(failed_root)),
                2,
                "Each failure must be retained under its own unique name",
            )

    def test_move_failure_is_not_fatal(self):
        """A rename failure must be logged, not raised — the compile error wins.

        _move_to_failed_dir runs inside an ``except`` block that re-raises the
        original compilation failure; masking it with an OSError from the move
        would lose the useful diagnostic.
        """
        from unittest.mock import patch

        with fresh_cache():
            tmp_dir = allocate_compile_dir("d" + "c" * 63)
            _populate_compile_dir(tmp_dir)

            with patch("os.rename", side_effect=OSError("cross-device link")):
                _move_to_failed_dir(tmp_dir)  # must not raise

            self.assertTrue(
                os.path.isdir(tmp_dir),
                "A failed move must leave the original dir for debugging",
            )


class TestCommitCompileDir(unittest.TestCase):
    """commit_compile_dir must distinguish a lost race from a real I/O error."""

    def test_existing_destination_discards_temp_and_returns_winner(self):
        """Losing the race must reuse the winner's entry, not fail."""
        with fresh_cache():
            key = "e" + "a" * 63

            winner = allocate_compile_dir(key)
            _populate_compile_dir(winner)
            cached_dir = commit_compile_dir(winner, key)

            loser = allocate_compile_dir(key)
            _populate_compile_dir(loser)
            result = commit_compile_dir(loser, key)

            self.assertEqual(result, cached_dir, "Must return the committed entry")
            self.assertFalse(
                os.path.exists(loser), "The losing temp dir must be discarded"
            )
            self.assertIsNotNone(get_cached_kernel_dir(key))

    def test_rename_error_without_destination_is_raised(self):
        """A non-race OSError must propagate instead of being called a race.

        Returning cached_dir here would hand the caller a path that does not
        exist, turning a full disk into a confusing missing-artifact error much
        later in the compile.
        """
        from unittest.mock import patch

        with fresh_cache():
            key = "e" + "b" * 63
            tmp_dir = allocate_compile_dir(key)
            _populate_compile_dir(tmp_dir)

            with patch("os.rename", side_effect=OSError(28, "No space left on device")):
                with self.assertRaises(OSError):
                    commit_compile_dir(tmp_dir, key)


class TestCacheEnabledByDefault(unittest.TestCase):
    def test_kernel_cache_is_on_by_default(self):
        """The cache must be enabled without any env var being set.

        Several tests in this file rely on the default being on rather than
        patching it, so this pins the default itself.
        """
        self.assertTrue(
            spyre_config.spyre_kernel_cache,
            "spyre_kernel_cache must default to True (SPYRE_KERNEL_CACHE=0 opts out)",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
