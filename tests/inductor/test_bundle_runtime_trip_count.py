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

"""Tests for launch-time loop trip counts (config.bundle_runtime_trip_count).

compile_op_spec is mocked in the bundle tests, so no Spyre hardware is needed.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

from torch_spyre._inductor import config
from torch_spyre._inductor.codegen.bundle import (
    bind_loop_counts,
    generate_bundle,
    runtime_trip_count_max,
)
from torch_spyre._inductor.codegen.compute_ops import SymbolKind
from torch_spyre._inductor.op_spec import LoopSpec, OpSpec

from tests.inductor.test_hash_coverage import _hash, _make_loop_spec


def _stub_op() -> OpSpec:
    return OpSpec(
        op="gelu", is_reduction=False, iteration_space={}, args=[], op_info={}
    )


def _entry(sdsc_idx: int = 0):
    json_ = {f"{sdsc_idx}_fused_test": {"numCoresUsed_": 1, "dscs_": [{"op": {}}]}}
    return (json_, [], [], [])


class TestRuntimeTripCountMax(unittest.TestCase):
    def test_off_by_default(self):
        with config.patch({"bundle_runtime_trip_count": False}):
            self.assertIsNone(runtime_trip_count_max(4))

    def test_in_range(self):
        with config.patch(
            {"bundle_runtime_trip_count": True, "bundle_trip_count_max": 64}
        ):
            self.assertEqual(runtime_trip_count_max(1), 64)
            self.assertEqual(runtime_trip_count_max(64), 64)
            self.assertIsNone(runtime_trip_count_max(65))


class TestBundleEmission(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmp.cleanup()

    def _bundle(self, specs, n_ops):
        side_effects = [e for _ in range(n_ops) for e in (_entry(), _entry())]
        with patch(
            "torch_spyre._inductor.codegen.bundle.compile_op_spec",
            side_effect=side_effects,
        ):
            kinds = generate_bundle("test", self._tmp.name, specs)
        with open(os.path.join(self._tmp.name, "bundle.mlir")) as f:
            return f.read(), kinds

    def test_flag_off_keeps_constant(self):
        with config.patch({"bundle_runtime_trip_count": False}):
            text, kinds = self._bundle([LoopSpec(count=4, body=[_stub_op()])], 1)
        self.assertIn("%loop_bound_0 = arith.constant 4 : index", text)
        self.assertNotIn("loop_count", text)
        self.assertFalse(any(k.is_loop_count for k in kinds))

    def test_flag_on_emits_input_arg(self):
        with config.patch(
            {"bundle_runtime_trip_count": True, "bundle_trip_count_max": 64}
        ):
            text, kinds = self._bundle([LoopSpec(count=4, body=[_stub_op()])], 1)
        arg_type = "!sdscbundle.input_arg<index, granularity=1, max_value=64>"
        self.assertIn(f"%loop_count_0: {arg_type}", text)
        self.assertIn(
            f"%loop_bound_0 = sdscbundle.input_arg_extract value from"
            f" %loop_count_0 : {arg_type} -> index",
            text,
        )
        self.assertNotIn("arith.constant 4 : index", text)
        self.assertEqual(kinds, [SymbolKind.loop_count(0, 64, 4)])

    def test_bundle_text_independent_of_count(self):
        with config.patch(
            {"bundle_runtime_trip_count": True, "bundle_trip_count_max": 64}
        ):
            t4, _ = self._bundle([LoopSpec(count=4, body=[_stub_op()])], 1)
            t32, k32 = self._bundle([LoopSpec(count=32, body=[_stub_op()])], 1)
        self.assertEqual(t4, t32)
        self.assertEqual(k32[0].trip_count, 32)

    def test_count_above_max_stays_constant(self):
        with config.patch(
            {"bundle_runtime_trip_count": True, "bundle_trip_count_max": 8}
        ):
            text, kinds = self._bundle([LoopSpec(count=16, body=[_stub_op()])], 1)
        self.assertIn("%loop_bound_0 = arith.constant 16 : index", text)
        self.assertFalse(any(k.is_loop_count for k in kinds))

    def test_nested_loops_each_get_a_param(self):
        inner = LoopSpec(count=3, body=[_stub_op()])
        with config.patch(
            {"bundle_runtime_trip_count": True, "bundle_trip_count_max": 64}
        ):
            text, kinds = self._bundle([LoopSpec(count=5, body=[inner])], 1)
        self.assertIn("%loop_count_0:", text)
        self.assertIn("%loop_count_1:", text)
        self.assertEqual(
            kinds,
            [SymbolKind.loop_count(0, 64, 5), SymbolKind.loop_count(1, 64, 3)],
        )


class TestCacheKey(unittest.TestCase):
    def test_counts_share_a_key_when_on(self):
        with config.patch(
            {"bundle_runtime_trip_count": True, "bundle_trip_count_max": 64}
        ):
            self.assertEqual(
                _hash([_make_loop_spec(count=4)]), _hash([_make_loop_spec(count=32)])
            )

    def test_counts_differ_when_off(self):
        with config.patch({"bundle_runtime_trip_count": False}):
            self.assertNotEqual(
                _hash([_make_loop_spec(count=4)]), _hash([_make_loop_spec(count=32)])
            )

    def test_max_value_is_in_the_key(self):
        loop = _make_loop_spec(count=4)
        with config.patch(
            {"bundle_runtime_trip_count": True, "bundle_trip_count_max": 64}
        ):
            h64 = _hash([loop])
        with config.patch(
            {"bundle_runtime_trip_count": True, "bundle_trip_count_max": 128}
        ):
            h128 = _hash([loop])
        self.assertNotEqual(h64, h128)

    def test_on_and_off_do_not_collide(self):
        loop = _make_loop_spec(count=4)
        with config.patch({"bundle_runtime_trip_count": False}):
            off = _hash([loop])
        with config.patch(
            {"bundle_runtime_trip_count": True, "bundle_trip_count_max": 64}
        ):
            on = _hash([loop])
        self.assertNotEqual(off, on)


class TestBindLoopCounts(unittest.TestCase):
    def test_binds_the_launching_kernel_counts(self):
        cached = [SymbolKind.kernel(0), SymbolKind.loop_count(0, 64, 4)]
        specs = [LoopSpec(count=32, body=[_stub_op()])]
        bound = bind_loop_counts(cached, specs)
        self.assertEqual(bound[0], SymbolKind.kernel(0))
        self.assertEqual(bound[1].trip_count, 32)

    def test_out_of_range_raises(self):
        cached = [SymbolKind.loop_count(0, 8, 4)]
        with self.assertRaises(ValueError):
            bind_loop_counts(cached, [LoopSpec(count=9, body=[_stub_op()])])

    def test_no_loop_counts_is_identity(self):
        cached = [SymbolKind.kernel(0)]
        self.assertIs(bind_loop_counts(cached, []), cached)


class TestRunnerPayload(unittest.TestCase):
    def test_loop_count_becomes_kdimension_value(self):
        from torch_spyre._C import SymbolicArgKind
        from torch_spyre.execution.kernel_runner import SpyreSDSCKernelRunner

        runner = SpyreSDSCKernelRunner(
            "k",
            "/nonexistent",
            symbol_kinds=[
                SymbolKind.pool(),
                SymbolKind.kernel(0),
                SymbolKind.loop_count(0, 64, 12),
            ],
        )
        payload = runner._symbolic_args
        self.assertEqual(payload[0].kind, SymbolicArgKind.kAddress)
        self.assertEqual(payload[0].tensor_id, 0)
        self.assertEqual(payload[1].tensor_id, 1)
        self.assertEqual(payload[2].kind, SymbolicArgKind.kDimension)
        self.assertEqual(payload[2].value, 12)

    def test_resolve_symbolic_args_passes_the_value(self):
        from torch_spyre._C import (
            SymbolicArg,
            SymbolicArgKind,
            _resolve_symbolic_args,
        )

        arg = SymbolicArg(kind=SymbolicArgKind.kDimension, tensor_id=-1, value=17)
        self.assertEqual(_resolve_symbolic_args([], [arg]), [17])

    def test_resolve_symbolic_args_rejects_negative(self):
        from torch_spyre._C import (
            SymbolicArg,
            SymbolicArgKind,
            _resolve_symbolic_args,
        )

        arg = SymbolicArg(kind=SymbolicArgKind.kDimension, tensor_id=-1, value=-1)
        with self.assertRaises(RuntimeError):
            _resolve_symbolic_args([], [arg])


if __name__ == "__main__":
    unittest.main()
