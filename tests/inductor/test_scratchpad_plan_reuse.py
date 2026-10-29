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

"""Tests for reusing a joint scratchpad plan across identical solver inputs."""

import unittest
from types import SimpleNamespace
from unittest import TestCase, mock

from torch_spyre._inductor import config
from torch_spyre._inductor.scratchpad import plan_reuse
from torch_spyre._inductor.scratchpad.allocator import CoOptimizingAllocator
from torch_spyre._inductor.scratchpad.plan_solver import (
    CoreDivision,
    CoreDivisionBuffer,
)

try:
    from torch_spyre._inductor.scratchpad.ilp_solver_ortools import (
        CpSatLayoutSolver,
    )

    _HAS_ORTOOLS = True
except ImportError:
    CpSatLayoutSolver = None  # type: ignore[assignment,misc]
    _HAS_ORTOOLS = False


def _buffers(size=256):
    """A producer and its consumer that can both reside under a 2-way split."""
    divisions = [CoreDivision(splits={"x": 1}), CoreDivision(splits={"x": 2})]
    producer = CoreDivisionBuffer("p", size, [0, 1], core_divisions=list(divisions))
    consumer = CoreDivisionBuffer(
        "c",
        size,
        [1, 2],
        core_divisions=list(divisions),
        parents=["p"],
        cd_parent_matches={"p": [(0, 0), (1, 1)]},
    )
    sink = CoreDivisionBuffer(
        "s", 1, [3], core_divisions=[CoreDivision()], parents=["c"]
    )
    return [producer, consumer, sink]


def _solved(buffers):
    """Write a fixed plan onto ``buffers``, as a solve would."""
    for i, b in enumerate(buffers):
        b.address = None if b.name == "s" else 128 * i
        b.chosen_division = 1 if b.name != "s" else 0
        b.spill_reason = "spilled by solver" if b.name == "s" else None
    return buffers


@unittest.skipUnless(_HAS_ORTOOLS, "plan reuse keys CP-SAT solves only")
class TestScratchpadPlanReuse(TestCase):
    def setUp(self):
        plan_reuse.clear()
        self.addCleanup(plan_reuse.clear)
        patcher = config.patch(
            {"scratchpad_plan_reuse": True, "dump_cost_expr_file": ""}
        )
        patcher.__enter__()
        self.addCleanup(patcher.__exit__, None, None, None)
        self.allocator = CoOptimizingAllocator(CpSatLayoutSolver, size=4096)

    def _solve(self, buffers, solve, graph=None):
        graph = graph or SimpleNamespace(operations=[])
        solver = CpSatLayoutSolver(buffers, 4096)
        with mock.patch.object(
            CoOptimizingAllocator, "_solve_and_price", side_effect=solve
        ) as priced:
            result = self.allocator._solve(solver, graph)
        return result, priced.call_count

    def test_second_identical_solve_replays_the_plan(self):
        first, calls = self._solve(_buffers(), lambda s, g: _solved(s.buffers))
        self.assertEqual(calls, 1)
        second, calls = self._solve(_buffers(), lambda s, g: self.fail("solved"))
        self.assertEqual(calls, 0)
        for a, b in zip(first, second):
            self.assertEqual(
                (a.name, a.address, a.chosen_division, a.spill_reason),
                (b.name, b.address, b.chosen_division, b.spill_reason),
            )

    def test_different_input_solves_again(self):
        self._solve(_buffers(), lambda s, g: _solved(s.buffers))
        _, calls = self._solve(_buffers(size=512), lambda s, g: _solved(s.buffers))
        self.assertEqual(calls, 1)

    def test_different_ops_solve_again(self):
        def graph(op_type):
            op = mock.Mock(spec=["get_name", "data"])
            op.get_name.return_value = "buf0"
            op.data = SimpleNamespace(origin_node=SimpleNamespace(name=op_type))
            return SimpleNamespace(operations=[op])

        self._solve(_buffers(), lambda s, g: _solved(s.buffers), graph("add"))
        _, calls = self._solve(
            _buffers(), lambda s, g: _solved(s.buffers), graph("mul")
        )
        self.assertEqual(calls, 1)

    def test_config_change_solves_again(self):
        self._solve(_buffers(), lambda s, g: _solved(s.buffers))
        with config.patch({"sencores": config.sencores // 2}):
            _, calls = self._solve(_buffers(), lambda s, g: _solved(s.buffers))
        self.assertEqual(calls, 1)

    def test_disabled_while_dumping_the_cost_expression(self):
        self._solve(_buffers(), lambda s, g: _solved(s.buffers))
        with config.patch({"dump_cost_expr_file": "/dev/null"}):
            _, calls = self._solve(_buffers(), lambda s, g: _solved(s.buffers))
        self.assertEqual(calls, 1)

    def test_replayed_plan_matches_a_real_solve(self):
        # The real solver: a replay must give the plan a fresh solve gives.
        solved = CpSatLayoutSolver(_buffers(), 4096).plan_layout_and_core_divisions()
        plan_reuse.store("k", solved)
        replayed = plan_reuse.lookup("k", _buffers())
        self.assertIsNotNone(replayed)
        for a, b in zip(solved, replayed):
            self.assertEqual(
                (a.address, a.chosen_division, a.spill_reason),
                (b.address, b.chosen_division, b.spill_reason),
            )


if __name__ == "__main__":
    unittest.main()
