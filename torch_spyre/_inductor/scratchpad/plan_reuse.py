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

"""Reuse a joint scratchpad plan across solves that have the same solver input.

The co-optimizing allocator hands its solver a list of buffers: sizes, uses,
candidate core divisions and the edges between them. A ``for_each_tile`` loop's
trip count does not appear there; it only scales the cost objective. So the
buckets of one attention kernel (one compile per ``num_blocks``) build the same
buffer list, and a plan solved for one bucket is a valid plan for every other.

This module keys a solved plan on that buffer list plus the solver and config it
was solved under, and replays it on a later solve with the same key. A replay
skips building the cost expression, lowering it to CP-SAT and the solve itself.
The replayed plan is always feasible, since the constraints are the same; it is
the optimum of the first solve's objective, which can differ from the optimum of
the current one.

The cache lives for the process. Enable it with ``SPYRE_SCRATCHPAD_PLAN_REUSE=1``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import threading
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any, Optional

import torch

from torch_spyre._inductor import config
from torch_spyre._inductor.logging_utils import get_inductor_logger
from torch_spyre._inductor.scratchpad.plan_solver import MemoryPlanSolver

logger = get_inductor_logger("scratchpad.plan_reuse")

# Enough for every bucket of every attention kernel of one model.
_MAX_ENTRIES = 256


@dataclasses.dataclass(frozen=True)
class _BufferPlan:
    """What a solve wrote back onto one buffer."""

    name: str
    address: Optional[int]
    chosen_division: Optional[int]
    spill_reason: Optional[str]
    # parent -> (index into ``cd_parent_relayouts[parent]``, copy address)
    relayouts: tuple[tuple[str, int, int], ...]


_plans: OrderedDict[str, tuple[_BufferPlan, ...]] = OrderedDict()
_lock = threading.Lock()


def enabled() -> bool:
    # A replay builds no cost expression, so it has no record to dump.
    return config.scratchpad_plan_reuse and not config.dump_cost_expr_file


def plan_key(solver: MemoryPlanSolver, graph: Any) -> Optional[str]:
    """Key for ``solver``'s input, or None when it cannot be keyed.

    Only CP-SAT is keyed: the key covers the buffer fields it reads, and other
    engines read more (the annealer walks ``division_space``). A repr carrying
    an object address is not stable, so it gets no key either. The op kinds of
    ``graph`` are in the key too, so two graphs that only share a buffer layout
    do not share a plan.
    """
    from torch_spyre._inductor.scratchpad.ilp_solver_ortools import CpSatLayoutSolver

    if type(solver) is not CpSatLayoutSolver:
        return None
    buffers = solver.buffers
    if any(b.paired_with or b.lx_relayout_plans for b in buffers):
        return None
    text = "\n".join(_buffer_repr(b) for b in buffers)
    if " at 0x" in text:
        return None
    parts = [
        f"limit={solver.limit}",
        f"alignment={solver.alignment}",
        f"time_limit={solver._time_limit_seconds}",
        f"deterministic={torch.are_deterministic_algorithms_enabled()}",
        repr(sorted(config.save_config_portable().items())),
        _op_kinds(graph),
        text,
    ]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def _op_kinds(graph: Any) -> str:
    from torch_spyre._inductor.dump_common import origin_op_name

    return ",".join(
        f"{op.get_name()}:{type(op).__name__}:{origin_op_name(op)}"
        for op in graph.operations
    )


def _buffer_repr(buffer: Any) -> str:
    """``repr(buffer)`` without the fields CP-SAT does not read.

    Those fields hold IR nodes, whose repr is slow and carries graph-local names.
    """
    fields = (
        f"{f.name}={getattr(buffer, f.name)!r}"
        for f in dataclasses.fields(buffer)
        if f.repr and f.name not in _UNREAD_FIELDS
    )
    return f"{type(buffer).__name__}({', '.join(fields)})"


# CoreDivisionBuffer fields the allocator fills for other engines.
_UNREAD_FIELDS = frozenset(
    {"residency_edges", "division_space", "op_position", "tile_reads"}
)


def lookup(key: str, buffers: Sequence[Any]) -> Optional[list[Any]]:
    """Write the plan stored under ``key`` onto ``buffers`` and return them.

    Returns None, leaving ``buffers`` untouched, when there is no plan or it
    does not fit them.
    """
    with _lock:
        plan = _plans.get(key)
        if plan is not None:
            _plans.move_to_end(key)
    if plan is None or [p.name for p in plan] != [b.name for b in buffers]:
        return None
    relayouts = []
    for p, b in zip(plan, buffers):
        chosen = {}
        for parent, index, address in p.relayouts:
            candidates = getattr(b, "cd_parent_relayouts", {}).get(parent, [])
            if index >= len(candidates):
                return None
            chosen[parent] = (candidates[index], address)
        relayouts.append(chosen)

    from torch_spyre._inductor.scratchpad.lx_relayout import ChosenRelayout

    for p, b, chosen in zip(plan, buffers, relayouts):
        b.address = p.address
        b.spill_reason = p.spill_reason
        if hasattr(b, "chosen_division"):
            b.chosen_division = p.chosen_division
            b.chosen_relayouts = {
                parent: ChosenRelayout(candidate, address)
                for parent, (candidate, address) in chosen.items()
            }
    logger.debug("Reused scratchpad plan %s for %d buffers", key[:16], len(buffers))
    return list(buffers)


def store(key: str, buffers: Sequence[Any]) -> None:
    """Record the plan ``buffers`` carry after a solve under ``key``."""
    plan = []
    for b in buffers:
        relayouts = []
        for parent, chosen in getattr(b, "chosen_relayouts", {}).items():
            candidates = b.cd_parent_relayouts.get(parent, [])
            try:
                index = candidates.index(chosen.candidate)
            except ValueError:
                return
            relayouts.append((parent, index, chosen.destination_address))
        plan.append(
            _BufferPlan(
                name=b.name,
                address=b.address,
                chosen_division=getattr(b, "chosen_division", None),
                spill_reason=b.spill_reason,
                relayouts=tuple(relayouts),
            )
        )
    with _lock:
        _plans[key] = tuple(plan)
        _plans.move_to_end(key)
        while len(_plans) > _MAX_ENTRIES:
            _plans.popitem(last=False)


def clear() -> None:
    with _lock:
        _plans.clear()
