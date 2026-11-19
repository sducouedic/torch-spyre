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

"""Run the Spyre inner compiles of many graphs in forked child processes.

Dynamo, AOTAutograd and the joint-graph passes stay in this process, because they
may run small ops on the device. The inner compile (lowering, LX planning, code
generation, dbo-opt) is host work only, and it writes the FX graph cache and the
Spyre kernel cache.

Inside ``fork_inner_compiles(n)``, each inner compile runs in a forked child, at
most ``n`` at once, and this process gets a stand-in graph that raises
``CompileOnlyAbort`` when run. Leaving the block waits for all children. The
caller then resets those frames (``torch._dynamo.reset_code``) and runs the graphs
again: each compile now finds the children's entries in the caches. A child that
fails only costs time, because the second run compiles whatever is missing.
``SecondPassCheck`` around that second run logs a warning with the numbers when
any compile missed the FX graph cache, so a lost speedup does not go unnoticed.

What a child may use: the compile itself (Python, the Inductor and Spyre lowering
code, CP-SAT, and dbo-opt as a subprocess). What it may not use: the device. Every
Python entry point of the Spyre runtime (start, allocation, copies, fills, streams,
synchronize, kernel preparation) raises in a child, and so does the eager copy op.
The child runs on one thread with ``compile_threads = 1``, a one-worker CP-SAT
(``config.num_cpus = 1``) and a single-threaded dbo-opt, so it starts no worker pool
of its own (#4427's pool stays in the parent), and it never takes a lock that another
thread of the parent could hold at fork time: such a lock stays held in the child, so
a compile that needed it would hang and be killed by the timeout.

Limits: Linux only (``os.fork``). Each child is a copy-on-write copy of the caller.
For a vLLM engine with Granite 3.3 8B and 8 workers, the peak resident memory of the
process tree grew from 16.9 GB to 25.1 GB (about 1 GB per child).
"""

import contextlib
import os
import signal
import time
from typing import Any, Callable, Optional

from .logging_utils import get_inductor_logger

logger = get_inductor_logger("parallel_precompile")


class CompileOnlyAbort(Exception):
    """Raised by a stand-in graph while its compile runs in a child."""


class _StandIn:
    """The graph this process gets back for a compile that runs in a child."""

    _boxed_call = True

    def __init__(self, pool: "_Pool") -> None:
        self._pool = pool

    def __call__(self, args: list) -> Any:
        if self._pool.open:
            raise CompileOnlyAbort()
        raise RuntimeError(
            "parallel precompile: a stand-in graph ran after the precompile; "
            "reset its frame with torch._dynamo.reset_code"
        )


# The ``torch_spyre._C`` functions that reach the runtime or the device.
_DEVICE_ENTRY_POINTS = (
    "start_runtime",
    "free_runtime",
    "set_device",
    "synchronize",
    "spyre_empty_with_layout",
    "spyre_empty_reserved",
    "empty_with_layout",
    "copy_tensor",
    "copy_kv_page_raw",
    "fill_tensor",
    "get_composite_address",
    "get_stream_from_pool",
    "set_current_stream",
    "prepare_kernel",
)


def _no_device(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError("parallel precompile: a forked compile tried to use the device")


def _child_setup() -> None:
    import torch

    import torch_spyre._C as spyre_c
    import torch_spyre.ops.eager as spyre_eager
    import torch_spyre.execution.async_compile as spyre_async_compile
    from torch_spyre._inductor import config as spyre_config

    os.setsid()  # own process group, so a kill also stops dbo-opt
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    torch.set_num_threads(1)  # the parent's OpenMP pool does not survive fork()
    torch._inductor.config.compile_threads = 1  # the parent's worker pool is not ours
    # CP-SAT sizes its worker pool from the CPU quota, which the parent and every
    # other child already use: n children would start n times that many threads.
    spyre_config.num_cpus = 1
    spyre_async_compile.single_threaded_backend = True  # dbo-opt without its thread pool
    for name in _DEVICE_ENTRY_POINTS:  # the device stays with the caller
        if hasattr(spyre_c, name):
            setattr(spyre_c, name, _no_device)
    spyre_eager.copy_tensor = _no_device


class _Pool:
    """At most ``workers`` children at once. A child is killed after ``timeout_s``."""

    def __init__(self, workers: int, timeout_s: float) -> None:
        self.workers = workers
        self.timeout_s = timeout_s
        self.open = True
        self.running: dict[int, float] = {}  # pid -> start time
        self.ok = 0
        self.failed = 0
        self.inline = 0  # compiles that ``should_fork`` kept in this process
        self.cached = 0  # FX graph cache hits, loaded in this process

    @staticmethod
    def _kill(pid: int) -> None:
        for kill in (os.killpg, os.kill):
            with contextlib.suppress(ProcessLookupError, PermissionError):
                kill(pid, signal.SIGKILL)

    def _reap(self) -> bool:
        """Collect finished children and kill late ones. True if one finished."""
        finished = False
        for pid, started in list(self.running.items()):
            if time.time() - started > self.timeout_s:
                self._kill(pid)
            try:
                done, status = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                done, status = pid, -1
            if done:
                del self.running[pid]
                finished = True
                if status == 0:
                    self.ok += 1
                else:
                    self.failed += 1
        return finished

    def submit(self, compile_fn: Callable[[], Any]) -> None:
        while len(self.running) >= self.workers:
            if not self._reap():
                time.sleep(0.02)
        pid = os.fork()
        if pid == 0:  # child: compile, then exit without returning to the caller
            code = 1
            try:
                _child_setup()
                compile_fn()
                code = 0
            except BaseException:  # noqa: BLE001 - the caller compiles it again
                logger.warning("parallel precompile: a compile failed", exc_info=True)
            finally:
                os._exit(code)
        self.running[pid] = time.time()

    def wait(self) -> None:
        while self.running:
            if not self._reap():
                time.sleep(0.05)

    def kill_all(self) -> None:
        for pid in list(self.running):
            self._kill(pid)
        self.wait()


class _CacheMiss(Exception):
    """Raised in place of the codegen of a compile that missed the FX graph cache."""


def _load_if_cached(original: Callable[..., Any], args: tuple, kwargs: dict) -> Any:
    """The compiled graph when the FX graph cache holds it, else None.

    The inner compile reads the FX graph cache first and generates code only on a
    miss, so running it here with the codegen replaced by a raise loads a hit in this
    process and costs only the cache key on a miss.
    """
    import torch._inductor.compile_fx as compile_fx

    codegen = compile_fx.fx_codegen_and_compile

    def miss(*a: Any, **k: Any) -> Any:
        raise _CacheMiss()

    compile_fx.fx_codegen_and_compile = miss
    try:
        return original(*args, **kwargs)
    except Exception as exc:
        # compile_fx_inner re-raises what codegen raised as an InductorError.
        if isinstance(getattr(exc, "inner_exception", exc), _CacheMiss):
            return None
        raise
    finally:
        compile_fx.fx_codegen_and_compile = codegen


@contextlib.contextmanager
def fork_inner_compiles(
    workers: int,
    timeout_s: float = 300.0,
    should_fork: Optional[Callable[[], bool]] = None,
    load_cache_hits: bool = False,
):
    """Run each Spyre inner compile in a forked child, at most ``workers`` at once.

    ``should_fork``, when given, is called at each inner compile. When it returns
    False, that compile runs in this process as usual and returns a real graph, so
    the caller can keep going past it (for example small graphs between the ones it
    wants in children).

    With ``load_cache_hits``, a compile whose graph is in the FX graph cache loads
    it in this process and returns the real graph instead of forking, so a restart
    with warm caches runs its graphs and forks nothing.

    Yields the pool. Leaving the block waits for every child, or kills them all if
    the block raised.
    """
    import torch_spyre._inductor as spyre_inductor

    original = spyre_inductor._spyre_inner_compile
    pool = _Pool(max(1, workers), timeout_s)

    def inner_compile(*args: Any, **kwargs: Any) -> Any:
        if should_fork is not None and not should_fork():
            pool.inline += 1
            return original(*args, **kwargs)
        if load_cache_hits:
            graph = _load_if_cached(original, args, kwargs)
            if graph is not None:
                pool.cached += 1
                return graph
        pool.submit(lambda: original(*args, **kwargs))
        return _StandIn(pool)

    started = time.time()
    spyre_inductor._spyre_inner_compile = inner_compile
    try:
        yield pool
    except BaseException:
        pool.kill_all()
        raise
    finally:
        spyre_inductor._spyre_inner_compile = original
        pool.wait()
        pool.open = False
    logger.info(
        "parallel precompile: %d compiles ok, %d failed, %d in this process, "
        "%d cache hits, %d workers, %.1f s",
        pool.ok,
        pool.failed,
        pool.inline,
        pool.cached,
        pool.workers,
        time.time() - started,
    )


class SecondPassCheck:
    """Count FX graph cache hits and misses of the run after the precompile.

    Use it around the second run of the graphs, as a context manager or with
    ``finish()``. ``expected`` is the number of compiles the children finished
    (``pool.ok``). Fewer hits than that (or, without ``expected``, any miss) logs a
    warning with the numbers, because the children's work was then lost.
    """

    def __init__(self, expected: Optional[int] = None) -> None:
        from torch._dynamo.utils import counters

        self._counters = counters["inductor"]
        self._start = (
            self._counters["fxgraph_cache_hit"],
            self._counters["fxgraph_cache_miss"],
        )
        self.expected = expected
        self.hits = self.misses = 0

    def finish(self) -> tuple[int, int]:
        self.hits = self._counters["fxgraph_cache_hit"] - self._start[0]
        self.misses = self._counters["fxgraph_cache_miss"] - self._start[1]
        if self.expected is None:
            lost = self.misses > 0
        else:  # misses beyond that are graphs the children never compiled
            lost = self.hits < self.expected
        log = logger.warning if lost else logger.info
        log(
            "parallel precompile: the second run hit the FX graph cache %d times "
            "(children finished %s) and compiled %d graphs%s",
            self.hits,
            self.expected,
            self.misses,
            ": the children's work was lost" if lost else "",
        )
        return self.hits, self.misses

    def __enter__(self) -> "SecondPassCheck":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.finish()
