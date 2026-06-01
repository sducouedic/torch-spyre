# Copyright 2025-2026 The Torch-Spyre Authors.
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

import shutil
import tempfile
from collections.abc import Sequence
import os
import subprocess
import torch
import uuid

from torch._inductor.async_compile import AsyncCompile
from torch._inductor.runtime.runtime_utils import cache_dir
from torch_spyre._inductor import config as spyre_config
from torch_spyre._inductor.logging_utils import get_inductor_logger
from torch_spyre._inductor.op_spec import (
    LoopSpec,
    OpSpec,
    UnimplementedOp,
    find_unimplemented,
)
from torch_spyre._inductor.codegen.bundle import generate_bundle
from torch_spyre.profiler._ffdc import CATEGORY_COMPILE, try_collect
from .kernel_runner import SpyreSDSCKernelRunner, SpyreUnimplementedRunner
from .kernel_cache import (
    allocate_compile_dir,
    commit_compile_dir,
    compute_specs_hash,
    get_cached_kernel_dir,
)

logger = get_inductor_logger("sdsc_compile")


def get_output_dir(kernel_name: str):
    spyre_dir = os.path.join(cache_dir(), "inductor-spyre")
    os.makedirs(spyre_dir, exist_ok=True)
    digest = uuid.uuid4().hex[:8]
    kernel_output_dir = tempfile.mkdtemp(
        dir=spyre_dir, prefix=f"{digest}_{kernel_name}_"
    )
    return kernel_output_dir


class SpyreAsyncCompile(AsyncCompile):
    """Spyre kernel compilation (`sdsc`), plus the upstream AsyncCompile.

    A graph mixing Spyre and CPU work emits `async_compile.cpp_pybinding(...)`
    against this same object, so we inherit AsyncCompile for `cpp_pybinding`/
    `wait` rather than stubbing them -- a no-op `wait()` alone can't compile a
    CPU kernel it was never given.

    """

    def triton(self, *args, **kwargs):
        raise NotImplementedError(
            "SpyreAsyncCompile does not support Triton kernels; only "
            "cpp_pybinding (CPU) and sdsc (Spyre) are validated."
        )

    def cpp(self, *args, **kwargs):
        raise NotImplementedError(
            "SpyreAsyncCompile does not support the cpp() path; CPU kernels "
            "go through cpp_pybinding (cpu_backend='cpp')."
        )

    def sdsc(
        self, kernel_name: str, specs: Sequence[OpSpec | LoopSpec | UnimplementedOp]
    ):
        unimp = find_unimplemented(list(specs))
        if unimp is not None:
            logger.warning(
                f"WARNING: Compiling unimplemented {unimp.op} to runtime exception"
            )
            return SpyreUnimplementedRunner(kernel_name, unimp.op)

        use_cache = spyre_config.spyre_kernel_cache and not (
            torch._inductor.config.force_disable_caches
        )

        if use_cache:
            # Hash the specs in-memory BEFORE any disk I/O.  On a cache hit
            # neither generate_bundle nor dxp_standalone runs at all.
            cache_key = compute_specs_hash(specs)
            logger.info("Bundle cache key: %s", cache_key)

            cached_dir = get_cached_kernel_dir(cache_key)
            if cached_dir is not None:
                logger.info("Cache HIT: Using cached kernel from: %s", cached_dir)
                # Runner points at the persistent cache dir so it is picklable
                # for FxGraphCache across process restarts.
                return SpyreSDSCKernelRunner(kernel_name, cached_dir)

            logger.info("Cache MISS: Compiling kernel")

            # Allocate a temp dir INSIDE the cache root (same filesystem) so
            # the rename in commit_compile_dir is atomic on POSIX.  There is
            # no separate workspace: dxp_standalone writes directly here.
            compile_dir = allocate_compile_dir(cache_key)
            try:
                generate_bundle(kernel_name, compile_dir, specs)

                with torch.profiler.record_function(f"dxp_standalone:{kernel_name}"):
                    try:
                        subprocess.run(
                            ["dxp_standalone", "-d", compile_dir], check=True
                        )
                    except Exception as exc:
                        try_collect(
                            exc,
                            logger=logger,
                            failure_category=CATEGORY_COMPILE,
                            kernel_name=kernel_name,
                            code_dir=compile_dir,
                        )
                        raise

                cached_dir = commit_compile_dir(compile_dir, cache_key)
                compile_dir = None  # ownership transferred — do not delete
                logger.info("Kernel compiled and cached at: %s", cached_dir)
                return SpyreSDSCKernelRunner(kernel_name, cached_dir)
            finally:
                # compile_dir is None when commit_compile_dir succeeded.
                # On failure it is still our temp dir and must be removed.
                if compile_dir is not None and os.path.isdir(compile_dir):
                    shutil.rmtree(compile_dir, ignore_errors=True)

        # Caching disabled (SPYRE_KERNEL_CACHE=0 or force_disable_caches).
        # Compile into a throw-away temp dir that lives for this process only.
        output_dir = get_output_dir(kernel_name)
        generate_bundle(kernel_name, output_dir, specs)

        # Invoke backend compiler of SDSC Bundle
        with torch.profiler.record_function(f"dxp_standalone:{kernel_name}"):
            try:
                subprocess.run(["dxp_standalone", "-d", output_dir], check=True)
            except Exception as exc:
                try_collect(
                    exc,
                    logger=logger,
                    failure_category=CATEGORY_COMPILE,
                    kernel_name=kernel_name,
                    code_dir=output_dir,
                )
                raise

        return SpyreSDSCKernelRunner(kernel_name, output_dir)
