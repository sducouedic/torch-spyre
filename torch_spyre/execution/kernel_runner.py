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

import torch
from torch_spyre._C import launch_jobplan, prepare_kernel
from torch_spyre._inductor.logging_utils import get_inductor_logger
from torch_spyre.profiler._ffdc import (
    CATEGORY_RUNTIME_LAUNCH,
    CATEGORY_UNIMPLEMENTED,
    with_ffdc,
)

logger = get_inductor_logger("kernel_runner")


class SpyreUnimplementedRunner:
    def __init__(self, name: str, op: str):
        self.kernel_name = name
        self.op = op

    @with_ffdc(CATEGORY_UNIMPLEMENTED, logger, code_dir_attr=None)
    def run(self, *args, **kw_args):
        raise RuntimeError(
            f"Invoked {self.kernel_name} which contains"
            f" unimplemented operation {self.op}"
        )


class SpyreSDSCKernelRunner:
    """Kernel runner for a compiled SDSC bundle.

    The live ``jobplan`` handle (a C-extension object returned by
    ``prepare_kernel``) is initialised lazily on first use so that this class
    is fully picklable.  Only ``kernel_name`` and ``code_dir`` are serialised;
    the handle is rebuilt from ``code_dir`` when the object is first called
    after deserialisation (e.g. after an FxGraphCache round-trip).
    """

    def __init__(self, name: str, code_dir: str):
        self.kernel_name = name
        self.code_dir = code_dir
        self._jobplan = None  # initialised lazily — not pickled

    # ------------------------------------------------------------------
    # Pickle support: serialise only the stable, path-based state so that
    # FxGraphCache can round-trip this object across process restarts.
    # ------------------------------------------------------------------
    def __getstate__(self):
        return {"kernel_name": self.kernel_name, "code_dir": self.code_dir}

    def __setstate__(self, state):
        self.kernel_name = state["kernel_name"]
        self.code_dir = state["code_dir"]
        self._jobplan = None  # will be rebuilt on first run()

    # ------------------------------------------------------------------
    # Lazy jobplan initialisation
    # ------------------------------------------------------------------
    @property
    def jobplan(self):
        if self._jobplan is None:
            logger.debug(
                "Initialising jobplan for %s from %s", self.kernel_name, self.code_dir
            )
            # prepare_kernel() → JobPlanBuilder → getDefaultStream() segfaults if
            # the C++ RuntimeContext is null.  _lazy_init() is idempotent and
            # ensures start_runtime() has been called before prepare_kernel().
            import torch
            torch.spyre._impl._lazy_init()
            self._jobplan = prepare_kernel(self.code_dir + "/spyreCodeDir")
        return self._jobplan

    @with_ffdc(CATEGORY_RUNTIME_LAUNCH, logger)
    def run(self, *args, **kw_args):
        logger.info("RUN: %s %s", self.kernel_name, self.code_dir)
        with torch.profiler.record_function(f"launch_jobplan:{self.kernel_name}"):
            launch_jobplan(self.jobplan, args)
