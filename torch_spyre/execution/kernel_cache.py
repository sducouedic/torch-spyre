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

import json
import os
import shutil
import uuid
from collections.abc import Sequence
from functools import lru_cache
from typing import Optional

import torch
from torch._inductor.codecache import code_hash
from torch._inductor.runtime.runtime_utils import cache_dir

from torch_spyre._inductor.logging_utils import get_inductor_logger


logger = get_inductor_logger("kernel_cache")

# All artifacts that dxp_standalone must produce for a valid compiled kernel.
# A cache entry is only considered a hit if every one of these is present.
_REQUIRED_ARTIFACTS = [
    "bundle.mlir",
    os.path.join("spyreCodeDir", "init_binary.bin"),
    os.path.join("spyreCodeDir", "spyrecode.json"),
]


@lru_cache(maxsize=1)
def _get_spyre_library_versions() -> dict[str, str]:
    """Return all Spyre library versions (deeptools, senlib, etc.) from LIB_VERSION_FILE.

    Reads from the file specified by the LIB_VERSION_FILE environment variable.
    This file should contain lines in the format "library-name:version".
    Returns a dict mapping library names to their versions (e.g., ibm-deeptools,
    ibm-senlib-core).

    Raises RuntimeError if LIB_VERSION_FILE is not set or the file cannot be read.
    To disable kernel caching, set SPYRE_KERNEL_CACHE=0.
    """
    lib_version_file = os.environ.get("LIB_VERSION_FILE")
    if not lib_version_file:
        raise RuntimeError(
            "LIB_VERSION_FILE environment variable is required for kernel caching. "
            "It should point to a .txt file containing Spyre library versions"
            "versions (deeptools, senlib, etc.). "
            "To disable caching, set SPYRE_KERNEL_CACHE=0."
        )

    try:
        libraries = {}
        with open(lib_version_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                name, version = line.split(":", 1)
                libraries[name.strip()] = version.strip()
        logger.info("Loaded %d Spyre library versions from %s",
                    len(libraries), lib_version_file)
        return libraries
    except FileNotFoundError as e:
        raise RuntimeError(
            f"LIB_VERSION_FILE={lib_version_file} not found. "
            "To disable caching, set SPYRE_KERNEL_CACHE=0."
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Error reading Spyre library versions from {lib_version_file}: {e}. "
            "To disable caching, set SPYRE_KERNEL_CACHE=0."
        ) from e


@lru_cache(maxsize=1)
def _get_torch_spyre_version() -> str:
    """Return torch_spyre version string, used as part of the cache key.

    Falls back to ``"unknown"`` with a warning if ``torch_spyre.__version__``
    is not set.  The cache will still work but will not be invalidated on
    torch-spyre upgrades.
    """
    try:
        import torch_spyre

        version = getattr(torch_spyre, "__version__", None)
        if version is None:
            logger.warning(
                "torch_spyre.__version__ is not set; cache key will use "
                "'unknown' for torch-spyre version. Kernel cache will not "
                "be invalidated on torch-spyre upgrades."
            )
            return "unknown"
        return version
    except ImportError as e:
        logger.warning(
            "Could not import torch_spyre to read version (%s); cache key "
            "will use 'unknown' for torch-spyre version.",
            e,
        )
        return "unknown"


def get_cache_root_dir() -> str:
    """Return the root directory for the persistent kernel cache."""
    cache_root = os.path.join(cache_dir(), "inductor-spyre-cache")
    os.makedirs(cache_root, exist_ok=True)
    return cache_root


# ---------------------------------------------------------------------------
# In-memory cache key — computed BEFORE any disk I/O
# ---------------------------------------------------------------------------


def _symbol_kind_key(kind) -> list:
    """Return the cache-relevant fields of a ``SymbolKind``.

    Only the parts of a symbol that are **baked into the compiled bundle** may
    enter the cache key.  Splitting ``SymbolKind`` this way is what makes the
    key both correct and reusable:

    The corresponding *value* in ``base_symbol_values`` (the raw HBM byte
    address) is deliberately **not** included: it is supplied at launch as an
    ``!sdscbundle.input_arg`` parameter, so two runs that place the same tensor
    at different addresses must still share a cache entry.  Hashing it would
    make every key allocation-specific and reduce the hit rate to ~zero.
    """
    return [
        kind.kind,
        kind.base_sym_idx,
        kind.offset,
        kind.arg_index,
        kind.granularity,
        kind.max_value,
        kind.pytorch_sym,
        kind.core_idx,
        kind.split_count,
    ]


def compute_specs_hash(specs: Sequence) -> str:
    """Compute a cache key directly from OpSpec objects — no disk I/O required.

    This is the preferred hashing entry point.  Because it operates entirely
    in-memory it allows the cache lookup to happen *before* ``generate_bundle``
    writes any files, so a cache hit skips ``generate_bundle`` and
    ``dxp_standalone`` entirely.

    The key is a SHA-256 hash (via Inductor's ``code_hash``) that covers:

    * The JSON serialisation of every ``sdsc_N.json`` dict produced by
      ``compile_op_spec`` — this captures the full op structure, iteration
      space, tiling, tensor shapes, and dtypes.
    * The ``SymbolKind`` *structure* of every registered symbol. This covers
      the compile-time address arithmetic that is baked into ``bundle.mlir``
      but is absent from the ``sdsc_N.json`` dicts.
    * ``torch.__version__`` — invalidates on PyTorch upgrades.
    * ``torch_spyre.__version__`` — invalidates on torch-spyre upgrades.
    * Spyre library versions (deeptools, senlib, etc.) from LIB_VERSION_FILE —
      invalidates when any Spyre tool version changes. Requires LIB_VERSION_FILE
      to be set; caching is disabled if it is not.

    ``bundle.mlir`` is not hashed directly, but everything in it that is not
    already implied by the ``sdsc_N.json`` dicts *is* covered via the
    ``SymbolKind`` structure.
    """
    from torch_spyre._inductor.codegen.superdsc import compile_op_spec
    from torch_spyre._inductor.op_spec import LoopSpec, OpSpec

    specs_list = list(specs)

    content_parts: list[bytes] = []
    symbols: list[int] = []
    symbol_id_offset = 0
    sdsc_idx = 0

    def _collect(entries):
        nonlocal sdsc_idx, symbol_id_offset
        for entry in entries:
            if isinstance(entry, LoopSpec):
                _collect(entry.body)
            elif isinstance(entry, OpSpec):
                sdsc_json, local_sym_values, _, symbol_kinds = compile_op_spec(
                    sdsc_idx,
                    entry,
                    symbols,
                    symbol_id_offset,
                )
                symbol_id_offset += len(local_sym_values)
                sdsc_idx += 1
                content_parts.append(json.dumps(sdsc_json, sort_keys=True).encode())

                # The sdsc_json refers to addresses only as opaque negative symbol
                # ids, so we also hash the symbol structure to keep them distinct
                content_parts.append(
                    json.dumps(
                        [_symbol_kind_key(k) for k in symbol_kinds],
                        sort_keys=True,
                    ).encode()
                )

    _collect(specs_list)

    content = b"||".join(content_parts)

    # Build Spyre library versions string for cache key (sorted for determinism)
    library_versions = _get_spyre_library_versions()
    libraries_str = json.dumps(library_versions, sort_keys=True)

    extra = "||".join(
        [
            torch.__version__,
            _get_torch_spyre_version(),
            libraries_str,
        ]
    )

    cache_key = code_hash(content, extra=extra)
    logger.info(
        "Computed specs hash from %d OpSpec(s): %s", len(content_parts), cache_key
    )
    return cache_key


# ---------------------------------------------------------------------------
# Cache lookup
# ---------------------------------------------------------------------------


def get_cached_kernel_dir(cache_key: str) -> Optional[str]:
    """Return the cached kernel directory if all required artifacts are present.

    Validates that every artifact in ``_REQUIRED_ARTIFACTS`` exists and that
    at least one ``sdsc_N.json`` file is present.  A partial write from a
    killed process will fail this check and trigger recompilation.
    """
    cache_root = get_cache_root_dir()
    cached_dir = os.path.join(cache_root, cache_key)

    if not os.path.isdir(cached_dir):
        logger.info("Cache MISS: No cached kernel found for key %s", cache_key)
        return None

    missing = [
        p
        for p in _REQUIRED_ARTIFACTS
        if not os.path.isfile(os.path.join(cached_dir, p))
    ]
    if missing:
        logger.info(
            "Cache MISS: Cached dir exists but missing artifacts %s for key %s",
            missing,
            cache_key,
        )
        return None

    has_sdsc = any(
        f.startswith("sdsc_") and f.endswith(".json")
        for f in os.listdir(cached_dir)
        if os.path.isfile(os.path.join(cached_dir, f))
    )
    if not has_sdsc:
        logger.info(
            "Cache MISS: No sdsc_N.json files found in cached dir for key %s",
            cache_key,
        )
        return None

    logger.info("Cache HIT: Found cached kernel at %s", cached_dir)
    return cached_dir


# ---------------------------------------------------------------------------
# Cache write — single-directory approach (no copy step)
# ---------------------------------------------------------------------------


def allocate_compile_dir(cache_key: str) -> str:
    """Reserve a unique temp directory *inside* the cache root for compilation.

    ``dxp_standalone`` writes its output directly into this directory.  After
    compilation the caller promotes it to the final cache entry with
    ``commit_compile_dir``.  Because the temp dir lives inside the same
    filesystem as the final entry, the promotion is an atomic ``os.rename``.

    Using a directory inside the cache root (rather than the system
    ``/tmp``) ensures same-filesystem atomicity on POSIX.
    """
    cache_root = get_cache_root_dir()
    tmp_dir = os.path.join(cache_root, f"{cache_key}.tmp.{uuid.uuid4().hex}")
    os.makedirs(tmp_dir, exist_ok=True)
    return tmp_dir


def commit_compile_dir(tmp_dir: str, cache_key: str) -> str:
    """Atomically promote *tmp_dir* to the final ``<cache_root>/<cache_key>/`` entry.

    If another process already committed the same key, the loser discards its
    temp directory and reuses the winner's copy.  Returns the final cache dir.
    """
    cache_root = get_cache_root_dir()
    cached_dir = os.path.join(cache_root, cache_key)

    if os.path.isdir(cached_dir):
        # Another process/thread won the race — discard our copy.
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.info("Cache race resolved: reusing existing entry at %s", cached_dir)
        return cached_dir

    try:
        os.rename(tmp_dir, cached_dir)  # Atomic on POSIX (same filesystem)
        logger.info("Saved compiled kernel to cache: %s", cached_dir)
    except OSError:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.info("Cache race resolved: reusing existing entry at %s", cached_dir)

    return cached_dir


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def get_cache_stats() -> dict:
    """Return summary statistics about the persistent kernel cache."""
    cache_root = get_cache_root_dir()

    if not os.path.exists(cache_root):
        return {"total_cached_kernels": 0, "cache_size_mb": 0.0}

    cached_dirs = [
        d
        for d in os.listdir(cache_root)
        if os.path.isdir(os.path.join(cache_root, d))
        and not d.endswith(".tmp")
        and ".tmp." not in d
    ]

    total_size = 0
    for dirpath, _dirnames, filenames in os.walk(cache_root):
        for filename in filenames:
            total_size += os.path.getsize(os.path.join(dirpath, filename))

    return {
        "total_cached_kernels": len(cached_dirs),
        "cache_size_mb": total_size / (1024 * 1024),
    }


def clear_cache() -> None:
    """Remove all entries from the persistent kernel cache."""
    cache_root = get_cache_root_dir()
    if os.path.exists(cache_root):
        shutil.rmtree(cache_root)
        os.makedirs(cache_root, exist_ok=True)
        logger.info("Kernel cache cleared")
