# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Compatibility polyfills for older Megatron-LM revisions.

This module is imported as a side-effect from ``megatron/bridge/__init__.py``
to monkey-patch ``sys.modules`` BEFORE any other Bridge code runs. It
adds back the symbols and submodules that were introduced in newer
Megatron-LM revisions but are missing from the
``super-vlm2``-pinned MLM (commit ``db659367``) we vendor in
``nemo-rl-super``.

This lets us keep our existing MLM pin while consuming a Bridge that
was authored against newer MLM, with zero consumer-side patches.

Each polyfill is gated on whether the symbol/module already exists, so
a future MLM bump that re-introduces the real definition transparently
takes precedence.

Identified by the static probe at
``/tmp/bridge_mlm_compat_check.py`` (8 mismatches as of the
``ec1cdd6b`` Bridge commit + ``db659367`` MLM commit).
"""

from __future__ import annotations

import sys
import types
from typing import Any


# ---------------------------------------------------------------------------
# 1. ``megatron.core.process_groups_config.ProcessGroupCollection``
#
# Newer Megatron-LM consolidated ``ModelCommProcessGroups`` +
# ``GradCommProcessGroups`` into a single ``ProcessGroupCollection``
# class with classmethod ``use_mpu_process_groups()``. Older MLM ships
# only the two predecessor classes (``ModelCommProcessGroups`` *does*
# have ``use_mpu_process_groups()``; ``GradCommProcessGroups`` does
# not).
#
# Bridge and Super both call this at runtime (not just as a type
# annotation):
#
#   pg_collection = ProcessGroupCollection.use_mpu_process_groups()
#   ... pg_collection.tp, pg_collection.pp, pg_collection.dp, ...
#
# So a ``ProcessGroupCollection = Any`` placeholder would crash with
# ``TypeError`` the moment ``setup.py`` builds the model. We need a
# real class whose ``use_mpu_process_groups`` returns an instance
# carrying both the model-parallel attrs (from
# ``ModelCommProcessGroups``) and the grad-comm attrs (``dp``,
# ``dp_cp``, ``expt_dp``, ``intra_dp_cp``, ``intra_expt_dp``,
# ``inter_dist_opt``).
#
# The polyfill below assembles such an instance by:
#   1. delegating to ``ModelCommProcessGroups.use_mpu_process_groups()``
#      for the model-parallel attrs (TP, PP, MP, CP, EP, ...)
#   2. calling ``parallel_state.get_*_parallel_group(...)`` directly
#      for the grad-comm attrs that older MLM exposes only on the
#      module level (since ``GradCommProcessGroups`` lacked the
#      classmethod)
#   3. silently skipping any field whose getter is missing from the
#      pinned ``parallel_state`` (e.g. the new ``intra_*`` ones)
#
# Field list mirrors fields(ModelCommProcessGroups) + fields(GradCommProcessGroups)
# in ``megatron/core/process_groups_config.py`` of our pinned MLM.
# ---------------------------------------------------------------------------
try:
    import megatron.core.process_groups_config as _pgc

    if not hasattr(_pgc, "ProcessGroupCollection"):
        from megatron.core.process_groups_config import (  # noqa: E402
            ModelCommProcessGroups as _ModelCommPGs,
        )

        class _ProcessGroupCollectionPolyfill:
            """Compat polyfill for ``ProcessGroupCollection`` on older MLM.

            Combines ``ModelCommProcessGroups`` + grad-comm process
            groups into a single object exposing the attribute surface
            ``Bridge`` / Super expect (``tp``, ``pp``, ``mp``, ``dp``,
            ``dp_cp``, ``cp``, ``tp_cp``, ``hcp``, ``ep``, ``expt_tp``,
            ``tp_ep``, ``tp_ep_pp``, ``embd``, ``pos_embd``, ``expt_dp``,
            ``intra_dp_cp``, ``intra_expt_dp``, ``inter_dist_opt``).
            Fields whose getters do not exist on the pinned
            ``megatron.core.parallel_state`` are simply not set.
            """

            # Declared as ``Any`` so basedpyright is happy with both the
            # ``setattr(instance, name, ...)`` and ``instance.name = ...``
            # styles below; the runtime values are
            # ``torch.distributed.ProcessGroup`` (or ``None`` on the
            # newer optionals when MLM does not have the getter).
            tp: Any = None
            pp: Any = None
            mp: Any = None
            cp: Any = None
            tp_cp: Any = None
            hcp: Any = None
            ep: Any = None
            expt_tp: Any = None
            tp_ep: Any = None
            tp_ep_pp: Any = None
            embd: Any = None
            pos_embd: Any = None
            expt_dp: Any = None
            dp: Any = None
            dp_cp: Any = None
            intra_dp_cp: Any = None
            intra_expt_dp: Any = None
            inter_dist_opt: Any = None

            @classmethod
            def use_mpu_process_groups(cls, required_pgs=None):
                from megatron.core import parallel_state as _ps

                instance = cls()

                # ---- model-parallel attrs (predecessor: ModelCommProcessGroups) ----
                model_pgs = _ModelCommPGs.use_mpu_process_groups(required_pgs)
                for attr in (
                    "tp",
                    "pp",
                    "mp",
                    "cp",
                    "tp_cp",
                    "hcp",
                    "ep",
                    "expt_tp",
                    "tp_ep",
                    "tp_ep_pp",
                    "embd",
                    "pos_embd",
                    "expt_dp",
                ):
                    if hasattr(model_pgs, attr):
                        setattr(instance, attr, getattr(model_pgs, attr))

                # ---- grad-comm attrs (predecessor: GradCommProcessGroups) ----
                # ``GradCommProcessGroups`` on the pinned MLM does not
                # have ``use_mpu_process_groups``; we wire each field
                # by calling the matching ``parallel_state.get_*_group``
                # directly. Older MLM exposes ``get_data_parallel_group``
                # with a ``with_context_parallel: bool`` flag (some
                # variants take it positional, some keyword), so try
                # both calling conventions.

                # ``dp`` (no CP)
                if not hasattr(instance, "dp"):
                    try:
                        instance.dp = _ps.get_data_parallel_group(
                            with_context_parallel=False
                        )
                    except TypeError:
                        try:
                            instance.dp = _ps.get_data_parallel_group(False)
                        except Exception:
                            instance.dp = _ps.get_data_parallel_group()

                # ``dp_cp`` (with CP)
                if not hasattr(instance, "dp_cp"):
                    try:
                        instance.dp_cp = _ps.get_data_parallel_group(
                            with_context_parallel=True
                        )
                    except TypeError:
                        try:
                            instance.dp_cp = _ps.get_data_parallel_group(True)
                        except Exception:
                            # Older MLM may not have CP-aware DP group;
                            # fall back to plain DP (correct for CP=1).
                            instance.dp_cp = instance.dp

                # ``expt_dp`` (already may be set by model_pgs above; only
                # set here if not)
                if not hasattr(instance, "expt_dp"):
                    try:
                        instance.expt_dp = _ps.get_expert_data_parallel_group(False)
                    except Exception:
                        try:
                            instance.expt_dp = _ps.get_expert_data_parallel_group()
                        except Exception:
                            pass

                # Newer MLM grad-comm attrs that may not exist on older
                # parallel_state at all. Set if available, skip
                # otherwise. Code paths that reference these (FSDP /
                # distributed-optimizer custom ranks) are not exercised
                # by the smoke.
                for attr, getter_name in (
                    ("intra_dp_cp", "get_intra_partial_data_parallel_group_with_cp"),
                    ("intra_expt_dp", "get_intra_expert_data_parallel_group"),
                    ("inter_dist_opt", "get_inter_distributed_optimizer_instance_group"),
                ):
                    getter = getattr(_ps, getter_name, None)
                    if getter is None:
                        continue
                    try:
                        setattr(instance, attr, getter(False))
                    except TypeError:
                        try:
                            setattr(instance, attr, getter())
                        except Exception:
                            pass

                return instance

        _pgc.ProcessGroupCollection = _ProcessGroupCollectionPolyfill  # type: ignore[attr-defined]
except Exception:  # ImportError, FileNotFoundError from broken TE, etc.
    pass


# ---------------------------------------------------------------------------
# 2. ``megatron.core.parallel_state.update_pg_timeout``
#
# Helper to mutate the default process-group timeout post-init. Used by
# ``megatron.bridge.training.train`` after ``initialize_megatron`` returns to
# bump the timeout for steady-state training. Older MLM defaults the
# timeout once at process-group creation; falling back to a no-op leaves
# the default in place, which matches the pre-helper behaviour.
# ---------------------------------------------------------------------------
try:
    import megatron.core.parallel_state as _ps

    if not hasattr(_ps, "update_pg_timeout"):
        def _update_pg_timeout(*_args, **_kwargs):  # type: ignore[no-redef]
            """No-op: older MLM does not support post-init timeout updates."""
            return None

        _ps.update_pg_timeout = _update_pg_timeout  # type: ignore[attr-defined]
except Exception:  # ImportError, FileNotFoundError from broken TE, etc.
    pass


# ---------------------------------------------------------------------------
# 3. ``megatron.core.pipeline_parallel.utils.is_pp_first_stage`` /
#    ``is_pp_last_stage``
#
# Newer MLM exposes pp_group-aware stage predicates under
# ``pipeline_parallel.utils``. Older MLM only has the global
# ``parallel_state.is_pipeline_first_stage`` / ``is_pipeline_last_stage``.
# Bridge calls them as ``is_pp_first_stage(pg_collection.pp)`` /
# ``is_pp_last_stage(pg_collection.pp)``; we ignore the explicit pp_group
# argument and delegate to the global parallel-state predicate, which
# matches the single-pp-group behaviour of older MLM.
# ---------------------------------------------------------------------------
try:
    from megatron.core import parallel_state as _ps_for_pp_utils

    try:
        import megatron.core.pipeline_parallel.utils as _pp_utils
    except Exception:  # ImportError, FileNotFoundError from broken TE, etc.
        # Make sure the parent package is loaded
        import megatron.core.pipeline_parallel  # noqa: F401
        _pp_utils = types.ModuleType("megatron.core.pipeline_parallel.utils")
        sys.modules["megatron.core.pipeline_parallel.utils"] = _pp_utils
        # Attach to parent package for ``from megatron.core.pipeline_parallel import utils``
        import megatron.core.pipeline_parallel as _pp_pkg
        _pp_pkg.utils = _pp_utils  # type: ignore[attr-defined]

    if not hasattr(_pp_utils, "is_pp_first_stage"):
        def _is_pp_first_stage(*_args, **_kwargs) -> bool:  # type: ignore[no-redef]
            """Delegate to global parallel_state (single-pp-group MLM)."""
            return _ps_for_pp_utils.is_pipeline_first_stage()

        _pp_utils.is_pp_first_stage = _is_pp_first_stage  # type: ignore[attr-defined]

    if not hasattr(_pp_utils, "is_pp_last_stage"):
        def _is_pp_last_stage(*_args, **_kwargs) -> bool:  # type: ignore[no-redef]
            """Delegate to global parallel_state (single-pp-group MLM)."""
            return _ps_for_pp_utils.is_pipeline_last_stage()

        _pp_utils.is_pp_last_stage = _is_pp_last_stage  # type: ignore[attr-defined]
except Exception:  # ImportError, FileNotFoundError from broken TE, etc.
    pass


# ---------------------------------------------------------------------------
# 4. ``megatron.core.jit.disable_jit_fuser``
#
# Newer MLM exposes a helper to disable PyTorch's NNC JIT fuser before
# graph capture. Bridge calls it from ``setup.py`` only when
# ``cfg.dist.disable_jit_fuser`` is true; in our smoke (and in the
# default config) that flag is false, so a no-op fallback is correct.
# ---------------------------------------------------------------------------
try:
    import megatron.core.jit as _jit

    if not hasattr(_jit, "disable_jit_fuser"):
        def _disable_jit_fuser():  # type: ignore[no-redef]
            """No-op: older MLM didn't ship the JIT fuser disable helper."""
            return None

        _jit.disable_jit_fuser = _disable_jit_fuser  # type: ignore[attr-defined]
except Exception:  # ImportError, FileNotFoundError from broken TE, etc.
    pass


# ---------------------------------------------------------------------------
# 5. ``megatron.core.fp8_utils.FP8_TENSOR_CLASS``
#
# Identifier (or class object) referencing the active FP8 tensor type
# used by Bridge's parameter mapping / refit path. Older MLM didn't
# export this constant. Bridge consumes it only on the FP8 refit code
# path (``megatron.bridge.models.conversion.param_mapping``); for our
# bf16 smoke nothing reads it, so ``None`` (sentinel meaning "no FP8
# class available") is a safe fallback.
# ---------------------------------------------------------------------------
try:
    import megatron.core.fp8_utils as _fp8u

    if not hasattr(_fp8u, "FP8_TENSOR_CLASS"):
        _fp8u.FP8_TENSOR_CLASS = None  # type: ignore[attr-defined]
    if not hasattr(_fp8u, "HAVE_TE_FP8_TENSOR_CLASS"):
        _fp8u.HAVE_TE_FP8_TENSOR_CLASS = False  # type: ignore[attr-defined]
except Exception:  # ImportError, FileNotFoundError from broken TE, etc.
    pass


# ---------------------------------------------------------------------------
# 6. ``megatron.core.transformer.cuda_graphs.TECudaGraphHelper``
#
# Newer MLM ships a TE-aware CUDA graph helper. Bridge instantiates it
# in ``train.py`` only when CUDA graphs are enabled in the config. In our
# smoke we set ``enforce_eager=true`` (no CUDA graphs), so a stub class
# whose constructor accepts arbitrary args and whose methods raise on
# unexpected use is sufficient.
# ---------------------------------------------------------------------------
try:
    import megatron.core.transformer.cuda_graphs as _cg

    if not hasattr(_cg, "TECudaGraphHelper"):
        class _TECudaGraphHelperStub:  # type: ignore[no-redef]
            """Stub used when older MLM doesn't ship TECudaGraphHelper.

            Bridge only instantiates this when CUDA graphs are enabled.
            Disable CUDA graphs (``enforce_eager=True`` /
            ``cudagraph_mode=NONE``) when running against this stub, or
            instantiation will raise.
            """

            def __init__(self, *args, **kwargs):
                raise NotImplementedError(
                    "TECudaGraphHelper polyfill from megatron.bridge._mlm_compat "
                    "was instantiated, which means CUDA graphs are enabled but "
                    "the pinned Megatron-LM revision does not ship the real "
                    "helper. Disable CUDA graphs in the run config."
                )

        _cg.TECudaGraphHelper = _TECudaGraphHelperStub  # type: ignore[attr-defined]
except Exception:  # ImportError, FileNotFoundError from broken TE, etc.
    pass


# ---------------------------------------------------------------------------
# 7. ``megatron.core.full_cuda_graph`` module + ``FullCudaGraphWrapper``
#
# An entire submodule that's missing in older MLM. Bridge imports
# ``FullCudaGraphWrapper`` from it in both ``train.py`` and ``eval.py``,
# but only wraps the forward-backward function when CUDA graphs are on.
# As with the helper above, our smoke disables CUDA graphs; a stub
# wrapper whose constructor raises clearly on use is sufficient.
# ---------------------------------------------------------------------------
if "megatron.core.full_cuda_graph" not in sys.modules:
    _full_cg_mod = types.ModuleType("megatron.core.full_cuda_graph")

    class _FullCudaGraphWrapperStub:  # type: ignore[no-redef]
        """Stub used when older MLM doesn't ship megatron.core.full_cuda_graph.

        Disable CUDA graphs in the run config when running against this
        stub, or wrapping will raise.
        """

        def __init__(self, *args, **kwargs):
            raise NotImplementedError(
                "FullCudaGraphWrapper polyfill from "
                "megatron.bridge._mlm_compat was instantiated. The pinned "
                "Megatron-LM revision does not ship megatron.core.full_cuda_graph; "
                "disable CUDA graphs in the run config."
            )

    _full_cg_mod.FullCudaGraphWrapper = _FullCudaGraphWrapperStub  # type: ignore[attr-defined]
    sys.modules["megatron.core.full_cuda_graph"] = _full_cg_mod

    # Attach to parent ``megatron.core`` so attribute access works too.
    try:
        import megatron.core as _core_pkg
        _core_pkg.full_cuda_graph = _full_cg_mod  # type: ignore[attr-defined]
    except Exception:  # ImportError, FileNotFoundError from broken TE, etc.
        pass
