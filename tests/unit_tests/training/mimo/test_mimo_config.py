# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
from types import SimpleNamespace

import pytest

from megatron.bridge.models.mimo.mimo_config import MimoParallelismConfig, ModuleParallelismConfig
from megatron.bridge.training.config import ConfigContainer


def test_module_parallelism_finalize_computes_dp():
    parallelism = ModuleParallelismConfig(tensor_model_parallel_size=2, pipeline_model_parallel_size=2)
    parallelism.finalize(world_size=16)
    assert parallelism.data_parallel_size == 4
    assert parallelism.total_model_parallel_size == 4
    assert parallelism.total_ranks == 16


def test_module_parallelism_finalize_invalid_world_size():
    parallelism = ModuleParallelismConfig(tensor_model_parallel_size=3, pipeline_model_parallel_size=2)
    with pytest.raises(ValueError, match="world_size .* not divisible"):
        parallelism.finalize(world_size=10)


def test_mimo_parallelism_finalize_requires_llm():
    module_parallelisms = {
        "vision": ModuleParallelismConfig(data_parallel_size=4),
    }
    mimo = MimoParallelismConfig(
        module_parallelisms=module_parallelisms,
    )
    with pytest.raises(ValueError, match="LLM module 'llm'"):
        mimo.finalize(world_size=None)


def test_mimo_heterogeneous_rank_offset_overlap():
    module_parallelisms = {
        "vision": ModuleParallelismConfig(data_parallel_size=4, rank_offset=0),
        "llm": ModuleParallelismConfig(data_parallel_size=4, rank_offset=2),
    }
    mimo = MimoParallelismConfig(
        module_parallelisms=module_parallelisms,
    )
    with pytest.raises(ValueError, match="overlap"):
        mimo.finalize(world_size=None)


def test_mimo_heterogeneous_valid_contiguous():
    module_parallelisms = {
        "vision": ModuleParallelismConfig(data_parallel_size=2, rank_offset=0),
        "llm": ModuleParallelismConfig(data_parallel_size=4, rank_offset=2),
    }
    mimo = MimoParallelismConfig(
        module_parallelisms=module_parallelisms,
    )
    mimo.finalize(world_size=None)
    assert mimo.total_world_size == 6


def _make_cfg(
    mimo_parallelism_config: MimoParallelismConfig,
    modality_submodules_spec=None,
) -> ConfigContainer:
    if modality_submodules_spec is None:
        modality_submodules_spec = {}
    model = SimpleNamespace(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        mimo_parallelism_config=mimo_parallelism_config,
        modality_submodules_spec=modality_submodules_spec,
    )
    train = SimpleNamespace(global_batch_size=8)
    placeholder = SimpleNamespace()
    return ConfigContainer(
        train=train,
        model=model,
        optimizer=placeholder,
        scheduler=placeholder,
        dataset=placeholder,
        logger=placeholder,
        tokenizer=placeholder,
        checkpoint=placeholder,
    )


def test_mimo_missing_modality_submodules(monkeypatch):
    module_parallelisms = {
        "vision": ModuleParallelismConfig(data_parallel_size=8),
        "llm": ModuleParallelismConfig(data_parallel_size=8),
    }
    mimo_parallelism_config = MimoParallelismConfig(
        module_parallelisms=module_parallelisms,
    )
    monkeypatch.setattr("megatron.bridge.training.config.get_world_size_safe", lambda: 1)
    cfg = _make_cfg(mimo_parallelism_config=mimo_parallelism_config, modality_submodules_spec={})
    with pytest.raises(ValueError, match="modality_submodules_spec missing modules"):
        cfg._validate_mimo()


def test_mimo_modality_submodule_unknown_key(monkeypatch):
    module_parallelisms = {
        "vision": ModuleParallelismConfig(data_parallel_size=8),
        "llm": ModuleParallelismConfig(data_parallel_size=8),
    }
    mimo_parallelism_config = MimoParallelismConfig(
        module_parallelisms=module_parallelisms,
    )
    monkeypatch.setattr("megatron.bridge.training.config.get_world_size_safe", lambda: 1)
    cfg = _make_cfg(
        mimo_parallelism_config=mimo_parallelism_config,
        modality_submodules_spec={"other": object()},
    )
    with pytest.raises(ValueError, match="unknown modules"):
        cfg._validate_mimo()
