# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for MiMo checkpoint saving wiring.

Tests validate that MiMo training correctly uses shared checkpoint helpers
(save_checkpoint_and_time, checkpoint_and_decide_exit) with the right
arguments, without actually saving/loading checkpoints.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import Mock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mimo_infra(*, num_active_pgs: int = 1) -> Mock:
    """Create a mock MimoModelInfra with the given number of active PG collections."""
    infra = Mock()
    pgs: Dict[str, Any] = {}
    for i in range(num_active_pgs):
        pgs[f"module_{i}"] = Mock()
    infra.pg_collections = pgs
    infra.module_to_grid_map = {"llm": Mock()}
    infra.topology = Mock()
    return infra


def _make_global_state(
    *,
    save_interval: int | None = 10,
    save_dir: str | None = "/tmp/ckpt",
    train_iters: int = 100,
    step: int = 0,
    non_persistent_save_interval: int | None = None,
    exit_signal_handler: bool = False,
    exit_duration_in_mins: float | None = None,
    exit_interval: int | None = None,
) -> SimpleNamespace:
    """Create a minimal GlobalState-like namespace for train_mimo tests."""
    timer_handle = Mock()
    timers = Mock(return_value=timer_handle)
    timers.log = Mock()

    state = SimpleNamespace(
        timers=timers,
        energy_monitor=None,
        cfg=SimpleNamespace(
            train=SimpleNamespace(
                train_iters=train_iters,
                micro_batch_size=1,
                exit_signal_handler=exit_signal_handler,
                exit_duration_in_mins=exit_duration_in_mins,
                exit_interval=exit_interval,
            ),
            dataset=SimpleNamespace(seq_length=128),
            checkpoint=SimpleNamespace(
                save=save_dir,
                save_interval=save_interval,
                non_persistent_save_interval=non_persistent_save_interval,
                async_save=False,
            ),
            ddp=SimpleNamespace(use_megatron_fsdp=False, overlap_param_gather=True),
            optimizer=SimpleNamespace(use_distributed_optimizer=True),
            model=SimpleNamespace(fp8=None, seq_length=128),
            logger=SimpleNamespace(
                log_progress=False,
                skip_train_metrics_log=True,
                timing_log_level=0,
                timing_log_option="minmax",
                log_timers_to_tensorboard=False,
                log_interval=1,
            ),
            profiling=None,
            data_parallel_size=1,
        ),
        train_state=SimpleNamespace(
            step=step,
            consumed_train_samples=0,
            floating_point_operations_so_far=0,
        ),
        start_time=time.time(),
        signal_handler=Mock(),
        nvrx_straggler_manager=None,
        tensorboard_logger=None,
        wandb_logger=None,
    )
    state.signal_handler.signals_received.return_value = []
    return state


# ---------------------------------------------------------------------------
# Tests: pg_collection forwarding in shared helpers
# ---------------------------------------------------------------------------


class TestPgCollectionForwarding:
    """Verify save_checkpoint_and_time and checkpoint_and_decide_exit
    forward pg_collection to save_checkpoint."""

    @patch("megatron.bridge.training.train.force_param_sync")
    @patch("megatron.bridge.training.train.should_disable_forward_pre_hook", return_value=False)
    @patch("megatron.bridge.training.train.save_checkpoint")
    def test_save_checkpoint_and_time_forwards_pg_collection(
        self,
        mock_save_checkpoint,
        mock_should_disable,
        mock_force_param_sync,
    ):
        from megatron.bridge.training.train import save_checkpoint_and_time

        state = _make_global_state()
        pg = Mock()

        save_checkpoint_and_time(
            state=state,
            model=[Mock()],
            optimizer=Mock(),
            opt_param_scheduler=Mock(),
            num_floating_point_operations_so_far=0,
            checkpointing_context={},
            pg_collection=pg,
        )

        _, kwargs = mock_save_checkpoint.call_args
        assert kwargs["pg_collection"] is pg

    @patch("megatron.bridge.training.train.force_param_sync")
    @patch("megatron.bridge.training.train.should_disable_forward_pre_hook", return_value=False)
    @patch("megatron.bridge.training.train.save_checkpoint")
    def test_save_checkpoint_and_time_defaults_pg_collection_to_none(
        self,
        mock_save_checkpoint,
        mock_should_disable,
        mock_force_param_sync,
    ):
        from megatron.bridge.training.train import save_checkpoint_and_time

        state = _make_global_state()

        save_checkpoint_and_time(
            state=state,
            model=[Mock()],
            optimizer=Mock(),
            opt_param_scheduler=Mock(),
            num_floating_point_operations_so_far=0,
            checkpointing_context={},
        )

        _, kwargs = mock_save_checkpoint.call_args
        assert kwargs["pg_collection"] is None

    @patch("megatron.bridge.training.train.save_checkpoint_and_time")
    @patch("megatron.bridge.training.train.barrier_and_log")
    @patch("megatron.bridge.training.train.check_nvrx_straggler_detection", return_value=False)
    def test_checkpoint_and_decide_exit_forwards_pg_collection(
        self,
        mock_check_nvrx,
        mock_barrier_log,
        mock_save_and_time,
    ):
        from megatron.bridge.training.train import checkpoint_and_decide_exit

        state = _make_global_state(save_interval=5, step=10)
        pg = Mock()

        checkpoint_and_decide_exit(
            state=state,
            model=[Mock()],
            optimizer=Mock(),
            opt_param_scheduler=Mock(),
            num_floating_point_operations_so_far=0,
            checkpointing_context={},
            train_data_iterator=None,
            pg_collection=pg,
        )

        _, kwargs = mock_save_and_time.call_args
        assert kwargs["pg_collection"] is pg


# ---------------------------------------------------------------------------
# Tests: pretrain_mimo setup wiring
# ---------------------------------------------------------------------------


class TestPretrainMimoSetup:
    """Verify pretrain_mimo properly initializes checkpointing runtime."""

    @patch("megatron.bridge.training.pretrain_mimo.init_checkpointing_context")
    @patch("torch.distributed.all_reduce")
    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.get_world_size", return_value=2)
    def test_setup_mimo_initializes_checkpointing_context(
        self,
        mock_world_size,
        mock_get_rank,
        mock_all_reduce,
        mock_init_ctx,
    ):
        from megatron.bridge.training.pretrain_mimo import setup_mimo

        mock_init_ctx.return_value = {"test": "context"}

        global_state = Mock()
        global_state.start_time = time.time()
        global_state.cfg = None

        cfg = Mock()
        cfg.checkpoint = Mock()
        cfg.train = Mock()
        cfg.train.grad_reduce_in_fp32 = False
        cfg.train.overlap_grad_reduce = True
        cfg.train.use_distributed_optimizer = False
        cfg.train.check_for_nan_in_grad = False
        cfg.model = Mock()
        cfg.model.fp16 = False
        cfg.model.bf16 = True

        provider = Mock()
        infra = Mock()
        infra.module_to_grid_map = {"llm": Mock()}
        infra.topology = Mock()
        infra.pg_collections = {"llm": Mock()}
        provider.build_infra.return_value = infra

        model = Mock()
        model_config = Mock()
        model_config.pipeline_dtype = None
        model_config.bf16 = True
        provider.provide_distributed_model.return_value = [model]

        with patch("megatron.bridge.training.pretrain_mimo.get_model_config", return_value=model_config):
            with patch("megatron.bridge.training.pretrain_mimo.validate_no_stub_ranks"):
                with patch("megatron.bridge.training.pretrain_mimo.build_pg_collection_for_schedule"):
                    with patch("megatron.bridge.training.pretrain_mimo.get_module_to_grid_tuple"):
                        result = setup_mimo(cfg, provider, global_state=global_state)

        mock_init_ctx.assert_called_once_with(cfg.checkpoint)
        global_state.initialize_async_checkpoint_worker.assert_called_once()
        assert result.checkpointing_context == {"test": "context"}

    def test_rampup_guard_rejects_rampup_batch_size(self):
        from megatron.bridge.training.pretrain_mimo import pretrain_mimo

        cfg = Mock()
        cfg.train.rampup_batch_size = [100, 200, 300]

        with pytest.raises(AssertionError, match="Microbatch rampup is not supported"):
            pretrain_mimo(
                cfg=cfg,
                mimo_provider=Mock(),
                forward_step_func=Mock(),
                build_data_iterators_fn=Mock(),
                opt_config=Mock(),
            )


# ---------------------------------------------------------------------------
# Tests: non-colocated runtime guard
# ---------------------------------------------------------------------------


class TestNonColocatedGuard:
    """Verify the non-colocated topology assertion in train_mimo."""

    def test_rejects_multiple_active_pgs(self):
        from megatron.bridge.training.train_mimo import train_mimo

        infra = _make_mimo_infra(num_active_pgs=2)
        state = _make_global_state(train_iters=0)

        with pytest.raises(AssertionError, match="exactly one active ProcessGroupCollection"):
            train_mimo(
                forward_step_func=Mock(),
                model=Mock(),
                optimizer=Mock(),
                schedulers={},
                train_data_iterator=Mock(),
                valid_data_iterator=None,
                global_state=state,
                mimo_infra=infra,
                multimodule_communicator=Mock(),
                checkpointing_context={},
            )

    def test_rejects_zero_active_pgs(self):
        from megatron.bridge.training.train_mimo import train_mimo

        infra = _make_mimo_infra(num_active_pgs=0)
        state = _make_global_state(train_iters=0)

        with pytest.raises(AssertionError, match="exactly one active ProcessGroupCollection"):
            train_mimo(
                forward_step_func=Mock(),
                model=Mock(),
                optimizer=Mock(),
                schedulers={},
                train_data_iterator=Mock(),
                valid_data_iterator=None,
                global_state=state,
                mimo_infra=infra,
                multimodule_communicator=Mock(),
                checkpointing_context={},
            )


# ---------------------------------------------------------------------------
# Tests: checkpoint_and_decide_exit integration in train_mimo
# ---------------------------------------------------------------------------


class TestTrainMimoCheckpointIntegration:
    """Verify train_mimo calls checkpoint_and_decide_exit with the right args."""

    @patch("megatron.bridge.training.train_mimo.checkpoint_and_decide_exit", return_value=False)
    @patch("megatron.bridge.training.train_mimo.maybe_finalize_async_save")
    @patch("megatron.bridge.training.train_mimo.train_step_mimo")
    @patch("megatron.bridge.training.train_mimo.build_pg_collection_for_schedule")
    @patch("megatron.bridge.training.train_mimo.get_module_to_grid_tuple")
    @patch("megatron.bridge.training.train_mimo.get_model_config")
    @patch("megatron.bridge.training.train_mimo.prepare_forward_step_func")
    @patch("megatron.bridge.training.train_mimo.get_num_microbatches", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.get_world_size", return_value=1)
    def test_calls_checkpoint_and_decide_exit_with_pg_collection(
        self,
        mock_world_size,
        mock_rank,
        mock_num_mb,
        mock_prep_fwd,
        mock_get_config,
        mock_get_grid,
        mock_build_pg,
        mock_train_step,
        mock_async_finalize,
        mock_ckpt_exit,
    ):
        from megatron.bridge.training.train_mimo import train_mimo

        mock_train_step.return_value = ({}, 0, 0.0, 0)
        mock_config = Mock()
        mock_config.variable_seq_lengths = True
        mock_get_config.return_value = mock_config

        pg = Mock()
        infra = Mock()
        infra.pg_collections = {"llm": pg}
        infra.module_to_grid_map = {"llm": Mock()}
        infra.topology = Mock()

        mock_build_pg.return_value = Mock(spec=[])  # not a list

        state = _make_global_state(train_iters=1, step=0)
        ctx = {"key": "value"}
        train_iter = Mock()

        train_mimo(
            forward_step_func=Mock(),
            model=Mock(),
            optimizer=Mock(),
            schedulers={"llm": Mock()},
            train_data_iterator=train_iter,
            valid_data_iterator=None,
            global_state=state,
            mimo_infra=infra,
            multimodule_communicator=Mock(),
            checkpointing_context=ctx,
        )

        mock_ckpt_exit.assert_called_once()
        _, kwargs = mock_ckpt_exit.call_args
        assert kwargs["pg_collection"] is pg
        assert kwargs["checkpointing_context"] is ctx
        assert kwargs["train_data_iterator"] is train_iter
        assert kwargs["num_floating_point_operations_so_far"] == 0

    @patch("megatron.bridge.training.train_mimo.checkpoint_and_decide_exit", return_value=True)
    @patch("megatron.bridge.training.train_mimo.maybe_finalize_async_save")
    @patch("megatron.bridge.training.train_mimo.train_step_mimo")
    @patch("megatron.bridge.training.train_mimo.build_pg_collection_for_schedule")
    @patch("megatron.bridge.training.train_mimo.get_module_to_grid_tuple")
    @patch("megatron.bridge.training.train_mimo.get_model_config")
    @patch("megatron.bridge.training.train_mimo.prepare_forward_step_func")
    @patch("megatron.bridge.training.train_mimo.get_num_microbatches", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.get_world_size", return_value=1)
    def test_exits_loop_when_checkpoint_and_decide_exit_returns_true(
        self,
        mock_world_size,
        mock_rank,
        mock_num_mb,
        mock_prep_fwd,
        mock_get_config,
        mock_get_grid,
        mock_build_pg,
        mock_train_step,
        mock_async_finalize,
        mock_ckpt_exit,
    ):
        from megatron.bridge.training.train_mimo import train_mimo

        mock_train_step.return_value = ({}, 0, 0.0, 0)
        mock_config = Mock()
        mock_config.variable_seq_lengths = True
        mock_get_config.return_value = mock_config

        infra = Mock()
        infra.pg_collections = {"llm": Mock()}
        infra.module_to_grid_map = {"llm": Mock()}
        infra.topology = Mock()
        mock_build_pg.return_value = Mock(spec=[])

        state = _make_global_state(train_iters=100, step=0)

        train_mimo(
            forward_step_func=Mock(),
            model=Mock(),
            optimizer=Mock(),
            schedulers={"llm": Mock()},
            train_data_iterator=Mock(),
            valid_data_iterator=None,
            global_state=state,
            mimo_infra=infra,
            multimodule_communicator=Mock(),
            checkpointing_context={},
        )

        # Should have exited after 1 iteration, not 100
        assert mock_train_step.call_count == 1
        assert state.train_state.step == 1

    @patch("megatron.bridge.training.train_mimo.checkpoint_and_decide_exit", return_value=False)
    @patch("megatron.bridge.training.train_mimo.maybe_finalize_async_save")
    @patch("megatron.bridge.training.train_mimo.train_step_mimo")
    @patch("megatron.bridge.training.train_mimo.build_pg_collection_for_schedule")
    @patch("megatron.bridge.training.train_mimo.get_module_to_grid_tuple")
    @patch("megatron.bridge.training.train_mimo.get_model_config")
    @patch("megatron.bridge.training.train_mimo.prepare_forward_step_func")
    @patch("megatron.bridge.training.train_mimo.get_num_microbatches", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.get_world_size", return_value=1)
    def test_async_finalize_called_at_top_of_loop(
        self,
        mock_world_size,
        mock_rank,
        mock_num_mb,
        mock_prep_fwd,
        mock_get_config,
        mock_get_grid,
        mock_build_pg,
        mock_train_step,
        mock_async_finalize,
        mock_ckpt_exit,
    ):
        from megatron.bridge.training.train_mimo import train_mimo

        mock_train_step.return_value = ({}, 0, 0.0, 0)
        mock_config = Mock()
        mock_config.variable_seq_lengths = True
        mock_get_config.return_value = mock_config

        infra = Mock()
        infra.pg_collections = {"llm": Mock()}
        infra.module_to_grid_map = {"llm": Mock()}
        infra.topology = Mock()
        mock_build_pg.return_value = Mock(spec=[])

        state = _make_global_state(train_iters=2, step=0)

        train_mimo(
            forward_step_func=Mock(),
            model=Mock(),
            optimizer=Mock(),
            schedulers={"llm": Mock()},
            train_data_iterator=Mock(),
            valid_data_iterator=None,
            global_state=state,
            mimo_infra=infra,
            multimodule_communicator=Mock(),
            checkpointing_context={},
        )

        # 2 non-blocking calls (top of each iteration) + 1 blocking call (shutdown)
        assert mock_async_finalize.call_count == 3

        non_blocking_calls = [
            c for c in mock_async_finalize.call_args_list if c.kwargs.get("blocking") is False
        ]
        blocking_calls = [
            c for c in mock_async_finalize.call_args_list if c.kwargs.get("blocking") is True
        ]
        assert len(non_blocking_calls) == 2
        assert len(blocking_calls) == 1
        assert blocking_calls[0].kwargs.get("terminate") is True

    @patch("megatron.bridge.training.train_mimo.checkpoint_and_decide_exit", return_value=False)
    @patch("megatron.bridge.training.train_mimo.maybe_finalize_async_save")
    @patch("megatron.bridge.training.train_mimo.train_step_mimo")
    @patch("megatron.bridge.training.train_mimo.build_pg_collection_for_schedule")
    @patch("megatron.bridge.training.train_mimo.get_module_to_grid_tuple")
    @patch("megatron.bridge.training.train_mimo.get_model_config")
    @patch("megatron.bridge.training.train_mimo.prepare_forward_step_func")
    @patch("megatron.bridge.training.train_mimo.get_num_microbatches", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.get_world_size", return_value=1)
    def test_no_inline_save_checkpoint_call(
        self,
        mock_world_size,
        mock_rank,
        mock_num_mb,
        mock_prep_fwd,
        mock_get_config,
        mock_get_grid,
        mock_build_pg,
        mock_train_step,
        mock_async_finalize,
        mock_ckpt_exit,
    ):
        """Verify there is no inline save_checkpoint call — all saves go through
        checkpoint_and_decide_exit."""
        from megatron.bridge.training.train_mimo import train_mimo

        mock_train_step.return_value = ({}, 0, 0.0, 0)
        mock_config = Mock()
        mock_config.variable_seq_lengths = True
        mock_get_config.return_value = mock_config

        infra = Mock()
        infra.pg_collections = {"llm": Mock()}
        infra.module_to_grid_map = {"llm": Mock()}
        infra.topology = Mock()
        mock_build_pg.return_value = Mock(spec=[])

        state = _make_global_state(save_interval=1, train_iters=3, step=0)

        with patch("megatron.bridge.training.train_mimo.save_checkpoint_and_time") as mock_direct_save:
            train_mimo(
                forward_step_func=Mock(),
                model=Mock(),
                optimizer=Mock(),
                schedulers={"llm": Mock()},
                train_data_iterator=Mock(),
                valid_data_iterator=None,
                global_state=state,
                mimo_infra=infra,
                multimodule_communicator=Mock(),
                checkpointing_context={},
            )

            # save_checkpoint_and_time should NOT be called directly from train_mimo.
            # All saves should go through checkpoint_and_decide_exit.
            mock_direct_save.assert_not_called()

        # But checkpoint_and_decide_exit should have been called
        assert mock_ckpt_exit.call_count == 3
