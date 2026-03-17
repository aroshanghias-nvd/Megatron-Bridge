# Bug: MiMo PP > 1 crashes with `embedding()` receiving `None` indices

## Status

Open — PP > 1 for LLM in non-colocated MiMo is broken during training forward pass.

## Symptoms

```
TypeError: embedding(): argument 'indices' (position 2) must be Tensor, not NoneType
```

Crash occurs on the first training step, on non-first pipeline-stage ranks.

## Stack trace

```
train_mimo() → train_step_mimo()
  → forward_backward_pipelining_without_interleaving()
    → forward_step()
      → forward_step_func(data_iterator, model)
        → model(**data_batch)
          → ... → embedding(input, indices)  # indices is None
```

File: `3rdparty/Megatron-LM/megatron/core/pipeline_parallel/schedules.py`, line 2243
Called from: `src/megatron/bridge/training/train_mimo.py`, line 109

## Reproduction

```bash
# 8 GPUs: LLM with PP=2 DP=2, Vision with DP=4
export MIMO_LLM_TP=1 MIMO_LLM_PP=2 MIMO_LLM_DP=2 MIMO_LLM_OFFSET=0
export MIMO_VISION_TP=1 MIMO_VISION_PP=1 MIMO_VISION_DP=4 MIMO_VISION_OFFSET=4

torchrun --nproc_per_node=8 \
    tests/e2e/mimo/test_mimo_checkpoint_resume_e2e.py --phase save --ckpt-dir /tmp/pp_test
```

Fails during the first forward step (before any checkpoint is created).

## Configuration

- LLM module: PP=2, DP=2 → 4 ranks (ranks 0–3)
  - Pipeline stage 0: ranks 0, 1
  - Pipeline stage 1: ranks 2, 3
- Vision module: DP=4 → 4 ranks (ranks 4–7)

## Root cause analysis

In standard (non-MiMo) pipeline parallelism, non-first pipeline stages receive
their input activations from the previous stage via P2P communication — they do
not consume data from the data iterator directly. The pipeline schedule
(`forward_backward_pipelining_without_interleaving`) passes `None` as input to
non-first stages, and the model is expected to handle this by using the received
activations instead.

In MiMo's forward path (`mimo_step.py:forward_step`), the entire `data_batch`
dict is passed to `model(**data_batch)`. For non-first pipeline stages, the
`data_batch` fields (e.g., `input_ids`) are `None`, which propagates to the
embedding layer that expects a tensor.

The fix likely needs to be in one of:
1. **`mimo_step.py:forward_step`** — Guard against `None` inputs for non-first
   pipeline stages, similar to how standard GPT models handle this.
2. **MiMo's model forward** — The MiMo model should detect when it is a
   non-first pipeline stage and skip embedding, using received activations.
3. **Data batch construction** — Ensure non-first pipeline stages receive
   appropriate sentinel values instead of `None`.

## Observed on

- Job ID: 10027385 (CS-DFW cluster)
- Branch: `mimo/wip-phase4-training`
- Commit: `405ce567`
- Date: 2026-03-16

## Workaround

PP > 1 for LLM in MiMo is disabled in the e2e checkpoint resume test
(`pp2_llm_dp4_vision` config removed from `run_mimo_checkpoint_resume.sh`).
Non-PP configs (DP-only, TP-only, mixed TP+DP) all pass.
