# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""End-to-end homogeneous MIMO LLaVA + Whisper training test.

Exercises the standard pretrain() loop with MimoModelProvider in homogeneous
mode (mimo_parallelism_config=None). All modules (LLM + CLIP vision encoder
+ Whisper audio encoder) run on every rank together. The LLM uses TP=4, PP=1
across all GPUs.

Run:
    torchrun --nproc_per_node=4 tests/e2e/mimo/test_mimo_training_llava_homogeneous_audio.py
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mimo.submodules.audio import AudioModalitySubmodules
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.models.vision.clip_vit_model import CLIPViTModel
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig

from typing import Optional

# The whisper Megatron-native encoder lives next to this test file so the
# checked-in MIMO codebase doesn't need to ship it.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from whisper.whisper_layer_specs import get_whisper_layer_with_transformer_engine_spec
from whisper.whisper_model import WhisperEncoder


# ---------------------------------------------------------------------------
# Heterogeneous-compat RNG reset across modality submodule construction
# ---------------------------------------------------------------------------
# In the homogeneous layout every rank builds image_enc → image_proj →
# audio_enc → audio_proj → language. In the heterogeneous layout each rank
# only builds the module(s) in its role, so the audio_projector init samples
# the CPU RNG fresh from seed=42 (no image prefix). Without this patch the
# trainable audio projector starts with different weights between the two
# layouts and the loss curves diverge from step 1.
#
# Resetting the CPU RNG before each modality submodule (and before the
# language model) makes every module's init independent of construction
# order, matching the heterogeneous "one rank, one module" RNG state.


_HOMO_COMPAT_BASE_SEED = 42  # must match the seed argv-passed to main()


def _install_per_module_rng_reset_patch(base_seed: int = _HOMO_COMPAT_BASE_SEED) -> None:
    """Wrap MimoModel._initialize_submodules / _initialize_language_model so
    each module's init starts from a freshly-seeded RNG. This makes the
    homogeneous run's per-module init independent of construction order,
    matching the heterogeneous "one rank, one module" RNG state.

    Reseeds CPU + CUDA + CudaRNGStatesTracker before each modality submodule
    construction (and before the language model build) using the same
    seed/tp_rank that ``_set_random_seed`` originally used. Concretely:

      - ``torch.manual_seed(base_seed)`` resets CPU torch RNG.
      - ``model_parallel_cuda_manual_seed(base_seed, tp_rank, ...)`` resets
        CUDA RNG and rebuilds the tracker's named states from scratch
        (after a tracker.reset()), bypassing the "seed already exists" check.

    tp_rank is read from ``parallel_state`` (set up by the standard
    homogeneous provide_distributed_model path before MimoModel.__init__).
    """
    import logging

    import torch
    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.models.mimo.model.base import MimoModel
    from megatron.core.transformer.spec_utils import build_module

    logger = logging.getLogger(__name__)

    def _reseed():
        # CPU + numpy + python random — _set_random_seed sets all three.
        import random as _random

        import numpy as _np

        _random.seed(base_seed)
        _np.random.seed(base_seed)
        torch.manual_seed(base_seed)

        if torch.cuda.device_count() > 0:
            tp_rank = (
                parallel_state.get_tensor_model_parallel_rank()
                if parallel_state.model_parallel_is_initialized()
                else 0
            )
            # model_parallel_cuda_manual_seed checks for "seed already exists"
            # in the tracker and raises. Force a tracker.reset() by calling it
            # with force_reset_rng=True.
            tensor_parallel.model_parallel_cuda_manual_seed(
                base_seed,
                tp_rank=tp_rank,
                ep_rank=0,
                etp_rank=0,
                force_reset_rng=True,
            )

    def _patched_init_submodules(self) -> None:
        logger.warning("[homo-compat] _patched_init_submodules firing")
        for modality_name, submodule_spec in self.mimo_config.modality_submodules_spec.items():
            if self.role is not None and modality_name not in self.role.modules:
                continue

            _reseed()
            logger.warning(f"[homo-compat] reseeded before {modality_name} submodule")

            is_first_stage = True
            is_last_stage = True
            if self.role is not None and modality_name in self.role.modules:
                stage_info = self.role.modules[modality_name]
                is_first_stage = stage_info.is_first_stage
                is_last_stage = stage_info.is_last_stage

            submodule_class = submodule_spec.module
            submodule = submodule_class.from_spec(
                submodule_spec,
                is_first_stage=is_first_stage,
                is_last_stage=is_last_stage,
            )
            self.modality_submodules[modality_name] = submodule

    def _patched_init_language_model(self) -> None:
        if self.role is not None and not self.role.has_language_module:
            self.language_model = None
            return

        _reseed()
        logger.warning("[homo-compat] reseeded before language_model")

        self.language_model = build_module(self.mimo_config.language_model_spec)

    MimoModel._initialize_submodules = _patched_init_submodules
    MimoModel._initialize_language_model = _patched_init_language_model


_install_per_module_rng_reset_patch()


class CLIPViTNoCLS(CLIPViTModel):
    """CLIPViTModel that drops the CLS token to match HF LLaVA (mm_vision_select_feature='patch')."""

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = super().forward(x, attention_mask=attention_mask)
        return x[:, self.class_token_len :, :]


# ---------------------------------------------------------------------------
# LLaVA model configs (Vicuna-7B + CLIP ViT-L/14 + MLP projection)
# ---------------------------------------------------------------------------

IMAGE_SPECIAL_TOKEN_ID = 32000
AUDIO_SPECIAL_TOKEN_ID = 32002
VOCAB_SIZE = 32256
CLIP_OUTPUT_DIM = 1024  # CLIP ViT-L/14 hidden size
WHISPER_OUTPUT_DIM = 512  # Whisper-base hidden size
MAX_SEQ_LENGTH = 4096
_IMG_SIZE = 336
_PATCH_DIM = 14
# CLIP ViT-L/14 @ 336×336: (336/14)^2 = 576 patches (CLS token dropped per HF LLaVA)
_ENCODER_SEQ_LEN = 576
# Whisper-base: 30s padded audio → 3000 mel frames → 1500 encoder output tokens
_AUDIO_ENCODER_SEQ_LEN = 1500
_AUDIO_NUM_MEL_BINS = 80
_AUDIO_MAX_SOURCE_POSITIONS = 1500


def _make_vision_config(deterministic: bool = False) -> TransformerConfig:
    """CLIP ViT-L/14 vision encoder config (23 layers = penultimate layer output per HF LLaVA)."""
    cfg = TransformerConfig(
        num_layers=23,
        hidden_size=1024,
        ffn_hidden_size=4096,
        num_attention_heads=16,
        use_cpu_initialization=True,
        pipeline_dtype=torch.float32 if deterministic else torch.bfloat16,
        bf16=not deterministic,
        variable_seq_lengths=True,
        moe_token_dispatcher_type="alltoall",
    )
    cfg.add_bias_linear = True
    cfg.add_qkv_bias = True
    cfg.hidden_dropout = 0.0
    cfg.attention_dropout = 0.0
    cfg.gated_linear_unit = False
    cfg.layernorm_zero_centered_gamma = False
    cfg.apply_query_key_layer_scaling = False
    cfg.bias_activation_fusion = False
    cfg.bias_dropout_fusion = False
    cfg.attention_softmax_in_fp32 = True
    cfg.normalization = "LayerNorm"
    cfg.apply_rope_fusion = False
    # CLIP uses "quick_gelu", not standard gelu
    cfg.activation_func = lambda x: x * torch.sigmoid(1.702 * x)
    cfg.calculate_per_token_loss = True

    if deterministic:
        cfg.attention_backend = AttnBackend.unfused
        cfg.deterministic_mode = True
        cfg.recompute_granularity = "full"
        cfg.recompute_method = "uniform"
        cfg.recompute_num_layers = 1

    return cfg


def _make_audio_config(deterministic: bool = False) -> TransformerConfig:
    """Whisper-base audio encoder config (6 encoder layers, d_model=512)."""
    cfg = TransformerConfig(
        num_layers=6,
        hidden_size=512,
        ffn_hidden_size=2048,
        num_attention_heads=8,
        use_cpu_initialization=True,
        pipeline_dtype=torch.float32 if deterministic else torch.bfloat16,
        bf16=not deterministic,
        variable_seq_lengths=True,
        moe_token_dispatcher_type="alltoall",
    )
    cfg.add_bias_linear = True
    cfg.add_qkv_bias = True
    cfg.hidden_dropout = 0.0
    cfg.attention_dropout = 0.0
    cfg.gated_linear_unit = False
    cfg.layernorm_zero_centered_gamma = False
    cfg.apply_query_key_layer_scaling = False
    cfg.bias_activation_fusion = False
    cfg.bias_dropout_fusion = False
    cfg.attention_softmax_in_fp32 = True
    cfg.normalization = "LayerNorm"
    cfg.apply_rope_fusion = False
    cfg.calculate_per_token_loss = True

    if deterministic:
        cfg.attention_backend = AttnBackend.unfused
        cfg.deterministic_mode = True
        cfg.recompute_granularity = "full"
        cfg.recompute_method = "uniform"
        cfg.recompute_num_layers = 1

    return cfg


def _make_language_config(deterministic: bool = False) -> TransformerConfig:
    """Vicuna-7B language model config (same arch as Llama-7B)."""
    cfg = TransformerConfig(
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        use_cpu_initialization=True,
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=1,
    )

    cfg.ffn_hidden_size = 11008
    cfg.activation_func = torch.nn.functional.silu
    cfg.gated_linear_unit = True

    cfg.normalization = "RMSNorm"
    cfg.rms_norm_eps = 1e-5

    cfg.position_embedding_type = "rope"
    cfg.rotary_base = 10000
    cfg.rotary_percent = 1.0

    cfg.seq_length = MAX_SEQ_LENGTH
    cfg.max_position_embeddings = MAX_SEQ_LENGTH

    cfg.attention_dropout = 0.0
    cfg.hidden_dropout = 0.0

    cfg.num_query_groups = 32
    cfg.add_bias_linear = False
    cfg.untie_embeddings_and_output_weights = True

    cfg.bias_activation_fusion = True
    cfg.masked_softmax_fusion = True
    cfg.persist_layer_norm = True
    cfg.bias_dropout_fusion = True
    cfg.apply_rope_fusion = True

    cfg.pipeline_dtype = torch.float32 if deterministic else torch.bfloat16
    cfg.bf16 = not deterministic
    cfg.cross_entropy_loss_fusion = not deterministic
    cfg.variable_seq_lengths = True
    cfg.calculate_per_token_loss = True

    if deterministic:
        cfg.attention_backend = AttnBackend.unfused
        cfg.deterministic_mode = True
        cfg.recompute_granularity = "full"
        cfg.recompute_method = "uniform"
        cfg.recompute_num_layers = 1

    return cfg


def _make_projection_config(hidden_size: int = 4096, deterministic: bool = False) -> TransformerConfig:
    """Vision→language projection MLP config."""
    cfg = TransformerConfig(num_layers=1, hidden_size=hidden_size, num_attention_heads=1, use_cpu_initialization=True)
    cfg.ffn_hidden_size = 4096
    cfg.bias_activation_fusion = True
    cfg.add_bias_linear = True
    cfg.activation_func = torch.nn.functional.gelu
    cfg.calculate_per_token_loss = True
    cfg.pipeline_dtype = torch.float32 if deterministic else torch.bfloat16
    cfg.bf16 = not deterministic

    if deterministic:
        cfg.deterministic_mode = True

    return cfg


def _build_model_specs(deterministic: bool = False, with_audio: bool = True):
    """Return (language_model_spec, modality_submodules_spec, special_token_ids)."""
    vision_config = _make_vision_config(deterministic=deterministic)
    language_config = _make_language_config(deterministic=deterministic)
    projection_config = _make_projection_config(hidden_size=language_config.hidden_size, deterministic=deterministic)

    # CLIP ViT-L/14 encoder
    vision_encoder = ModuleSpec(
        module=CLIPViTNoCLS,
        params={
            "transformer_config": vision_config,
            "transformer_layer_spec": get_vit_layer_with_transformer_engine_spec(),
            "patch_dim": _PATCH_DIM,
            "img_h": _IMG_SIZE,
            "img_w": _IMG_SIZE,
        },
    )

    # Vision→language projection MLP
    vision_projection = ModuleSpec(
        module=MultimodalProjector,
        params={
            "config": projection_config,
            "submodules": MLPSubmodules(
                linear_fc1=TEColumnParallelLinear,
                linear_fc2=TERowParallelLinear,
            ),
            "projector_type": "mlp",
            "input_size": CLIP_OUTPUT_DIM,
        },
    )

    vision_submodule_spec = ModuleSpec(
        module=VisionModalitySubmodules,
        params={},
        submodules={
            "encoders": {"clip": vision_encoder},
            "input_projections": [vision_projection],
        },
    )

    language_model_spec = ModuleSpec(
        module=GPTModel,
        params={
            "config": language_config,
            "transformer_layer_spec": get_gpt_layer_with_transformer_engine_spec(),
            "vocab_size": VOCAB_SIZE,
            "max_sequence_length": MAX_SEQ_LENGTH,
            "position_embedding_type": "rope",
        },
    )

    modality_submodules_spec = {"images": vision_submodule_spec}
    special_token_ids = {"images": IMAGE_SPECIAL_TOKEN_ID}

    if with_audio:
        audio_config = _make_audio_config(deterministic=deterministic)
        audio_projection_config = _make_projection_config(
            hidden_size=language_config.hidden_size, deterministic=deterministic
        )

        # Megatron-native Whisper encoder (TP-shardable)
        audio_encoder = ModuleSpec(
            module=WhisperEncoder,
            params={
                "transformer_config": audio_config,
                "transformer_layer_spec": get_whisper_layer_with_transformer_engine_spec(),
                "num_mel_bins": _AUDIO_NUM_MEL_BINS,
                "max_source_positions": _AUDIO_MAX_SOURCE_POSITIONS,
            },
        )

        # Audio→language projection MLP
        audio_projection = ModuleSpec(
            module=MultimodalProjector,
            params={
                "config": audio_projection_config,
                "submodules": MLPSubmodules(
                    linear_fc1=TEColumnParallelLinear,
                    linear_fc2=TERowParallelLinear,
                ),
                "projector_type": "mlp",
                "input_size": WHISPER_OUTPUT_DIM,
            },
        )

        audio_submodule_spec = ModuleSpec(
            module=AudioModalitySubmodules,
            params={},
            submodules={
                "encoders": {"whisper": audio_encoder},
                "input_projections": [audio_projection],
            },
        )

        modality_submodules_spec["audios"] = audio_submodule_spec
        special_token_ids["audios"] = AUDIO_SPECIAL_TOKEN_ID

    return language_model_spec, modality_submodules_spec, special_token_ids


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------

from megatron.bridge.data.mimo.dataset import MimoDataset
from megatron.bridge.data.mimo.hf_provider import HFMimoDatasetProvider
from megatron.bridge.training.config import DatasetBuildContext


def _llava_preprocess(example, dataset_root):
    """Convert LLaVA conversations format to plain text and resolve media paths.

    Emits the full conversation (human + gpt turns) as ``text`` so the LM
    conditions on the human prompt during training. Loss masking to the
    assistant-answer tokens is applied by ``_AnswerMaskedMimoDataset``,
    matching HF LLaVA's ``preprocess_plain`` and the Megatron-LM
    examples/mimo task encoders.
    """
    conversations = example.get("conversations", [])
    text_parts = [turn.get("value", "") for turn in conversations]
    example["text"] = " ".join(text_parts).replace("<image>", "").replace("<audio>", "").strip()
    # Resolve relative image paths to absolute paths
    if "image" in example and example["image"] and not os.path.isabs(example["image"]):
        example["image"] = os.path.join(dataset_root, example["image"])
    # Load audio from file path into a numpy array for WhisperProcessor
    if "audio" in example and example["audio"]:
        audio_val = example["audio"]
        if isinstance(audio_val, str):
            audio_path = audio_val if os.path.isabs(audio_val) else os.path.join(dataset_root, audio_val)
            import soundfile as sf

            audio_array, sr = sf.read(audio_path)
            if sr != 16000:
                raise ValueError(
                    f"Whisper expects 16 kHz audio but {audio_path} has sample rate {sr}. "
                    "Resample the dataset to 16 kHz before training."
                )
            example["audio"] = audio_array
        elif isinstance(audio_val, dict) and "array" in audio_val:
            # HuggingFace Audio feature format
            example["audio"] = audio_val["array"]
    return example


def _find_token_span(seq: torch.Tensor, pattern: torch.Tensor, start_idx: int = 0,
                     allow_first_mismatch: bool = False) -> tuple[int, int]:
    """Return (start, end) of the first occurrence of ``pattern`` in ``seq``.

    Mirrors ``_find_pattern_indices`` in Megatron-LM examples/mimo task encoders.
    ``allow_first_mismatch`` handles SentencePiece boundary differences when the
    answer is tokenized standalone vs. embedded in the full prompt.
    Returns (-1, -1) if not found.
    """
    n, p = seq.size(0), pattern.size(0)
    if p == 0 or p > n:
        return -1, -1
    for i in range(start_idx, n - p + 1):
        match = seq[i : i + p] == pattern
        if torch.all(match) or (allow_first_mismatch and torch.all(match[1:])):
            return i, i + p
    return -1, -1


class _AnswerMaskedMimoDataset(MimoDataset):
    """MimoDataset variant that masks loss to assistant-answer tokens only.

    The base class sets ``loss_mask=1`` for every non-placeholder, non-pad
    position, which trains the LM on the human instruction as well as the
    caption. For LLaVA-Pretrain loss must be computed on the assistant ("gpt")
    turn only — the HF LLaVA ``preprocess_plain`` contract, also implemented
    by the Megatron-LM examples/mimo task encoders.

    Works identically for vision-only and audio-augmented variants: the audio
    placeholders (if any) fall outside the answer span and remain ``-100`` /
    ``loss_mask=0``.
    """

    def __getitem__(self, idx):
        item = super().__getitem__(idx)

        raw = self.examples[idx]  # HF datasets return a fresh dict per access
        answers = [
            t.get("value", "") for t in raw.get("conversations", []) if t.get("from") == "gpt"
        ]
        if not any(a.strip() for a in answers):
            return item

        input_ids = item["input_ids"]
        labels = torch.full_like(input_ids, -100)
        search_idx = 0
        for ans in answers:
            ans = ans.replace("<image>", "").replace("<audio>", "").strip()
            if not ans:
                continue
            ans_ids = self.tokenizer(
                ans, add_special_tokens=False, return_tensors="pt"
            )["input_ids"].squeeze(0)
            if ans_ids.numel() == 0:
                continue
            s, e = _find_token_span(
                input_ids, ans_ids, start_idx=search_idx, allow_first_mismatch=True
            )
            if s < 0:
                # Answer span not found (e.g. truncated); skip this answer.
                continue
            # labels[i] predicts input_ids[i+1]; answer tokens at input_ids[s:e]
            # are predicted at positions [s-1, e-1).
            lo, hi = max(0, s - 1), e - 1
            if hi > lo:
                labels[lo:hi] = input_ids[lo + 1 : hi + 1]
            search_idx = e

        item["labels"] = labels
        item["loss_mask"] = (labels != -100).to(item["loss_mask"].dtype)
        return item


def _pad_1d(tensors, pad_value=0):
    """Pad a list of 1-D tensors to the longest length and stack."""
    max_len = max(t.size(0) for t in tensors)
    padded = []
    for t in tensors:
        if t.size(0) < max_len:
            padding = torch.full((max_len - t.size(0),), pad_value, dtype=t.dtype)
            padded.append(torch.cat([t, padding]))
        else:
            padded.append(t)
    return torch.stack(padded)


def _mimo_collate_with_loss_masking(batch, modality_names, pad_token_id=0):
    """Collate per-sample dicts into a padded batch.

    ``_AnswerMaskedMimoDataset`` already produces shifted ``labels`` (``-100``
    outside the assistant-answer span) and the matching ``loss_mask``, so the
    collate just pads and stacks — it does not re-shift or re-mask.
    """
    import warnings

    if not batch:
        return {}

    lengths = [item["input_ids"].size(0) for item in batch]
    variable_lengths = len(set(lengths)) > 1

    if variable_lengths:
        input_ids = _pad_1d([item["input_ids"] for item in batch], pad_value=pad_token_id)
        attention_mask = _pad_1d([item["attention_mask"] for item in batch], pad_value=0)
        position_ids = _pad_1d([item["position_ids"] for item in batch], pad_value=0)
        labels = _pad_1d([item["labels"] for item in batch], pad_value=-100)
        loss_mask = _pad_1d([item["loss_mask"] for item in batch], pad_value=0)
    else:
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        position_ids = torch.stack([item["position_ids"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        loss_mask = torch.stack([item["loss_mask"] for item in batch])

    # Collate modality inputs
    modality_inputs = {}
    for modality_name in modality_names:
        modality_batch_items = [item.get("modality_inputs", {}).get(modality_name, {}) for item in batch]
        if not any(modality_batch_items):
            continue
        first_non_empty = next((item for item in modality_batch_items if item), {})
        if not first_non_empty:
            continue
        modality_inputs[modality_name] = {}
        for key in first_non_empty.keys():
            values = [item[key] for item in modality_batch_items if key in item]
            if values and isinstance(values[0], torch.Tensor):
                try:
                    modality_inputs[modality_name][key] = torch.stack(values)
                except RuntimeError:
                    warnings.warn(
                        f"Cannot stack tensors for '{modality_name}.{key}' - shapes differ. Keeping as list.",
                        stacklevel=2,
                    )
                    modality_inputs[modality_name][key] = values
            elif values:
                modality_inputs[modality_name][key] = values

    return {
        "input_ids": input_ids,
        "labels": labels,
        "loss_mask": loss_mask,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "modality_inputs": modality_inputs,
    }


class HomogeneousHFMimoDatasetProvider(HFMimoDatasetProvider):
    """HFMimoDatasetProvider that builds ``_AnswerMaskedMimoDataset`` instances
    and attaches the custom padding collate_fn.

    ``_AnswerMaskedMimoDataset`` produces per-sample shifted labels with loss
    restricted to the assistant-answer span (LLaVA-Pretrain contract). The
    collate just pads and stacks.
    """

    def _build_split_dataset(self, split, target_samples, processors, tokenizer):
        if target_samples <= 0:
            return None
        hf_dataset = self._load_hf_dataset(split)
        if hf_dataset is None:
            return None
        return _AnswerMaskedMimoDataset(
            examples=hf_dataset,
            processors=processors,
            tokenizer=tokenizer,
            seq_length=self.seq_length,
            special_token_ids=self.special_token_ids,
            encoder_seq_lengths=self.encoder_seq_lengths,
            modality_columns=self.modality_columns,
            text_column=self.text_column,
            max_samples=target_samples,
            preprocess_fn=self.preprocess_fn,
        )

    def build_datasets(self, context: DatasetBuildContext):
        train_ds, valid_ds, test_ds = super().build_datasets(context)
        collate_fn = partial(
            _mimo_collate_with_loss_masking,
            modality_names=list(self.modality_columns.keys()),
        )
        for ds in (train_ds, valid_ds, test_ds):
            if ds is not None:
                ds.collate_fn = collate_fn
        return train_ds, valid_ds, test_ds


def _build_hf_data_provider(
    dataset_root: str,
    audio_column: str | None = None,
    hf_data_files: str = "blip_laion_cc_sbu_558k.json",
) -> HomogeneousHFMimoDatasetProvider:
    """Build an HFMimoDatasetProvider for LLaVA-Pretrain with optional audio."""
    processor_paths = {"images": "openai/clip-vit-large-patch14-336"}
    special_token_ids = {"images": IMAGE_SPECIAL_TOKEN_ID}
    encoder_seq_lengths = {"images": _ENCODER_SEQ_LEN}
    modality_columns = {"images": "image"}

    if audio_column:
        processor_paths["audios"] = "openai/whisper-base"
        special_token_ids["audios"] = AUDIO_SPECIAL_TOKEN_ID
        encoder_seq_lengths["audios"] = _AUDIO_ENCODER_SEQ_LEN
        modality_columns["audios"] = audio_column

    provider = HomogeneousHFMimoDatasetProvider(
        seq_length=MAX_SEQ_LENGTH,
        hf_dataset_path=dataset_root,
        hf_data_files=hf_data_files,
        hf_tokenizer_path="llava-hf/llava-1.5-7b-hf",
        processor_paths=processor_paths,
        special_token_ids=special_token_ids,
        encoder_seq_lengths=encoder_seq_lengths,
        modality_columns=modality_columns,
        text_column="text",
        train_split="train",
        preprocess_fn=lambda example: _llava_preprocess(example, dataset_root),
    )
    provider.drop_last = True

    return provider


# ---------------------------------------------------------------------------
# Forward step function (homogeneous path)
# ---------------------------------------------------------------------------

from megatron.bridge.training.mimo_step import loss_func as mimo_loss_func


def forward_step_func(data_iterator, model):
    """Forward step for homogeneous MIMO via pretrain()."""
    batch = next(data_iterator)

    input_ids = batch["input_ids"].cuda(non_blocking=True)
    labels = batch["labels"].cuda(non_blocking=True)
    position_ids = batch["position_ids"].cuda(non_blocking=True)

    # Use the loss_mask produced by the collate (accounts for padding, special tokens, label shifting)
    batch_loss_mask = batch.get("loss_mask")
    if batch_loss_mask is not None:
        batch_loss_mask = batch_loss_mask.cuda(non_blocking=True)

    pipeline_dtype = getattr(model, "module", model).language_model.config.pipeline_dtype

    modality_inputs: dict = {}
    raw_modality_inputs = batch.get("modality_inputs") or {}

    if "images" in raw_modality_inputs:
        pv = raw_modality_inputs["images"].get("pixel_values")
        if pv is not None:
            modality_inputs["images"] = {
                "clip": {"x": pv.cuda(non_blocking=True).to(pipeline_dtype)}
            }

    if "audios" in raw_modality_inputs:
        af = raw_modality_inputs["audios"].get("input_features")
        if af is not None:
            af = af.cuda(non_blocking=True).to(pipeline_dtype)
            audio_kwargs = {"input_features": af}

            # Compute per-sample valid encoder output lengths.
            # WhisperFeatureExtractor pads mel spectrograms with zeros;
            # real frames always have non-zero energy in at least one bin.
            frame_energy = af.abs().sum(dim=-2)  # [B, mel_frames], sum over mel bins
            valid_frames = (frame_energy > 0).sum(dim=-1)  # [B]
            # Conv2 uses stride=2: output_len = (input_len - 1) // 2 + 1
            seq_lengths = ((valid_frames - 1) // 2 + 1).clamp(min=0).long()
            audio_kwargs["seq_lengths"] = seq_lengths

            # Replace excess audio placeholder tokens in input_ids so that
            # align_embeddings_by_token_positions sees the correct count.
            for i in range(input_ids.size(0)):
                positions = (input_ids[i] == AUDIO_SPECIAL_TOKEN_ID).nonzero(as_tuple=True)[0]
                n_valid = seq_lengths[i].item()
                if n_valid < len(positions):
                    input_ids[i, positions[n_valid:]] = 0  # replace with pad token

            modality_inputs["audios"] = {"whisper": audio_kwargs}

    output = model(
        input_ids=input_ids,
        position_ids=position_ids,
        labels=labels,
        attention_mask=None,
        modality_inputs=modality_inputs,
    )

    output_tensor, model_loss_mask = output

    # Prefer the batch loss_mask (from collate) over the model's
    loss_mask = batch_loss_mask if batch_loss_mask is not None else model_loss_mask
    if loss_mask is None:
        loss_mask = torch.ones_like(output_tensor)

    return output_tensor, partial(mimo_loss_func, loss_mask)


# ---------------------------------------------------------------------------
# Config assembly
# ---------------------------------------------------------------------------

from megatron.bridge.models.mimo.mimo_provider import MimoModelProvider
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    LoggerConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
    ValidationConfig,
)
from megatron.bridge.training.tokenizers.config import TokenizerConfig


def _build_config(
    mimo_provider: MimoModelProvider,
    data_provider: HomogeneousHFMimoDatasetProvider,
    opt_config: OptimizerConfig,
    micro_batch_size: int = 1,
    global_batch_size: int = 1,
    train_iters: int = 2,
    log_interval: int = 1,
    wandb_project: str | None = None,
    wandb_exp_name: str | None = None,
    wandb_entity: str | None = None,
    wandb_save_dir: str | None = None,
    lr_warmup_iters: int = 0,
    seed: int = 42,
    deterministic: bool = False,
) -> ConfigContainer:
    train_cfg = TrainingConfig(
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        train_iters=train_iters,
    )
    train_cfg.log_interval = log_interval

    logger_cfg = LoggerConfig()
    logger_cfg.log_timers_to_tensorboard = True
    logger_cfg.log_interval = log_interval
    logger_cfg.wandb_project = wandb_project
    logger_cfg.wandb_exp_name = wandb_exp_name
    logger_cfg.wandb_entity = wandb_entity
    logger_cfg.wandb_save_dir = wandb_save_dir
    logger_cfg.tensorboard_dir = os.path.join(wandb_save_dir or "/tmp/tb_logs", "tb_logs") if wandb_project else None

    scheduler_cfg = SchedulerConfig(
        lr_decay_style="cosine",
        lr_warmup_iters=lr_warmup_iters,
        lr_warmup_init=opt_config.min_lr,
        start_weight_decay=opt_config.weight_decay,
        end_weight_decay=opt_config.weight_decay,
    )

    ddp_cfg = DistributedDataParallelConfig(
        grad_reduce_in_fp32=deterministic,
        overlap_grad_reduce=False,
        check_for_nan_in_grad=False,
        use_distributed_optimizer=True,
    )

    cfg = ConfigContainer(
        train=train_cfg,
        model=mimo_provider,
        optimizer=opt_config,
        scheduler=scheduler_cfg,
        dataset=data_provider,
        ddp=ddp_cfg,
        logger=logger_cfg,
        tokenizer=TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model="llava-hf/llava-1.5-7b-hf",
        ),
        checkpoint=CheckpointConfig(),
        validation=ValidationConfig(eval_interval=2, eval_iters=0),
    )
    cfg.rng.seed = seed
    cfg.optimizer.use_distributed_optimizer = True
    # data_parallel_size=1 because the sampler does not shard by DP.
    # All data-loading ranks receive identical global micro-batches;
    # per-module DP sub-sharding is handled by slice_batch_for_mimo in the
    # forward step.  num_microbatches = global_batch_size / micro_batch_size.
    cfg.data_parallel_size = 1
    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

from megatron.bridge.training.pretrain import pretrain


# ---------------------------------------------------------------------------
# Per-submodule checkpoint loading
# ---------------------------------------------------------------------------


def _get_pp_layer_offset(module: torch.nn.Module) -> int:
    """Return the global layer offset for a module's PP stage.

    Inspects the first transformer layer's ``layer_number`` attribute (1-based
    global index set by Megatron) and compares it with the local ModuleList
    index to derive the offset.  Returns 0 when PP=1 or for non-transformer
    modules (e.g. the vision encoder).
    """
    decoder = getattr(module, "decoder", None)
    if decoder is None:
        return 0
    layers = getattr(decoder, "layers", None)
    if not layers or len(layers) == 0:
        return 0
    first_layer = layers[0]
    global_layer_number = getattr(first_layer, "layer_number", None)
    if global_layer_number is None:
        return 0
    # layer_number is 1-based; local index is 0-based
    return global_layer_number - 1


def _remap_checkpoint_for_pp(
    state_dict: dict[str, torch.Tensor],
    module: torch.nn.Module,
    layer_offset: int,
) -> dict[str, torch.Tensor]:
    """Remap globally-numbered checkpoint layer keys to local PP stage indices.

    The HF→Megatron converters produce globally-numbered keys
    (``decoder.layers.0`` … ``decoder.layers.31``), but each PP stage stores
    layers in a ``nn.ModuleList`` with 0-based local indices.  For PP stage 1
    with offset=16, checkpoint key ``decoder.layers.16`` must become
    ``decoder.layers.0`` so it matches the module's state dict.

    Non-layer keys (embedding, output_layer, final_layernorm) are passed
    through unchanged, then filtered to keys the module actually owns.
    """
    import re

    module_keys = set(module.state_dict().keys())
    remapped = {}

    for key, value in state_dict.items():
        m = re.match(r"^(decoder\.layers\.)(\d+)(\..*)", key)
        if m:
            global_idx = int(m.group(2))
            local_idx = global_idx - layer_offset
            if local_idx < 0:
                continue  # belongs to an earlier PP stage
            new_key = f"{m.group(1)}{local_idx}{m.group(3)}"
            if new_key in module_keys:
                remapped[new_key] = value
        else:
            # Non-layer key (embedding, output_layer, final_layernorm, etc.)
            if key in module_keys:
                remapped[key] = value

    return remapped


def _load_tp_rank_weights(
    module: torch.nn.Module,
    ckpt_dir: str,
    tp_rank: int,
    label: str,
) -> None:
    """Load per-TP-rank ``.pt`` weights produced by the HF→Megatron converters.

    Both ``convert_hf_clip_to_megatron.py`` and ``convert_hf_llama_to_megatron.py``
    write the same layout::

        {ckpt_dir}/tp_rank_{NN}/model_weights.pt   →  {"model": {key: tensor}}

    When pipeline parallelism (PP) > 1, checkpoint layer keys are globally
    numbered but the module uses local 0-based indices.  We remap
    ``decoder.layers.<global_idx>`` → ``decoder.layers.<local_idx>`` using
    the PP stage's layer offset so each stage loads the correct layer weights.

    After loading, a spot-check compares up to 5 parameter tensors against the
    file to verify the weights actually landed in the module.
    """
    ckpt_file = os.path.join(ckpt_dir, f"tp_rank_{tp_rank:02d}", "model_weights.pt")
    if not os.path.exists(ckpt_file):
        raise FileNotFoundError(f"[{label}] Checkpoint not found: {ckpt_file}")

    saved = torch.load(ckpt_file, map_location="cpu", weights_only=True)
    state_dict = {k: v for k, v in saved["model"].items() if v is not None}

    # With pipeline parallelism (PP > 1), checkpoint layer keys are globally
    # numbered (decoder.layers.0 … decoder.layers.31) but each PP stage's
    # nn.ModuleList uses local 0-based indices.  Remap before loading.
    layer_offset = _get_pp_layer_offset(module)
    state_dict = _remap_checkpoint_for_pp(state_dict, module, layer_offset)
    if layer_offset > 0:
        print(f"[{label}] PP layer offset={layer_offset}, remapped checkpoint keys to local indices")

    incompat = module.load_state_dict(state_dict, strict=False)
    unexpected = [k for k in incompat.unexpected_keys if "_extra_state" not in k]
    missing = [k for k in incompat.missing_keys if "_extra_state" not in k]
    if unexpected or missing:
        raise RuntimeError(
            f"[{label}] load_state_dict mismatch.\n  Missing:    {missing}\n  Unexpected: {unexpected}"
        )

    # Spot-check: re-read module state and compare against checkpoint tensors
    model_sd = module.state_dict()
    checked = 0
    for key, ref_tensor in state_dict.items():
        if key not in model_sd or ref_tensor is None:
            continue
        if not torch.equal(model_sd[key].float().cpu(), ref_tensor.float().cpu()):
            max_diff = (model_sd[key].float().cpu() - ref_tensor.float().cpu()).abs().max().item()
            raise RuntimeError(
                f"[{label}] Weight verification FAILED for '{key}': max abs diff = {max_diff}"
            )
        checked += 1
        if checked >= 5:
            break
    if checked == 0:
        raise RuntimeError(f"[{label}] Weight verification found 0 overlapping keys to check")
    print(f"[{label}] Loaded and verified from {ckpt_file} ({checked} keys spot-checked)")


def _make_checkpoint_loader_hook(
    language_model_ckpt: str | None = None,
    vision_encoder_ckpt: str | None = None,
    audio_encoder_ckpt: str | None = None,
):
    """Return a ``pre_wrap_hook`` that loads per-module checkpoints.

    In homogeneous MIMO every rank materialises all modules, so the language
    model, vision encoder and (optional) audio encoder are always present.
    The hook uses ``parallel_state`` to determine the TP rank (shared across
    all modules in homogeneous mode).

    Checkpoint dirs are expected to contain per-TP-rank ``.pt`` files
    produced by the HF→Megatron converters.
    """

    def _hook(model_list):
        from megatron.core import parallel_state

        model = model_list[0]
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        tp_size = parallel_state.get_tensor_model_parallel_world_size()

        if language_model_ckpt and model.language_model is not None:
            _load_tp_rank_weights(
                model.language_model,
                language_model_ckpt,
                tp_rank,
                label=f"LLM tp_rank={tp_rank}/{tp_size}",
            )

        if vision_encoder_ckpt and "images" in model.modality_submodules:
            images_sub = model.modality_submodules["images"]
            encoder = getattr(images_sub.encoders, "clip", None) if hasattr(images_sub, "encoders") else None
            if encoder is not None:
                _load_tp_rank_weights(
                    encoder,
                    vision_encoder_ckpt,
                    tp_rank,
                    label=f"CLIP tp_rank={tp_rank}/{tp_size}",
                )

        if audio_encoder_ckpt and "audios" in model.modality_submodules:
            audios_sub = model.modality_submodules["audios"]
            encoder = getattr(audios_sub.encoders, "whisper", None) if hasattr(audios_sub, "encoders") else None
            if encoder is not None:
                _load_tp_rank_weights(
                    encoder,
                    audio_encoder_ckpt,
                    tp_rank,
                    label=f"Whisper tp_rank={tp_rank}/{tp_size}",
                )

        return model_list

    return _hook


_rank_log_file = None


def _log(msg):
    """Write with rank prefix to per-rank log file and flush."""
    global _rank_log_file
    rank = dist.get_rank() if dist.is_initialized() else "?"
    line = f"[Rank {rank}] {msg}\n"
    if _rank_log_file:
        _rank_log_file.write(line)
        _rank_log_file.flush()
    print(line, end="", flush=True)


def _str2bool(v):
    """Parse boolean values from command line arguments."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{v}'")


def parse_args():
    parser = argparse.ArgumentParser(description="Homogeneous MIMO LLaVA + Whisper training")
    parser.add_argument("--micro-batch-size", type=int, default=1, help="Micro batch size per GPU")
    parser.add_argument("--global-batch-size", type=int, default=1, help="Global batch size across all GPUs")
    parser.add_argument("--train-iters", type=int, default=2, help="Number of training iterations")
    parser.add_argument("--min-lr", type=float, default=2.0e-5)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--checkpoint-interval", type=int, default=None, help="Checkpoint save interval (iterations)")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Checkpoint output directory")
    parser.add_argument("--load-checkpoint", type=str, default=None, help="Checkpoint directory to resume from")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--wandb-project", type=str, default="Megatron-Bridge-MIMO", help="W&B project name")
    parser.add_argument("--wandb-exp-name", type=str, default="mimo-llava-audio-homo-e2e-test", help="W&B experiment name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity")
    parser.add_argument("--wandb-save-dir", type=str, default="/tmp/wandb", help="W&B save directory")
    parser.add_argument(
        "--lr-warmup-iters", type=int, default=20, help="Number of iterations to linearly warmup learning rate"
    )
    parser.add_argument("--dataset-root", type=str, required=True, help="Root directory of the LLaVA-Pretrain dataset")
    parser.add_argument(
        "--hf-data-files",
        type=str,
        default="blip_laion_cc_sbu_558k.json",
        help="JSON file under --dataset-root to load (e.g. the audio-augmented variant).",
    )
    parser.add_argument(
        "--audio-column",
        type=str,
        default=None,
        help="Dataset column name for audio data (e.g. 'audio'). Enables the audio encoder when set.",
    )
    parser.add_argument(
        "--vision-encoder-checkpoint",
        type=str,
        default=None,
        help="Path to pre-converted CLIP ViT checkpoint (TP-sharded, with tp_rank_XX/model_weights.pt)",
    )
    parser.add_argument(
        "--language-model-checkpoint",
        type=str,
        default=None,
        help="Path to pre-converted LLM checkpoint (TP-sharded, with tp_rank_XX/model_weights.pt)",
    )
    parser.add_argument(
        "--audio-encoder-checkpoint",
        type=str,
        default=None,
        help="Path to pre-converted Whisper checkpoint (TP-sharded, with tp_rank_XX/model_weights.pt)",
    )
    parser.add_argument("--freeze-vision", type=_str2bool, default=True, help="Freeze the vision encoder (default: True)")
    parser.add_argument("--freeze-llm", type=_str2bool, default=True, help="Freeze the language model (default: True)")
    parser.add_argument("--freeze-projector", type=_str2bool, default=False, help="Freeze the vision projector (default: False)")
    parser.add_argument("--freeze-audio", type=_str2bool, default=True, help="Freeze the audio encoder (default: True)")
    parser.add_argument("--freeze-audio-projector", type=_str2bool, default=False, help="Freeze the audio projector (default: False)")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=False,
        help="Enable deterministic mode: FP32 precision, unfused attention, disabled CE-loss fusion, "
        "full activation recompute, deterministic torch/cuDNN/NCCL/TE algorithms (slower, more reproducible).",
    )
    return parser.parse_args()


def main():
    global _rank_log_file

    args = parse_args()
    with_audio = bool(args.audio_column)

    # 1. Initialize distributed first so we know rank
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    # Seed all RNGs for reproducible weight initialization
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Open per-rank log file
    log_dir = "/tmp/claude-0/mimo_rank_logs"
    os.makedirs(log_dir, exist_ok=True)
    _rank_log_file = open(f"{log_dir}/rank_{rank}.log", "w")

    logging.basicConfig(
        level=logging.INFO,
        format=f"[Rank {rank}] %(name)s: %(message)s",
        handlers=[logging.FileHandler(f"{log_dir}/rank_{rank}_full.log", mode="w"), logging.StreamHandler(sys.stderr)],
        force=True,
    )

    _log(f"distributed initialized (world_size={dist.get_world_size()})")

    # 2. Build model provider
    _log(f"building model specs (with_audio={with_audio})")
    language_model_spec, modality_submodules_spec, special_token_ids = _build_model_specs(
        deterministic=args.deterministic, with_audio=with_audio
    )

    topology = {"images": ["llm"], "llm": []}
    freeze_modality_encoders = {"images": args.freeze_vision}
    freeze_modality_projections = {"images": args.freeze_projector}
    if with_audio:
        topology["audios"] = ["llm"]
        freeze_modality_encoders["audios"] = args.freeze_audio
        freeze_modality_projections["audios"] = args.freeze_audio_projector

    mimo_provider = MimoModelProvider(
        language_model_spec=language_model_spec,
        modality_submodules_spec=modality_submodules_spec,
        special_token_ids=special_token_ids,
        mimo_parallelism_config=None,  # Homogeneous mode
        topology=topology,
        use_cpu_initialization=True,
        bf16=not args.deterministic,
        vocab_size=VOCAB_SIZE,
        seq_length=MAX_SEQ_LENGTH,
        freeze_language_model=args.freeze_llm,
        freeze_modality_encoders=freeze_modality_encoders,
        freeze_modality_projections=freeze_modality_projections,
    )
    # Register per-module checkpoint loading hook (runs before DDP wrapping)
    if args.language_model_checkpoint or args.vision_encoder_checkpoint or args.audio_encoder_checkpoint:
        mimo_provider.register_pre_wrap_hook(
            _make_checkpoint_loader_hook(
                language_model_ckpt=args.language_model_checkpoint,
                vision_encoder_ckpt=args.vision_encoder_checkpoint,
                audio_encoder_ckpt=args.audio_encoder_checkpoint,
            )
        )
        _log(
            f"Registered checkpoint hooks: "
            f"LLM={args.language_model_checkpoint}, "
            f"vision={args.vision_encoder_checkpoint}, "
            f"audio={args.audio_encoder_checkpoint}"
        )

    # 3. Build data provider
    _log("building data provider")
    data_provider = _build_hf_data_provider(
        args.dataset_root,
        audio_column=args.audio_column,
        hf_data_files=args.hf_data_files,
    )

    # 4. Build optimizer config (Bridge OptimizerConfig for ConfigContainer)
    _log("building optimizer config")
    print_rank_0 = lambda msg: _log(msg) if dist.get_rank() == 0 else None
    print_rank_0(
        f"Optimizer config: lr={args.lr}, min_lr={args.min_lr}, weight_decay={args.weight_decay}, "
        f"adam_beta1={args.adam_beta1}, adam_beta2={args.adam_beta2}, clip_grad={args.clip_grad}"
    )
    opt_config = OptimizerConfig(
        lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        clip_grad=args.clip_grad,
        bf16=not args.deterministic,
    )

    # 5. Build config container
    _log("building config")
    cfg = _build_config(
        mimo_provider,
        data_provider,
        opt_config,
        micro_batch_size=args.micro_batch_size,
        global_batch_size=args.global_batch_size,
        train_iters=args.train_iters,
        log_interval=args.log_interval,
        wandb_project=args.wandb_project,
        wandb_exp_name=args.wandb_exp_name,
        wandb_entity=args.wandb_entity,
        wandb_save_dir=args.wandb_save_dir,
        lr_warmup_iters=args.lr_warmup_iters,
        seed=seed,
        deterministic=args.deterministic,
    )

    # Configure checkpointing from CLI args
    if args.checkpoint_interval is not None:
        cfg.checkpoint.save_interval = args.checkpoint_interval
    if args.checkpoint_dir is not None:
        cfg.checkpoint.save = args.checkpoint_dir
    if args.load_checkpoint is not None:
        cfg.checkpoint.load = args.load_checkpoint

    # Pre-cache the HF tokenizer on rank 0 before all ranks try simultaneously
    if rank == 0:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained("llava-hf/llava-1.5-7b-hf")
    dist.barrier()

    # 6. Run training
    _log("launching pretrain()")
    pretrain(cfg, forward_step_func)

    _log("PASSED")

    # 7. Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
