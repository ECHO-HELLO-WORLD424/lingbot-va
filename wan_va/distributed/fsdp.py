# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc

import torch
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)


def apply_ac(model):
    """Apply activation checkpointing to the model."""
    for layer_id, transformer_block in enumerate(model.blocks):
        transformer_block = ptd_checkpoint_wrapper(
            transformer_block, preserve_rng_state=False
        )
        model.blocks[layer_id] = transformer_block


def shard_model(model, param_dtype=torch.bfloat16, reduce_dtype=torch.float32):
    # FSDP2 requires all parameters within a sharded group to share the same
    # dtype.  WanTransformer3DModel deliberately keeps certain sub-modules
    # (FP32LayerNorm weights, scale_shift_table, time_embedder, etc.) in
    # float32 via _keep_in_fp32_modules.  Those modules cast their *inputs* to
    # float32 at runtime, so storing the weights in bfloat16 is safe.  Cast
    # them here so every parameter in the model has a uniform dtype before
    # fully_shard is applied.
    for module in model.modules():
        for param in module.parameters(recurse=False):
            if param.dtype != param_dtype:
                param.data = param.data.to(param_dtype)

    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    fsdp_config = {"mp_policy": mp_policy, "reshard_after_forward": True}

    for block in model.blocks:
        fully_shard(block.attn1, **fsdp_config)
        fully_shard(block.attn2, **fsdp_config)
        fully_shard(block.ffn, **fsdp_config)
        fully_shard(block, **fsdp_config)

    fully_shard(model, **fsdp_config)
    return model


def free_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()
