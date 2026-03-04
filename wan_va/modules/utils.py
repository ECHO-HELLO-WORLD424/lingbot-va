# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import logging
import time

import torch
from diffusers import AutoencoderKLWan
from transformers import (
    T5TokenizerFast,
    UMT5EncoderModel,
)

from .model import WanTransformer3DModel

logger = logging.getLogger(__name__)


def load_vae(
    vae_path,
    torch_dtype,
    torch_device,
):
    logger.info(
        f"load_vae: loading AutoencoderKLWan from {vae_path!r} (dtype={torch_dtype}, device={torch_device}) …"
    )
    t0 = time.perf_counter()
    vae = AutoencoderKLWan.from_pretrained(
        vae_path,
        torch_dtype=torch_dtype,
    )
    vae = vae.to(torch_device)
    elapsed = time.perf_counter() - t0
    n_params = sum(p.numel() for p in vae.parameters())
    logger.info(
        f"load_vae: done in {elapsed:.2f}s  params={n_params:,}  device={next(vae.parameters()).device}"
    )
    return vae


def load_text_encoder(
    text_encoder_path,
    torch_dtype,
    torch_device,
):
    logger.info(
        f"load_text_encoder: loading UMT5EncoderModel from {text_encoder_path!r} (dtype={torch_dtype}, device={torch_device}) …"
    )
    t0 = time.perf_counter()
    text_encoder = UMT5EncoderModel.from_pretrained(
        text_encoder_path,
        torch_dtype=torch_dtype,
    )
    text_encoder = text_encoder.to(torch_device)
    elapsed = time.perf_counter() - t0
    n_params = sum(p.numel() for p in text_encoder.parameters())
    logger.info(
        f"load_text_encoder: done in {elapsed:.2f}s  params={n_params:,}  device={next(text_encoder.parameters()).device}"
    )
    return text_encoder


def load_tokenizer(
    tokenizer_path,
):
    logger.info(f"load_tokenizer: loading T5TokenizerFast from {tokenizer_path!r} …")
    t0 = time.perf_counter()
    tokenizer = T5TokenizerFast.from_pretrained(
        tokenizer_path,
    )
    logger.info(f"load_tokenizer: done in {time.perf_counter() - t0:.2f}s")
    return tokenizer


def load_transformer(
    transformer_path,
    torch_dtype,
    torch_device,
):
    logger.info(
        f"load_transformer: loading WanTransformer3DModel from {transformer_path!r} (dtype={torch_dtype}, device={torch_device}) …"
    )
    t0 = time.perf_counter()
    model = WanTransformer3DModel.from_pretrained(
        transformer_path,
        torch_dtype=torch_dtype,
    )
    model = model.to(torch_device)
    elapsed = time.perf_counter() - t0
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"load_transformer: done in {elapsed:.2f}s  params={n_params:,}  device={next(model.parameters()).device}"
    )
    return model


def patchify(x, patch_size):
    if patch_size is None or patch_size == 1:
        return x
    batch_size, channels, frames, height, width = x.shape
    x = x.view(
        batch_size,
        channels,
        frames,
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
    )
    x = x.permute(0, 1, 6, 4, 2, 3, 5).contiguous()
    x = x.view(
        batch_size,
        channels * patch_size * patch_size,
        frames,
        height // patch_size,
        width // patch_size,
    )
    return x


class WanVAEStreamingWrapper:
    def __init__(self, vae_model):
        self.vae = vae_model
        self.encoder = vae_model.encoder
        self.quant_conv = vae_model.quant_conv

        if hasattr(self.vae, "_cached_conv_counts"):
            self.enc_conv_num = self.vae._cached_conv_counts["encoder"]
        else:
            count = 0
            for m in self.encoder.modules():
                if m.__class__.__name__ == "WanCausalConv3d":
                    count += 1
            self.enc_conv_num = count

        self.clear_cache()

    def clear_cache(self):
        self.feat_cache = [None] * self.enc_conv_num

    def encode_chunk(self, x_chunk):
        if (
            hasattr(self.vae.config, "patch_size")
            and self.vae.config.patch_size is not None
        ):
            x_chunk = patchify(x_chunk, self.vae.config.patch_size)
        feat_idx = [0]
        out = self.encoder(x_chunk, feat_cache=self.feat_cache, feat_idx=feat_idx)
        enc = self.quant_conv(out)
        return enc
