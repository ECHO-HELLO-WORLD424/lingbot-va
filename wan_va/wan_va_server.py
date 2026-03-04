# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import argparse
import os
import sys
import time
from contextlib import contextmanager
from functools import partial
from PIL import Image
from diffusers.video_processor import VideoProcessor
from diffusers.utils import export_to_video

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from einops import rearrange
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import VA_CONFIGS


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def _gpu_mem_str(device: torch.device) -> str:
    """Return a compact GPU memory summary string for *device*."""
    if not torch.cuda.is_available():
        return "CUDA N/A"
    alloc = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    total = torch.cuda.get_device_properties(device).total_memory / 1024**3
    return f"GPU mem: {alloc:.2f}/{reserved:.2f}/{total:.2f} GiB (alloc/reserved/total)"


@contextmanager
def _timed(label: str, device: torch.device = None):
    """Context manager that logs elapsed wall-clock time for a code block."""
    from utils import logger  # local import to avoid circular at module level

    t0 = time.perf_counter()
    logger.info(f"[START] {label}")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        mem = f"  {_gpu_mem_str(device)}" if device is not None else ""
        logger.info(f"[DONE ] {label} — {elapsed:.2f}s{mem}")


from distributed.fsdp import shard_model
from distributed.util import _configure_model, init_distributed
from modules.utils import (
    WanVAEStreamingWrapper,
    load_text_encoder,
    load_tokenizer,
    load_transformer,
    load_vae,
)
from utils import (
    FlowMatchScheduler,
    data_seq_to_patch,
    get_mesh_id,
    init_logger,
    logger,
    run_async_server_mode,
    save_async,
)


class VA_Server:
    def __init__(self, job_config):
        self.cache_name = "pos"
        self.job_config = job_config
        self.save_root = job_config.save_root
        self.dtype = job_config.param_dtype
        self.device = torch.device(f"cuda:{job_config.local_rank}")
        self.enable_offload = getattr(
            job_config, "enable_offload", True
        )  # offload vae & text_encoder to save vram

        logger.info(
            f"Initialising VA_Server | rank={job_config.rank} "
            f"local_rank={job_config.local_rank} world_size={job_config.world_size} "
            f"device={self.device} dtype={self.dtype} offload={self.enable_offload}"
        )

        with _timed("Build schedulers", self.device):
            self.scheduler = FlowMatchScheduler(
                shift=self.job_config.snr_shift, sigma_min=0.0, extra_one_step=True
            )
            self.action_scheduler = FlowMatchScheduler(
                shift=self.job_config.action_snr_shift,
                sigma_min=0.0,
                extra_one_step=True,
            )
            self.scheduler.set_timesteps(1000, training=True)
            self.action_scheduler.set_timesteps(1000, training=True)

        vae_target = "CPU" if self.enable_offload else str(self.device)
        with _timed(f"Load VAE → {vae_target}"):
            self.vae = load_vae(
                os.path.join(job_config.wan22_pretrained_model_name_or_path, "vae"),
                torch_dtype=self.dtype,
                torch_device="cpu" if self.enable_offload else self.device,
            )
            self.streaming_vae = WanVAEStreamingWrapper(self.vae)
            logger.info(
                f"  VAE streaming wrapper: {self.streaming_vae.enc_conv_num} causal-conv layers in cache"
            )

        with _timed("Load tokenizer"):
            self.tokenizer = load_tokenizer(
                os.path.join(
                    job_config.wan22_pretrained_model_name_or_path, "tokenizer"
                ),
            )

        te_target = "CPU" if self.enable_offload else str(self.device)
        with _timed(f"Load text encoder → {te_target}"):
            self.text_encoder = load_text_encoder(
                os.path.join(
                    job_config.wan22_pretrained_model_name_or_path, "text_encoder"
                ),
                torch_dtype=self.dtype,
                torch_device="cpu" if self.enable_offload else self.device,
            )

        with _timed(f"Load transformer → {self.device}", self.device):
            self.transformer = load_transformer(
                os.path.join(
                    job_config.wan22_pretrained_model_name_or_path, "transformer"
                ),
                torch_dtype=self.dtype,
                torch_device=self.device,
            )

        with _timed("Configure transformer (FSDP shard)", self.device):
            shard_fn = shard_model
            self.transformer = _configure_model(
                model=self.transformer,
                shard_fn=shard_fn,
                param_dtype=self.dtype,
                device=self.device,
                eval_mode=True,
            )

        self.env_type = job_config.env_type
        self.streaming_vae_half = None
        if self.env_type == "robotwin_tshape":
            with _timed(f"Load VAE (half-res wrist cams) → {vae_target}"):
                vae_half = load_vae(
                    os.path.join(job_config.wan22_pretrained_model_name_or_path, "vae"),
                    torch_dtype=self.dtype,
                    torch_device="cpu" if self.enable_offload else self.device,
                )
                self.streaming_vae_half = WanVAEStreamingWrapper(vae_half)

        logger.info(f"VA_Server ready. {_gpu_mem_str(self.device)}")

    def _get_t5_prompt_embeds(
        self,
        prompt=None,
        num_videos_per_prompt=1,
        max_sequence_length=512,
        device=None,
        dtype=None,
    ):
        device = device or self.device
        dtype = dtype or self.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        text_encoder_device = next(self.text_encoder.parameters()).device
        prompt_embeds = self.text_encoder(
            text_input_ids.to(text_encoder_device), mask.to(text_encoder_device)
        ).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [
                torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
                for u in prompt_embeds
            ],
            dim=0,
        )

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_videos_per_prompt, seq_len, -1
        )

        return prompt_embeds.to(device)

    def encode_prompt(
        self,
        prompt,
        negative_prompt=None,
        do_classifier_free_guidance=True,
        num_videos_per_prompt=1,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        max_sequence_length=226,
        device=None,
        dtype=None,
    ):
        r"""
        TODO
        """
        device = device or self.device
        dtype = dtype or self.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = (
                batch_size * [negative_prompt]
                if isinstance(negative_prompt, str)
                else negative_prompt
            )

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
        return prompt_embeds, negative_prompt_embeds

    def normalize_latents(
        self,
        latents: torch.Tensor,
        latents_mean: torch.Tensor,
        latents_std: torch.Tensor,
    ) -> torch.Tensor:
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(device=latents.device)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(device=latents.device)
        latents = ((latents.float() - latents_mean) * latents_std).to(latents)
        return latents

    def preprocess_action(self, action):
        action_model_input = torch.from_numpy(action)
        CA, FA, HA = action_model_input.shape  # C, F, H
        action_model_input_paded = F.pad(
            action_model_input, [0, 0, 0, 0, 0, 1], mode="constant", value=0
        )

        action_model_input = action_model_input_paded[
            self.job_config.inverse_used_action_channel_ids
        ]

        if self.action_norm_method == "quantiles":
            action_model_input = (action_model_input - self.actions_q01) / (
                self.actions_q99 - self.actions_q01 + 1e-6
            ) * 2.0 - 1.0
        else:
            raise NotImplementedError
        return action_model_input.unsqueeze(0).unsqueeze(-1)  # B, C, F, H, W

    def postprocess_action(self, action):
        action = action.cpu()  # B, C, F, H, W

        action = action[0, ..., 0]  # C, F, H
        if self.action_norm_method == "quantiles":
            action = (action + 1) / 2 * (
                self.actions_q99 - self.actions_q01 + 1e-6
            ) + self.actions_q01
        else:
            raise NotImplementedError
        action = action.squeeze(0).detach().cpu().numpy()
        return action[self.job_config.used_action_channel_ids]

    def _repeat_input_for_cfg(self, input_dict):
        if self.use_cfg:
            input_dict["noisy_latents"] = input_dict["noisy_latents"].repeat(
                2, 1, 1, 1, 1
            )
            input_dict["text_emb"] = torch.cat(
                [
                    self.prompt_embeds.to(self.dtype).clone(),
                    self.negative_prompt_embeds.to(self.dtype).clone(),
                ],
                dim=0,
            )
            input_dict["grid_id"] = input_dict["grid_id"][None].repeat(2, 1, 1)
            input_dict["timesteps"] = input_dict["timesteps"][None].repeat(2, 1)
        else:
            input_dict["grid_id"] = input_dict["grid_id"][None]
            input_dict["timesteps"] = input_dict["timesteps"][None]
        return input_dict

    def _prepare_latent_input(
        self,
        latent_model_input,
        action_model_input,
        latent_t=0,
        action_t=0,
        latent_cond=None,
        action_cond=None,
        frame_st_id=0,
        patch_size=(1, 2, 2),
    ):
        logger.info(f"FRAME START ID: {frame_st_id}")
        input_dict = dict()
        if latent_model_input is not None:
            input_dict["latent_res_lst"] = {
                "noisy_latents": latent_model_input,
                "timesteps": torch.ones(
                    [latent_model_input.shape[2]],
                    dtype=torch.float32,
                    device=self.device,
                )
                * latent_t,
                "grid_id": get_mesh_id(
                    latent_model_input.shape[-3] // patch_size[0],
                    latent_model_input.shape[-2] // patch_size[1],
                    latent_model_input.shape[-1] // patch_size[2],
                    0,
                    1,
                    frame_st_id,
                ).to(self.device),
                "text_emb": self.prompt_embeds.to(self.dtype).clone(),
            }
            if latent_cond is not None:
                input_dict["latent_res_lst"]["noisy_latents"][:, :, 0:1] = latent_cond[
                    :, :, 0:1
                ]
                input_dict["latent_res_lst"]["timesteps"][0:1] *= 0

        if action_model_input is not None:
            input_dict["action_res_lst"] = {
                "noisy_latents": action_model_input,
                "timesteps": torch.ones(
                    [action_model_input.shape[2]],
                    dtype=torch.float32,
                    device=self.device,
                )
                * action_t,
                "grid_id": get_mesh_id(
                    action_model_input.shape[-3],
                    action_model_input.shape[-2],
                    action_model_input.shape[-1],
                    1,
                    1,
                    frame_st_id,
                    action=True,
                ).to(self.device),
                "text_emb": self.prompt_embeds.to(self.dtype).clone(),
            }

            if action_cond is not None:
                input_dict["action_res_lst"]["noisy_latents"][:, :, 0:1] = action_cond[
                    :, :, 0:1
                ]
                input_dict["action_res_lst"]["timesteps"][0:1] *= 0
            input_dict["action_res_lst"]["noisy_latents"][:, ~self.action_mask] *= 0
        return input_dict

    def _encode_obs(self, obs):
        t0 = time.perf_counter()
        images = obs["obs"]
        if not isinstance(images, list):
            images = [images]
        if len(images) < 1:
            logger.warning("_encode_obs: empty observation list — returning None")
            return None

        n_frames = len(images)
        cam_keys = self.job_config.obs_cam_keys
        logger.info(
            f"_encode_obs | frames={n_frames}  cams={cam_keys}  "
            f"VAE on {'CPU (offloaded)' if next(self.streaming_vae.vae.parameters()).device.type == 'cpu' else self.device}"
        )

        videos = []
        for k_i, k in enumerate(cam_keys):
            if self.env_type == "robotwin_tshape":
                if k_i == 0:  # camera high
                    height_i, width_i = self.height, self.width
                else:
                    height_i, width_i = self.height // 2, self.width // 2
            else:
                height_i, width_i = self.height, self.width

            history_video_k = (
                torch.from_numpy(np.stack([each[k] for each in images]))
                .float()
                .permute(3, 0, 1, 2)
            )
            history_video_k = F.interpolate(
                history_video_k,
                size=(height_i, width_i),
                mode="bilinear",
                align_corners=False,
            ).unsqueeze(0)
            logger.info(
                f"  cam[{k_i}] '{k}' resized → {tuple(history_video_k.shape)} (B,C,T,H,W)"
            )
            videos.append(history_video_k)

        if self.env_type == "robotwin_tshape":
            videos_high = videos[0] / 255.0 * 2.0 - 1.0
            videos_left_and_right = torch.cat(videos[1:], dim=0) / 255.0 * 2.0 - 1.0
            vae_device = next(self.streaming_vae.vae.parameters()).device
            logger.info(
                f"  VAE encoding cam_high {tuple(videos_high.shape)} on {vae_device} …"
            )
            t_enc = time.perf_counter()
            enc_out_high = self.streaming_vae.encode_chunk(
                videos_high.to(vae_device).to(self.dtype)
            )
            logger.info(
                f"  cam_high encoded in {time.perf_counter() - t_enc:.2f}s → {tuple(enc_out_high.shape)}"
            )
            logger.info(
                f"  VAE encoding wrist cams {tuple(videos_left_and_right.shape)} on {vae_device} …"
            )
            t_enc = time.perf_counter()
            enc_out_left_and_right = self.streaming_vae_half.encode_chunk(
                videos_left_and_right.to(vae_device).to(self.dtype)
            )
            logger.info(
                f"  wrist cams encoded in {time.perf_counter() - t_enc:.2f}s → {tuple(enc_out_left_and_right.shape)}"
            )
            enc_out = torch.cat(
                [
                    torch.cat(enc_out_left_and_right.split(1, dim=0), dim=-1),
                    enc_out_high,
                ],
                dim=-2,
            )
        else:
            videos = torch.cat(videos, dim=0) / 255.0 * 2.0 - 1.0
            vae_device = next(self.streaming_vae.vae.parameters()).device
            videos_chunk = videos.to(vae_device).to(self.dtype)
            logger.info(
                f"  VAE encoding video chunk {tuple(videos_chunk.shape)} on {vae_device} …"
            )
            t_enc = time.perf_counter()
            enc_out = self.streaming_vae.encode_chunk(videos_chunk)
            logger.info(
                f"  VAE encoding done in {time.perf_counter() - t_enc:.2f}s → {tuple(enc_out.shape)}"
            )

        mu, logvar = torch.chunk(enc_out, 2, dim=1)
        latents_mean = torch.tensor(self.vae.config.latents_mean).to(mu.device)
        latents_std = torch.tensor(self.vae.config.latents_std).to(mu.device)
        mu_norm = self.normalize_latents(mu, latents_mean, 1.0 / latents_std)
        video_latent = torch.cat(mu_norm.split(1, dim=0), dim=-1)
        logger.info(
            f"_encode_obs done in {time.perf_counter() - t0:.2f}s  "
            f"video_latent={tuple(video_latent.shape)}"
        )
        return video_latent.to(self.device)

    def _reset(self, prompt=None):
        t_reset_start = time.perf_counter()
        logger.info(
            f"_reset() | prompt={'<none>' if prompt is None else repr(prompt[:80])} "
            f"| {_gpu_mem_str(self.device)}"
        )
        self.use_cfg = (self.job_config.guidance_scale > 1) or (
            self.job_config.action_guidance_scale > 1
        )
        logger.info(
            f"  CFG enabled={self.use_cfg}  "
            f"guidance_scale={self.job_config.guidance_scale}  "
            f"action_guidance_scale={self.job_config.action_guidance_scale}"
        )

        #### Reset all parameters
        self.frame_st_id = 0
        self.init_latent = None
        #### clean vae and transformer cache
        logger.info("  Clearing transformer KV cache and VAE streaming cache …")
        self.transformer.clear_cache(self.cache_name)
        self.streaming_vae.clear_cache()

        self.action_per_frame = self.job_config.action_per_frame
        self.height, self.width = self.job_config.height, self.job_config.width

        if self.env_type == "robotwin_tshape":
            self.latent_height, self.latent_width = (
                ((self.height // 16) * 3) // 2,
                self.width // 16,
            )
            self.streaming_vae_half.clear_cache()
        else:
            self.latent_height, self.latent_width = (
                self.height // 16,
                self.width // 16 * len(self.job_config.obs_cam_keys),
            )

        logger.info(
            f"  Image size: {self.height}×{self.width}  "
            f"Latent size: {self.latent_height}×{self.latent_width}  "
            f"frame_chunk_size={self.job_config.frame_chunk_size}  "
            f"action_per_frame={self.action_per_frame}"
        )

        patch_size = self.job_config.patch_size
        latent_token_per_chunk = (
            self.job_config.frame_chunk_size * self.latent_height * self.latent_width
        ) // (patch_size[0] * patch_size[1] * patch_size[2])
        action_token_per_chunk = (
            self.job_config.frame_chunk_size * self.action_per_frame
        )
        logger.info(
            f"  Creating empty KV cache: patch_size={patch_size}  "
            f"latent_tokens/chunk={latent_token_per_chunk}  "
            f"action_tokens/chunk={action_token_per_chunk}  "
            f"attn_window={self.job_config.attn_window}  "
            f"batch_size={'2 (CFG)' if self.use_cfg else '1'}"
        )
        self.transformer.create_empty_cache(
            self.cache_name,
            self.job_config.attn_window,
            latent_token_per_chunk,
            action_token_per_chunk,
            dtype=self.dtype,
            device=self.device,
            batch_size=2 if self.use_cfg else 1,
        )

        self.action_mask = torch.zeros([self.job_config.action_dim]).bool()
        self.action_mask[self.job_config.used_action_channel_ids] = True

        self.actions_q01 = torch.tensor(
            self.job_config.norm_stat["q01"], dtype=torch.float32
        ).reshape(-1, 1, 1)
        self.actions_q99 = torch.tensor(
            self.job_config.norm_stat["q99"], dtype=torch.float32
        ).reshape(-1, 1, 1)
        self.action_norm_method = self.job_config.action_norm_method
        logger.info(
            f"  Action dim={self.job_config.action_dim}  "
            f"used_channels={self.job_config.used_action_channel_ids}  "
            f"norm_method={self.action_norm_method}"
        )

        ##### get prompt
        if prompt is None:
            self.prompt_embeds = self.negative_prompt_embeds = None
            logger.info("  No prompt — running without text conditioning.")
        else:
            logger.info("  Encoding text prompt with T5 …")
            t_enc = time.perf_counter()
            self.prompt_embeds, self.negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=None,
                do_classifier_free_guidance=self.job_config.guidance_scale > 1,
                num_videos_per_prompt=1,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                max_sequence_length=512,
                device=self.device,
                dtype=self.dtype,
            )
            logger.info(
                f"  T5 encoding done in {time.perf_counter() - t_enc:.2f}s  "
                f"prompt_embeds={tuple(self.prompt_embeds.shape)}"
            )

        self.exp_name = (
            f"{prompt}_{time.strftime('%Y%m%d_%H%M%S')}" if prompt else "default"
        )
        self.exp_save_root = os.path.join(self.save_root, "real", self.exp_name)
        os.makedirs(self.exp_save_root, exist_ok=True)
        torch.cuda.empty_cache()
        logger.info(
            f"_reset() done in {time.perf_counter() - t_reset_start:.2f}s  "
            f"save_root={self.exp_save_root}  {_gpu_mem_str(self.device)}"
        )

    def _infer(self, obs, frame_st_id=0):
        t_infer_start = time.perf_counter()
        frame_chunk_size = self.job_config.frame_chunk_size
        logger.info(
            f"_infer() | frame_st_id={frame_st_id}  frame_chunk_size={frame_chunk_size}  "
            f"{_gpu_mem_str(self.device)}"
        )

        if frame_st_id == 0:
            logger.info(
                "  frame_st_id=0: encoding initial observation for conditioning …"
            )
            init_latent = self._encode_obs(obs)
            self.init_latent = init_latent
            logger.info(f"  init_latent={tuple(init_latent.shape)}")

        latents = torch.randn(
            1,
            48,
            frame_chunk_size,
            self.latent_height,
            self.latent_width,
            device=self.device,
            dtype=self.dtype,
        )
        actions = torch.randn(
            1,
            self.job_config.action_dim,
            frame_chunk_size,
            self.action_per_frame,
            1,
            device=self.device,
            dtype=self.dtype,
        )
        logger.info(
            f"  Noise initialised: latents={tuple(latents.shape)}  actions={tuple(actions.shape)}"
        )

        video_inference_step = self.job_config.num_inference_steps
        action_inference_step = self.job_config.action_num_inference_steps
        video_step = self.job_config.video_exec_step

        self.scheduler.set_timesteps(video_inference_step)
        self.action_scheduler.set_timesteps(action_inference_step)
        timesteps = self.scheduler.timesteps
        action_timesteps = self.action_scheduler.timesteps

        timesteps = F.pad(timesteps, (0, 1), mode="constant", value=0)

        if video_step != -1:
            timesteps = timesteps[:video_step]

        action_timesteps = F.pad(
            action_timesteps,
            (0, 1),  # pad 1 element at the end (right side) of the last dimension
            mode="constant",
            value=0,
        )

        logger.info(
            f"  Denoising schedule: video_steps={len(timesteps)} (exec_step={video_step})  "
            f"action_steps={len(action_timesteps)}"
        )

        with (
            torch.no_grad(),
        ):
            # 1. Video Generation Loop
            logger.info(
                f"  --- Video denoising loop: {len(timesteps)} steps ---  "
                f"{_gpu_mem_str(self.device)}"
            )
            t_video_loop = time.perf_counter()
            step_times = []
            for i, t in enumerate(tqdm(timesteps, desc="video denoise")):
                t_step = time.perf_counter()
                last_step = i == len(timesteps) - 1
                latent_cond = (
                    init_latent[:, :, 0:1].to(self.dtype) if frame_st_id == 0 else None
                )
                input_dict = self._prepare_latent_input(
                    latents, None, t, t, latent_cond, None, frame_st_id=frame_st_id
                )

                video_noise_pred = self.transformer(
                    self._repeat_input_for_cfg(input_dict["latent_res_lst"]),
                    update_cache=1 if last_step else 0,
                    cache_name=self.cache_name,
                    action_mode=False,
                )

                if not last_step or video_step != -1:
                    video_noise_pred = data_seq_to_patch(
                        self.job_config.patch_size,
                        video_noise_pred,
                        frame_chunk_size,
                        self.latent_height,
                        self.latent_width,
                        batch_size=2 if self.use_cfg else 1,
                    )
                    if self.job_config.guidance_scale > 1:
                        video_noise_pred = video_noise_pred[
                            1:
                        ] + self.job_config.guidance_scale * (
                            video_noise_pred[:1] - video_noise_pred[1:]
                        )
                    else:
                        video_noise_pred = video_noise_pred[:1]
                    latents = self.scheduler.step(
                        video_noise_pred, t, latents, return_dict=False
                    )

                latents[:, :, 0:1] = (
                    latent_cond if frame_st_id == 0 else latents[:, :, 0:1]
                )

                step_elapsed = time.perf_counter() - t_step
                step_times.append(step_elapsed)
                logger.info(
                    f"  video step [{i + 1}/{len(timesteps)}] t={t:.4f}  "
                    f"step={step_elapsed:.2f}s  last_step={last_step}  "
                    f"update_cache={1 if last_step else 0}  "
                    f"{_gpu_mem_str(self.device)}"
                )

            avg_step = sum(step_times) / len(step_times) if step_times else 0.0
            logger.info(
                f"  Video denoising loop done in {time.perf_counter() - t_video_loop:.2f}s  "
                f"avg_step={avg_step:.2f}s  {_gpu_mem_str(self.device)}"
            )

            # 2. Action Generation Loop
            logger.info(
                f"  --- Action denoising loop: {len(action_timesteps)} steps ---  "
                f"{_gpu_mem_str(self.device)}"
            )
            t_action_loop = time.perf_counter()
            action_step_times = []
            for i, t in enumerate(tqdm(action_timesteps, desc="action denoise")):
                t_step = time.perf_counter()
                last_step = i == len(action_timesteps) - 1
                action_cond = (
                    torch.zeros(
                        [1, self.job_config.action_dim, 1, self.action_per_frame, 1],
                        device=self.device,
                        dtype=self.dtype,
                    )
                    if frame_st_id == 0
                    else None
                )

                input_dict = self._prepare_latent_input(
                    None, actions, t, t, None, action_cond, frame_st_id=frame_st_id
                )
                action_noise_pred = self.transformer(
                    self._repeat_input_for_cfg(input_dict["action_res_lst"]),
                    update_cache=1 if last_step else 0,
                    cache_name=self.cache_name,
                    action_mode=True,
                )

                if not last_step:
                    action_noise_pred = rearrange(
                        action_noise_pred, "b (f n) c -> b c f n 1", f=frame_chunk_size
                    )
                    if self.job_config.action_guidance_scale > 1:
                        action_noise_pred = action_noise_pred[
                            1:
                        ] + self.job_config.action_guidance_scale * (
                            action_noise_pred[:1] - action_noise_pred[1:]
                        )
                    else:
                        action_noise_pred = action_noise_pred[:1]
                    actions = self.action_scheduler.step(
                        action_noise_pred, t, actions, return_dict=False
                    )

                actions[:, :, 0:1] = (
                    action_cond if frame_st_id == 0 else actions[:, :, 0:1]
                )

                step_elapsed = time.perf_counter() - t_step
                action_step_times.append(step_elapsed)
                logger.info(
                    f"  action step [{i + 1}/{len(action_timesteps)}] t={t:.4f}  "
                    f"step={step_elapsed:.2f}s  last_step={last_step}  "
                    f"update_cache={1 if last_step else 0}  "
                    f"{_gpu_mem_str(self.device)}"
                )

            avg_action_step = (
                sum(action_step_times) / len(action_step_times)
                if action_step_times
                else 0.0
            )
            logger.info(
                f"  Action denoising loop done in {time.perf_counter() - t_action_loop:.2f}s  "
                f"avg_step={avg_action_step:.2f}s  {_gpu_mem_str(self.device)}"
            )

        actions[:, ~self.action_mask] *= 0

        save_async(
            latents, os.path.join(self.exp_save_root, f"latents_{frame_st_id}.pt")
        )
        save_async(
            actions, os.path.join(self.exp_save_root, f"actions_{frame_st_id}.pt")
        )

        actions = self.postprocess_action(actions)
        torch.cuda.empty_cache()
        logger.info(
            f"_infer() done in {time.perf_counter() - t_infer_start:.2f}s  "
            f"actions.shape={actions.shape}  {_gpu_mem_str(self.device)}"
        )
        return actions, latents

    def _compute_kv_cache(self, obs):
        t0 = time.perf_counter()
        logger.info(
            f"_compute_kv_cache() | frame_st_id={self.frame_st_id}  "
            f"{_gpu_mem_str(self.device)}"
        )
        ### optional async save obs for debug
        self.transformer.clear_pred_cache(self.cache_name)
        save_async(
            obs["obs"],
            os.path.join(self.exp_save_root, f"obs_data_{self.frame_st_id}.pt"),
        )

        logger.info("  Encoding obs with VAE …")
        latent_model_input = self._encode_obs(obs)
        if self.frame_st_id == 0:
            logger.info("  frame_st_id=0: prepending init_latent to history …")
            latent_model_input = (
                torch.cat([self.init_latent, latent_model_input], dim=2)
                if latent_model_input is not None
                else self.init_latent
            )

        action_model_input = self.preprocess_action(obs["state"])
        action_model_input = action_model_input.to(latent_model_input)
        logger.info(
            f"  KV cache input: latent={tuple(latent_model_input.shape)}  "
            f"action={tuple(action_model_input.shape)}"
        )
        input_dict = self._prepare_latent_input(
            latent_model_input, action_model_input, frame_st_id=self.frame_st_id
        )

        with (
            torch.no_grad(),
        ):
            logger.info("  Transformer forward (video KV cache update=2) …")
            t_fwd = time.perf_counter()
            self.transformer(
                self._repeat_input_for_cfg(input_dict["latent_res_lst"]),
                update_cache=2,
                cache_name=self.cache_name,
                action_mode=False,
            )
            logger.info(
                f"  Video KV cache forward done in {time.perf_counter() - t_fwd:.2f}s"
            )

            logger.info("  Transformer forward (action KV cache update=2) …")
            t_fwd = time.perf_counter()
            self.transformer(
                self._repeat_input_for_cfg(input_dict["action_res_lst"]),
                update_cache=2,
                cache_name=self.cache_name,
                action_mode=True,
            )
            logger.info(
                f"  Action KV cache forward done in {time.perf_counter() - t_fwd:.2f}s"
            )

        torch.cuda.empty_cache()
        self.frame_st_id += latent_model_input.shape[2]
        logger.info(
            f"_compute_kv_cache() done in {time.perf_counter() - t0:.2f}s  "
            f"new frame_st_id={self.frame_st_id}  {_gpu_mem_str(self.device)}"
        )

    @torch.no_grad()
    def infer(self, obs):
        reset = obs.get("reset", False)
        prompt = obs.get("prompt", None)
        compute_kv_cache = obs.get("compute_kv_cache", False)

        if reset:
            logger.info(f"******************* Reset server ******************")
            self._reset(prompt=prompt)
            return dict()
        elif compute_kv_cache:
            logger.info(f"################# Compute KV Cache #################")
            self._compute_kv_cache(obs)
            return dict()
        else:
            logger.info(f"################# Infer One Chunk #################")
            action, _ = self._infer(obs, frame_st_id=self.frame_st_id)
            return dict(action=action)

    def decode_one_video(self, latents, output_type):
        latents = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1, self.vae.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)
        latents = latents / latents_std + latents_mean
        video = self.vae.decode(latents, return_dict=False)[0]
        video = self.video_processor.postprocess_video(video, output_type=output_type)
        return video

    def load_init_obs(self):
        imf_dict = {
            v: np.array(
                Image.open(
                    os.path.join(self.job_config.input_img_path, f"{v}.png")
                ).convert("RGB")
            )
            for v in self.job_config.obs_cam_keys
        }
        init_obs = {}
        init_obs["obs"] = [imf_dict]
        return init_obs

    @torch.no_grad()
    def generate(self):
        t_generate_start = time.perf_counter()
        num_chunks = self.job_config.num_chunks_to_infer
        logger.info(
            f"generate() | i2va mode | num_chunks={num_chunks}  "
            f"frame_chunk_size={self.job_config.frame_chunk_size}  "
            f"{_gpu_mem_str(self.device)}"
        )
        self.video_processor = VideoProcessor(vae_scale_factor=1)
        self._reset(self.job_config.prompt)

        logger.info(
            f"  Loading initial observation images from {self.job_config.input_img_path} …"
        )
        init_obs = self.load_init_obs()
        logger.info(f"  Loaded init_obs with {len(init_obs['obs'])} frames")

        pred_latent_lst = []
        pred_action_lst = []
        for chunk_id in range(num_chunks):
            frame_st_id = chunk_id * self.job_config.frame_chunk_size
            logger.info(
                f"  Chunk [{chunk_id + 1}/{num_chunks}] frame_st_id={frame_st_id}  "
                f"{_gpu_mem_str(self.device)}"
            )
            t_chunk = time.perf_counter()
            actions, latents = self._infer(init_obs, frame_st_id=frame_st_id)
            logger.info(
                f"  Chunk [{chunk_id + 1}/{num_chunks}] done in {time.perf_counter() - t_chunk:.2f}s"
            )
            actions = torch.from_numpy(actions)
            pred_latent_lst.append(latents)
            pred_action_lst.append(actions)

        pred_latent = torch.cat(pred_latent_lst, dim=2)
        pred_action = torch.cat(pred_action_lst, dim=1).flatten(1)
        logger.info(
            f"  All chunks inferred: pred_latent={tuple(pred_latent.shape)}  "
            f"pred_action={tuple(pred_action.shape)}"
        )

        logger.info("  Clearing caches and freeing transformer / text_encoder …")
        self.transformer.clear_cache(self.cache_name)
        self.streaming_vae.clear_cache()
        if self.streaming_vae_half:
            self.streaming_vae_half.clear_cache()
        del self.transformer
        del self.streaming_vae_half
        del self.text_encoder
        torch.cuda.empty_cache()
        logger.info(f"  After freeing transformer: {_gpu_mem_str(self.device)}")

        # Move VAE to GPU for decoding
        if self.enable_offload:
            logger.info(f"  Moving VAE from CPU → {self.device} for decoding …")
            t_move = time.perf_counter()
            self.vae = self.vae.to(self.device).to(self.dtype)
            logger.info(
                f"  VAE moved in {time.perf_counter() - t_move:.2f}s  {_gpu_mem_str(self.device)}"
            )

        logger.info(f"  Decoding latents → video frames …")
        t_decode = time.perf_counter()
        decoded_video = self.decode_one_video(pred_latent, "np")[0]
        out_path = os.path.join(self.save_root, "demo.mp4")
        logger.info(
            f"  Decoded in {time.perf_counter() - t_decode:.2f}s  frames={len(decoded_video)}"
        )
        export_to_video(decoded_video, out_path, fps=10)
        logger.info(
            f"generate() done in {time.perf_counter() - t_generate_start:.2f}s  "
            f"output saved to {out_path}"
        )


def run(args):
    # Initialise logging first so every rank emits structured log lines
    init_logger()

    config = VA_CONFIGS[args.config_name]
    port = config.port if args.port is None else args.port
    if args.save_root is not None:
        config.save_root = args.save_root
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    logger.info(
        f"run() start | config_name={args.config_name}  "
        f"rank={rank} local_rank={local_rank} world_size={world_size}  "
        f"infer_mode={config.infer_mode}"
    )

    with _timed(f"init_distributed(world_size={world_size})"):
        init_distributed(world_size, local_rank, rank)

    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size

    with _timed(
        "VA_Server.__init__ (model loading)", torch.device(f"cuda:{local_rank}")
    ):
        model = VA_Server(config)

    if config.infer_mode == "i2va":
        logger.info("=" * 60)
        logger.info("MODE: i2va (offline image-to-video+action generation)")
        logger.info("=" * 60)
        model.generate()
    elif config.infer_mode == "server":
        logger.info("=" * 60)
        logger.info(f"MODE: server (WebSocket policy server on {config.host}:{port})")
        logger.info("=" * 60)
        run_async_server_mode(model, local_rank, config.host, port)
    else:
        raise ValueError(f"Unknown infer mode: {config.infer_mode}")


def main():
    """
    TODO
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-name",
        type=str,
        required=False,
        default="robotwin",
        help="config name.",
    )
    parser.add_argument("--port", type=int, default=None, help="(start) port")
    parser.add_argument("--save_root", type=str, default=None, help="save root")
    args = parser.parse_args()
    run(args)
    logger.info("Finish all process!!!!!!!!!!!!")


if __name__ == "__main__":
    init_logger()
    main()
