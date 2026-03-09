"""
Batch i2va evaluation for left-arm vs right-arm instruction following.

Loads the wan_va model ONCE and iterates through all samples in the
eval_starting_images directory.  For each sample it runs inference twice:
once with the left-arm instruction and once with the right-arm instruction.

Output structure (under --save-root):
    sample_{idx:03d}/
        left_arm/
            video.mp4
            actions.npy
        right_arm/
            video.mp4
            actions.npy

Launch (single GPU):
    NGPU=1 bash scripts/run_arm_eval.sh

Launch (multi-GPU):
    NGPU=4 bash scripts/run_arm_eval.sh
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from diffusers.video_processor import VideoProcessor
from diffusers.utils import export_to_video

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from wan_va.configs import VA_CONFIGS
from wan_va.distributed.util import init_distributed
from wan_va.utils import init_logger, logger
from wan_va.wan_va_server import VA_Server, _timed, _gpu_mem_str


def load_obs_from_dir(img_dir: str, cam_keys: list) -> dict:
    """Load the 3 camera PNGs from a sample directory into the obs format
    expected by VA_Server._infer()."""
    imf_dict = {
        k: np.array(
            Image.open(os.path.join(img_dir, f"{k}.png")).convert("RGB")
        )
        for k in cam_keys
    }
    return {"obs": [imf_dict]}


@torch.no_grad()
def generate_one(
    model: VA_Server,
    prompt: str,
    init_obs: dict,
    num_chunks: int,
    save_dir: str,
):
    """Run i2va inference for one (image, prompt) pair and save results."""
    os.makedirs(save_dir, exist_ok=True)

    model._reset(prompt)

    pred_latent_lst = []
    pred_action_lst = []

    for chunk_id in range(num_chunks):
        frame_st_id = chunk_id * model.job_config.frame_chunk_size
        actions, latents = model._infer(init_obs, frame_st_id=frame_st_id)
        actions = torch.from_numpy(actions)
        pred_latent_lst.append(latents)
        pred_action_lst.append(actions)

    pred_latent = torch.cat(pred_latent_lst, dim=2)
    pred_action = torch.cat(pred_action_lst, dim=1).flatten(1)

    # Save actions
    np.save(os.path.join(save_dir, "actions.npy"), pred_action.cpu().numpy())

    # Decode video — need VAE on GPU
    if model.enable_offload:
        model.vae = model.vae.to(model.device).to(model.dtype)

    decoded_video = model.decode_one_video(pred_latent, "np")[0]
    export_to_video(decoded_video, os.path.join(save_dir, "video.mp4"), fps=10)

    # Move VAE back to CPU to free VRAM for the next transformer pass
    if model.enable_offload:
        model.vae = model.vae.to("cpu")
        torch.cuda.empty_cache()


def main():
    init_logger()

    parser = argparse.ArgumentParser(
        description="Batch arm-distinction evaluation via i2va"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing sample_XXX/ dirs (from generate_starting_images.py)",
    )
    parser.add_argument(
        "--save-root",
        type=str,
        default="./train_out/arm_eval",
        help="Output root for generated videos and actions",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="robotwin_i2av",
        help="Config name from VA_CONFIGS (default: robotwin_i2av)",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=None,
        help="Override number of chunks to infer per sample (default: from config)",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="First sample index to process (for resuming)",
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        default=None,
        help="Last sample index (exclusive) to process",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to process (alternative to --end-idx)",
    )
    args = parser.parse_args()

    if args.num_samples is not None and args.end_idx is None:
        args.end_idx = args.start_idx + args.num_samples

    # ── Load config ──
    config = VA_CONFIGS[args.config_name]
    config.save_root = args.save_root

    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    logger.info(
        f"arm_eval | config={args.config_name} rank={rank} "
        f"local_rank={local_rank} world_size={world_size}"
    )

    with _timed(f"init_distributed(world_size={world_size})"):
        init_distributed(world_size, local_rank, rank)

    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size

    # ── Load model (once) ──
    with _timed("VA_Server.__init__", torch.device(f"cuda:{local_rank}")):
        model = VA_Server(config)
    model.video_processor = VideoProcessor(vae_scale_factor=1)

    num_chunks = args.num_chunks or config.num_chunks_to_infer
    cam_keys = config.obs_cam_keys

    # ── Load manifest ──
    manifest_path = os.path.join(args.input_dir, "manifest.json")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    end_idx = args.end_idx or len(manifest)
    samples = manifest[args.start_idx : end_idx]

    logger.info(
        f"Processing samples [{args.start_idx}..{end_idx}) "
        f"({len(samples)} samples × 2 arms = {len(samples) * 2} inferences)  "
        f"num_chunks={num_chunks}"
    )

    results = []

    for entry in samples:
        idx = entry["sample_idx"]
        sample_dir = os.path.join(args.input_dir, f"sample_{idx:03d}")

        # Support both old key names and new key names
        instr_left = entry.get("instruction_left_arm") or entry.get("instruction_left")
        instr_right = entry.get("instruction_right_arm") or entry.get("instruction_right")

        logger.info(f"{'='*60}")
        logger.info(f"Sample {idx}: {sample_dir}")
        logger.info(f"  LEFT ARM:  {instr_left}")
        logger.info(f"  RIGHT ARM: {instr_right}")

        init_obs = load_obs_from_dir(sample_dir, cam_keys)

        for arm_label, prompt in [("left_arm", instr_left), ("right_arm", instr_right)]:
            out_dir = os.path.join(args.save_root, f"sample_{idx:03d}", arm_label)
            if os.path.exists(os.path.join(out_dir, "video.mp4")):
                logger.info(f"  [{arm_label}] Already exists, skipping.")
                continue

            logger.info(f"  [{arm_label}] Generating {num_chunks} chunks ...")
            t0 = time.perf_counter()
            generate_one(model, prompt, init_obs, num_chunks, out_dir)
            elapsed = time.perf_counter() - t0
            logger.info(
                f"  [{arm_label}] Done in {elapsed:.1f}s  "
                f"→ {out_dir}/video.mp4"
            )

        results.append({
            "sample_idx": idx,
            "instruction_left_arm": instr_left,
            "instruction_right_arm": instr_right,
            "left_arm_video": os.path.join(
                args.save_root, f"sample_{idx:03d}", "left_arm", "video.mp4"
            ),
            "right_arm_video": os.path.join(
                args.save_root, f"sample_{idx:03d}", "right_arm", "video.mp4"
            ),
        })

    # Save results manifest
    results_path = os.path.join(args.save_root, "eval_results.json")
    os.makedirs(args.save_root, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"All done! Results written to {results_path}")


if __name__ == "__main__":
    main()
