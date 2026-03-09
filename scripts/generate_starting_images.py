"""
Generate starting images from RoboTwin simulator for instruction-following evaluation.

Uses place_a2b_left and place_a2b_right tasks to produce 50 starting-image sets,
each paired with a left-arm and right-arm text instruction variant.
The two instructions differ ONLY in which arm the robot should use, while the
spatial goal (place left/right of B) stays the same.

Output structure:
    {output_dir}/
        sample_{i:03d}/
            observation.images.cam_high.png
            observation.images.cam_left_wrist.png
            observation.images.cam_right_wrist.png
            metadata.json   # seed, task, instruction_left_arm, instruction_right_arm
"""

import sys
import os
import json
import traceback
import argparse
import random

import numpy as np
import cv2
import yaml
import importlib
from pathlib import Path

# ── path setup (mirror what eval_polict_client_openpi.py does) ──
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBOTWIN_ROOT = PROJECT_ROOT / "robotwin"

sys.path.insert(0, str(ROBOTWIN_ROOT))
os.chdir(ROBOTWIN_ROOT)

from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError
from description.utils.generate_episode_instructions import (
    generate_episode_descriptions,
)


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    env_class = getattr(envs_module, task_name)
    return env_class()


def get_embodiment_config(robot_file):
    cfg_file = os.path.join(robot_file, "config.yml")
    with open(cfg_file, "r") as f:
        return yaml.safe_load(f)


def build_args(task_name: str, config_name: str = "demo_clean") -> dict:
    """Build the args dict expected by setup_demo, matching the eval client."""
    config_path = f"./task_config/{config_name}.yml"
    with open(config_path, "r") as f:
        args = yaml.safe_load(f)

    args["task_name"] = task_name

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r") as f:
        _embodiment_types = yaml.safe_load(f)

    def get_embodiment_file(et):
        rf = _embodiment_types[et]["file_path"]
        if rf is None:
            raise RuntimeError("missing embodiment files")
        return rf

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    args["eval_mode"] = True
    args["render_freq"] = 0  # no video during seed search
    args["collect_data"] = False
    return args


def save_obs_images(obs: dict, save_dir: Path):
    """Save the 3 camera RGB images as PNGs."""
    save_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "observation.images.cam_high": ("head_camera", "rgb"),
        "observation.images.cam_left_wrist": ("left_camera", "rgb"),
        "observation.images.cam_right_wrist": ("right_camera", "rgb"),
    }
    for filename_key, (cam_name, data_key) in mapping.items():
        rgb = obs["observation"][cam_name][data_key]  # H,W,3 uint8 (RGB)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(save_dir / f"{filename_key}.png"), bgr)


def generate_instruction_pair(task_name: str, episode_info: dict):
    """Generate a (left_arm_instruction, right_arm_instruction) pair.

    Both instructions describe the SAME spatial goal (e.g. "place A to the
    left of B") but differ ONLY in which arm the robot should use.  This
    isolates arm-choice as the single variable for instruction-following
    evaluation.
    """
    episode_info_list = [episode_info]
    results = generate_episode_descriptions(task_name, episode_info_list, 100)
    seen_instructions = results[0]["seen"] if results else []
    unseen_instructions = results[0]["unseen"] if results else []

    all_instructions = seen_instructions + unseen_instructions
    if not all_instructions:
        return None, None

    # Pick an instruction that already mentions an arm (has {a} filled in).
    # Filter for ones containing "left arm" or "right arm".
    arm_instructions = [
        i for i in all_instructions
        if "left arm" in i.lower() or "right arm" in i.lower()
    ]
    if not arm_instructions:
        # Fallback: pick any instruction and prepend arm prefix
        base = random.choice(all_instructions)
        left_arm_instr = f"Use the left arm. {base}"
        right_arm_instr = f"Use the right arm. {base}"
        return left_arm_instr, right_arm_instr

    instruction = random.choice(arm_instructions)

    # Replace ONLY the arm reference, not spatial "left"/"right" directions
    left_arm_instr = swap_arm_only(instruction, target_arm="left")
    right_arm_instr = swap_arm_only(instruction, target_arm="right")

    return left_arm_instr, right_arm_instr


def swap_arm_only(text: str, target_arm: str) -> str:
    """Replace 'the left arm' / 'the right arm' with 'the {target_arm} arm'.

    Only touches the arm reference, leaving spatial directions intact.
    """
    import re
    return re.sub(
        r'\bthe (left|right) arm\b',
        f'the {target_arm} arm',
        text,
        flags=re.IGNORECASE,
    )


def find_valid_seeds(task_env, task_name: str, args: dict,
                     num_needed: int, start_seed: int = 10000):
    """Find seeds where the task expert can solve the scene.

    Returns list of (seed, episode_info) tuples.
    """
    valid = []
    seed = start_seed
    max_attempts = num_needed * 20  # generous budget
    attempts = 0

    while len(valid) < num_needed and attempts < max_attempts:
        attempts += 1
        try:
            task_env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **args)
            episode_info = task_env.play_once()
            if task_env.plan_success and task_env.check_success():
                valid.append((seed, episode_info["info"]))
                print(f"  [seed {seed}] valid ({len(valid)}/{num_needed})")
            task_env.close_env()
        except UnStableError:
            task_env.close_env()
        except Exception as e:
            task_env.close_env()
            print(f"  [seed {seed}] error: {e}")
        seed += 1

    return valid


def capture_initial_obs(task_env, seed: int, args: dict) -> dict:
    """Re-init the scene at the given seed, capture obs.

    render_freq must stay 0 (headless) — the off-screen cameras in get_obs()
    work without a Viewer window.
    """
    task_env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **args)
    obs = task_env.get_obs()
    task_env.close_env()
    return obs


def main():
    parser = argparse.ArgumentParser(
        description="Generate starting images for instruction-following evaluation"
    )
    parser.add_argument(
        "--num-samples", type=int, default=50,
        help="Total number of starting-image sets to generate (default: 50)",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(PROJECT_ROOT / "eval_starting_images"),
        help="Output directory (use absolute path; cwd changes to robotwin/)",
    )
    parser.add_argument(
        "--start-seed", type=int, default=10000,
        help="Starting seed for valid-seed search",
    )
    parser.add_argument(
        "--config", type=str, default="demo_clean",
        help="RoboTwin config name (default: demo_clean)",
    )
    cli_args = parser.parse_args()

    output_dir = Path(cli_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    num_samples = cli_args.num_samples

    # We split samples between place_a2b_left and place_a2b_right
    # so the *scene layout* itself varies (object placed left vs right).
    # Each sample gets BOTH a left-instruction and right-instruction.
    tasks = ["place_a2b_left", "place_a2b_right"]
    samples_per_task = [num_samples // 2, num_samples - num_samples // 2]

    sample_idx = 0
    manifest = []

    for task_name, n_samples in zip(tasks, samples_per_task):
        print(f"\n{'='*60}")
        print(f"Task: {task_name}  |  Samples needed: {n_samples}")
        print(f"{'='*60}")

        args = build_args(task_name, cli_args.config)
        task_env = class_decorator(task_name)

        # Phase 1: find valid seeds (no rendering, fast)
        print("Phase 1: Finding valid seeds...")
        valid_seeds = find_valid_seeds(
            task_env, task_name, args,
            num_needed=n_samples,
            start_seed=cli_args.start_seed,
        )

        if len(valid_seeds) < n_samples:
            print(f"WARNING: Only found {len(valid_seeds)}/{n_samples} valid seeds "
                  f"for {task_name}")

        # Phase 2: re-render each valid seed and capture starting images
        print(f"\nPhase 2: Capturing {len(valid_seeds)} starting images...")
        for seed, episode_info in valid_seeds:
            sample_dir = output_dir / f"sample_{sample_idx:03d}"

            # Capture the initial observation
            obs = capture_initial_obs(task_env, seed, args)
            save_obs_images(obs, sample_dir)

            # Generate instruction pair
            left_instr, right_instr = generate_instruction_pair(
                task_name, episode_info
            )

            metadata = {
                "sample_idx": sample_idx,
                "seed": seed,
                "task_name": task_name,
                "episode_info": episode_info,
                "instruction_left_arm": left_instr,
                "instruction_right_arm": right_instr,
            }
            with open(sample_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            manifest.append(metadata)
            print(f"  sample_{sample_idx:03d}: seed={seed}  "
                  f"L_arm=\"{left_instr[:60]}...\"  "
                  f"R_arm=\"{right_instr[:60]}...\"")
            sample_idx += 1

    # Write manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nDone! Generated {sample_idx} samples in {output_dir}")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
