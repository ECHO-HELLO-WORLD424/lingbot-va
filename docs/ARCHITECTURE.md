# LingBot-VA Architecture Overview

LingBot-VA is an **Autoregressive Diffusion (AR-Diffusion)** framework that jointly models visual world dynamics and robot action in a single interleaved sequence. It is built on top of the [Wan2.2](https://github.com/Wan-Video) video diffusion backbone and extends it with a dual-stream Mixture-of-Transformers (MoT) design.

## Component Breakdown

### 1. `wan_va/` — Core Model Package

The main Python package containing all model, training, and inference code.

#### `wan_va/modules/model.py`
The heart of the system. Implements `WanTransformer3DModel`, a 3-D video transformer extended with:

- **Dual-stream MoT blocks**: separate transformer streams for video latents and action tokens that share cross-attention with text embeddings but have independent self-attention weights.
- **FlexAttnFunc**: a compiled `torch.compile`-d flex-attention kernel with a custom block-causal mask. The mask enforces:
  - *clean → clean*: causal across frames (past frames attend to past frames).
  - *noise → clean*: noisy tokens attend to all past clean tokens (AR denoising).
  - *noise → noise*: noisy tokens attend only to tokens in the same frame (within-frame diffusion).
- **Sliding-window attention** (`attn_window`): limits the temporal receptive field to avoid quadratic cost on long sequences.
- **KV Cache** (`cache_name = "pos"`): caches key/value tensors of already-denoised frames for efficient autoregressive rollout at inference time.
- Three attention backends selectable via `attn_mode` in `transformer/config.json`:
  - `"flex"` — required for training (uses `FlexAttnFunc`).
  - `"torch"` — standard SDPA, for inference.
  - `"flashattn"` — FlashAttention2/3, for fast inference.

#### `wan_va/modules/utils.py`
Model loading helpers:
- `load_transformer` — loads `WanTransformer3DModel` from a pretrained checkpoint.
- `load_vae` / `WanVAEStreamingWrapper` — loads the Wan2.2 VAE; the streaming wrapper encodes/decodes video in temporal chunks to reduce peak VRAM.
- `load_text_encoder` / `load_tokenizer` — loads the Wan2.2 T5-based text encoder and tokenizer.

#### `wan_va/wan_va_server.py`
The **inference server** (`VA_Server`). Runs on one or more GPUs and:
1. Loads the transformer, VAE, and text encoder.
2. Listens on a WebSocket port (default `29536`).
3. On each request: receives camera images + task instruction, encodes them, runs the AR diffusion loop (chunk by chunk), and returns predicted video frames and actions.
4. Supports **asynchronous execution** (`run_async_server_mode`): the next diffusion chunk is computed while the robot is executing the current action chunk, hiding latency.
5. Supports **CPU offload** (`enable_offload`) for VAE and text encoder to reduce VRAM usage (~24 GB without offload, ~18 GB with).

#### `wan_va/train.py`
The **post-training (fine-tuning) trainer** (`Trainer`). Uses PyTorch FSDP2 for distributed training:
- Loads a pretrained checkpoint and wraps it with `shard_model` (FSDP2) and optional activation checkpointing (`apply_ac`).
- Trains with **Flow Matching** loss (`FlowMatchScheduler`) on video latents and action tokens simultaneously.
- Logs metrics to Weights & Biases.
- Saves checkpoints every `save_interval` steps in SafeTensors format.

#### `wan_va/dataset/lerobot_latent_dataset.py`
The **dataset loader** (`LatentLeRobotDataset` / `MultiLatentLeRobotDataset`). Reads pre-extracted VAE latents and action sequences from a LeRobot-format dataset:
- Latent `.pth` files contain pre-encoded video features, text embeddings, frame metadata, and action segments.
- Actions are normalized using quantile statistics (`action_norm_method = 'quantiles'`).
- Supports multi-dataset loading with parallel initialization via `multiprocessing.Pool`.

#### `wan_va/configs/`
EasyDict-based configuration hierarchy:

| File | Purpose |
|---|---|
| `shared_config.py` | Base settings: host/port, dtype (`bfloat16`), patch size `(1,2,2)` |
| `va_robotwin_cfg.py` | RoboTwin eval config: 256×320 resolution, 30-dim actions, 3 cameras, guidance scales, normalization stats |
| `va_robotwin_train_cfg.py` | Training overrides: learning rate, batch size, FSDP, W&B |
| `va_robotwin_i2va.py` | Image-to-video-action (i2va) inference config |
| `va_franka_cfg.py` / `va_franka_i2va.py` | Franka robot variant configs |
| `va_demo_cfg.py` / `va_demo_i2va.py` | Demo / quick-test configs |

#### `wan_va/utils/`
Shared utilities:
- `FlowMatchScheduler` — flow-matching noise schedule with configurable SNR shift.
- `data_seq_to_patch` — reshapes flat latent sequences into spatial patch grids.
- `run_async_server_mode` — async WebSocket server loop.
- `save_async` — non-blocking checkpoint saving.
- `sample_timestep_id` — samples diffusion timesteps during training.

#### `wan_va/distributed/`
- `fsdp.py` — `shard_model` (FSDP2 wrapping with uniform bfloat16 dtype) and `apply_ac` (activation checkpointing per transformer block).
- `util.py` — `init_distributed`, `_configure_model`, `dist_mean`, `dist_max`.

---

### 2. `models/` — Pretrained Weights

A symlink, points to the downloaded Wan2.2 pretrained model components:

```
models/
├── transformer/        # WanTransformer3DModel weights + config.json (contains attn_mode)
├── vae/                # Wan2.2 VAE encoder/decoder
├── text_encoder/       # T5-based text encoder
└── tokenizer/          # T5 tokenizer
```

> **Important**: `transformer/config.json` must have `attn_mode = "flex"` for training and `"torch"` or `"flashattn"` for inference.

---

### 3. `evaluation/robotwin/` — RoboTwin Evaluation Client

Utils for launching evaluation servers and clients that connects to the RoboTwin 2.0 simulation environment.

| File | Purpose |
|---|---|
| `eval_polict_client_openpi.py` | Main eval loop: instantiates a RoboTwin task env, sends observations to the server, executes returned actions, records success/failure |
| `websocket_client_policy.py` | `WebsocketClientPolicy` — thin WebSocket client that serializes observations (msgpack-numpy) and deserializes action responses |
| `launch_server.sh` / `launch_server_multigpus.sh` | Shell scripts to start the inference server (single or multi-GPU) |
| `launch_client.sh` / `launch_client_multigpus.sh` | Shell scripts to run evaluation for one task or a group of tasks across GPUs |
| `launch_ood_eval.sh` | Runs the 10 OOD (out-of-distribution) evaluation tasks |
| `calc_stat.py` | Aggregates per-task success rates from result JSON files |
| `geometry.py` | Rotation/pose utilities (`euler2quat`, etc.) |
| `msgpack_numpy.py` | msgpack serialization with numpy array support |
| `test_render.py` | `Sapien_TEST` — headless SAPIEN renderer for generating starting-state images |

---

### 4. `robotwin/` — RoboTwin Simulation (symlink)

Symlink to the RoboTwin 2.0 repository. Provides:

- **Task environments** (`envs/<task_name>.py`): SAPIEN-based simulation scenes with scripted expert planners. Each task defines object placement, success criteria, and a `play_once()` method that generates demonstration data.
- **Base task** (`envs/_base_task.py`): shared robot control primitives — `grasp_actor`, `move`, `together_move_to_pose`, `back_to_origin`.
- **Instruction JSONs** (`description/task_instruction/<task_name>.json`): natural language instruction templates with placeholders filled from `play_once()` info dicts.
- **Step limits** (`task_config/_eval_step_limit.yml`): per-task maximum action steps for evaluation.
- **Asset downloader** (`script/_download_assets.sh`): fetches 3-D object meshes and textures.

---

### 5. `scripts/` — Utility Scripts

| File | Purpose |
|---|---|
| `generate_starting_images.py` | Renders 50 RoboTwin scenes and saves 3-camera PNGs + metadata for the arm-distinction evaluation |
| `run_arm_eval.py` | Batch i2va evaluation: loads the model once, iterates through all samples, generates video+actions for left-arm and right-arm instruction variants |
| `run_arm_eval.sh` | Shell wrapper for `run_arm_eval.py` (supports `NGPU=N`) |
| `run_launch_va_server_sync.sh` | Launches the server in synchronous i2va mode |
| `run_va_posttrain.sh` | Launches distributed post-training with `NGPU=8` |

---

### 6. `robotwin/script/` — RoboTwin Installation Scripts

Scripts used during RoboTwin environment setup (not part of the model):
- `_install.sh` — installs RoboTwin Python dependencies.
- `_download_assets.sh` — downloads simulation assets.
- `requirements.txt` — RoboTwin-specific pip requirements.

---

### 7. `docs/` — Documentation

| File | Purpose |
|---|---|
| `SETUP_DOCKER.md` | Docker-based installation guide (recommended) |
| `SETUP_LOCAL.md` | Local installation guide |
| `EVAL_TASKS.md` | List of the 50 standard RoboTwin evaluation tasks |
| `CREATE_OOD_TASK.md` | Guide for adding new OOD evaluation tasks |
| `ARCHITECTURE.md` | This file |

---

## Action Space

All actions are 30-dimensional, structured as:

| Dims | Description |
|---|---|
| 0–6 | Left arm end-effector pose (xyz + quaternion) |
| 7–13 | Right arm end-effector pose (xyz + quaternion) |
| 14–20 | Left arm joint angles (7 DOF) |
| 21–27 | Right arm joint angles (7 DOF) |
| 28 | Left gripper (open/close) |
| 29 | Right gripper (open/close) |

During RoboTwin evaluation, only the EEF + gripper channels (dims 0–13, 28–29) are used (`used_action_channel_ids`).

---

## Inference Flow (Step by Step)

1. **Client** (`eval_polict_client_openpi.py`) resets the RoboTwin environment and captures 3 camera images.
2. Client sends `{images, instruction}` to the **server** via WebSocket.
3. **Server** (`wan_va_server.py`):
   a. Encodes images with the VAE encoder → latent tokens.
   b. Encodes the instruction with the text encoder → text embeddings.
   c. Runs the AR diffusion loop: for each frame chunk, denoises video latents and action tokens jointly using `WanTransformer3DModel`.
   d. Decodes video latents back to frames with the VAE decoder.
   e. Returns predicted frames and raw action arrays.
4. Client de-normalizes actions and sends them to the robot controller.
5. Steps 1–4 repeat until the task succeeds or the step limit is reached.

---

## Training Flow (Post-Training)

1. Pre-extract VAE latents and text embeddings offline → store as `.pth` files.
2. `MultiLatentLeRobotDataset` loads latent files and normalizes actions.
3. `Trainer` samples a batch, adds flow-matching noise at a random timestep, and runs a forward pass through `WanTransformer3DModel`.
4. Loss = MSE between predicted denoised latents/actions and ground truth.
5. Gradients are accumulated and applied with AdamW; model is sharded across GPUs via FSDP2.
6. Checkpoints are saved in SafeTensors format every `save_interval` steps.
