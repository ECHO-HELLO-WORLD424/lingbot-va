# LingBot-VA Setup Guide (uv, Potentially Tricky)

This guide documents the full setup process for running LingBot-VA inference with RoboTwin 2.0
using a **uv-managed virtual environment**. The original README omits several non-obvious steps
that are required to get the system working. Follow this guide in order.

> If you want to use docker, refer to [docker setup guide](./SETUP_DOCKER.md)
> Note: Since you are in a venv you need to replace all `python ...` command in the shell scripts to `.venv/bin/python

---

## Prerequisites

- Python 3.10.16 (exact)
- CUDA 12.6 + matching CUDA toolkit headers (required for cuRobo compilation)
- Vulkan drivers (required by the SAPIEN simulator)
- [uv](https://docs.astral.sh/uv/) installed

Install Vulkan drivers if not already present:

```bash
sudo apt install libvulkan1 mesa-vulkan-drivers vulkan-tools
```

---

## 1. Clone, Setup and Link RoboTwin

The eval client imports RoboTwin's `envs` package directly into the lingbot-va Python process,
so RoboTwin must live at `robotwin/` inside the lingbot-va repo root. The recommended approach
is by using the git submodule, which points to a fork of RoboTwin:

```bash
git clone --recurse-submodules https://github.com/Robbyant/lingbot-va.git
```

If you already cloned without `--recurse-submodules`, run:

```bash
git submodule update --init --recursive
```

Then mount your `lingbot-va` directory to the PVC of your docker.

> Note: To let the lingbot-va find RoboTwin installation properly you want to set `ROBOTWIN_ROOT` as the **absolute path** to your RoboTwin installation.

Now, create a virtual env with uv **in the RoboTwin repo**, follow the below setup guide:

1. modify script/requirements.txt 
   
   ```bash
   transforms3d==0.4.2
   sapien==3.0.0b1
   scipy==1.10.1
   mplib==0.2.1
   gymnasium==0.29.1
   trimesh==4.4.3
   open3d==0.18.0
   imageio==2.34.2
   pydantic
   zarr
   openai
   huggingface_hub==0.36.2
   h5py
   # For Description Generation
   azure==4.0.0
   azure-ai-inference
   pyglet<2
   wandb
   moviepy
   imageio
   termcolor
   av
   matplotlib
   ffmpeg
   ```

2. modify line 8 of script/_install.sh:
   
   ```bash
   uv pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation
   ```
   
   > **Slow build:** `pytorch3d` can take from 30 minutes to 1h to build on RTX-4090 + i7-14700KF

3. run:
   
   ```bash
   bash script/_install.sh
   ```

4. run:
   
   ```bash
   bash script/_download_assets.sh
   ```

After this, you are good with RoboTwin. `cd` back to your `lingbot-va` repo and proceed with the below instructions.

---

## 2. Create the uv Virtual Environment for Lingbot-VA

```bash
cd /path/to/lingbot-va
uv venv --python 3.10.16
source .venv/bin/activate
```

This creates `.venv/` at the repo root. All shell scripts in this repo are already configured to
use `.venv/bin/python` automatically — no manual activation is needed when running the scripts.

---

## 3. Install lingbot-va Dependencies

```bash
uv pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
    --index-url https://download.pytorch.org/whl/cu126

uv pip install websockets einops diffusers==0.36.0 transformers==4.55.2 \
    accelerate msgpack opencv-python matplotlib ftfy easydict \
    safetensors Pillow imageio[ffmpeg] tqdm scipy

uv pip install flash-attn --no-build-isolation
```

> **Slow build:** `flash-attn` can take from 60min to 90min to build on RTX-4090 + i7-14700KF

---

## 4. Install RoboTwin Dependencies into the lingbot-va venv

Because the eval client runs RoboTwin's simulation code inside the same Python process as
lingbot-va, **all RoboTwin dependencies must also be installed into the lingbot-va venv** —
not just into a separate RoboTwin environment.

```bash
uv pip install -r robotwin/script/requirements.txt
```

> **Note:** `open3d==0.18.0` (listed in that file) is a large package. It is required at import
> time by `robotwin/envs/camera/camera.py` and cannot be skipped. Installed it manually if it is missing

### 4a. Patch sapien and mplib

RoboTwin requires two small patches to installed packages. Run these from the lingbot-va repo
root (the `uv pip show` commands resolve the correct site-packages path automatically):

```bash
# Patch sapien: add encoding="utf-8" and fix .srdf extension
SAPIEN_LOCATION=$(uv pip show sapien | grep 'Location' | awk '{print $2}')/sapien
sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' \
    "$SAPIEN_LOCATION/wrapper/urdf_loader.py"

# Patch mplib: remove spurious collision check in screw planner
MPLIB_LOCATION=$(uv pip show mplib | grep 'Location' | awk '{print $2}')/mplib
sed -i -E \
    's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' \
    "$MPLIB_LOCATION/planner.py"
```

### 4b. Build and Install pytorch3d from Source

The `@stable` tag of pytorch3d on PyPI is built against an older PyTorch ABI and **will fail
to import** with torch 2.9.0 despite installing without errors. You must build from source
against the exact torch version in the venv:

```bash
uv pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" \
    --no-build-isolation
```

> **Note:** "missing pytorch3d" printed at client startup is a **harmless warning** from
> `robotwin/envs/camera/camera.py`. The `fps()` function that requires pytorch3d is only called
> when `pcd_down_sample_num > 0` and point cloud observations are requested. Since LingBot-VA
> uses only RGB camera inputs, this code path is never reached and the warning can be ignored.

### 4c. Build and Install cuRobo

cuRobo is not available on PyPI — it must be compiled from source. It is required by the
`aloha-agilex` embodiment (the default in `demo_clean.yml`) for motion planning during both
the expert-validation pass and policy inference.

```bash
cd /path/to/lingbot-va/robotwin/envs
git clone https://github.com/NVlabs/curobo.git # If there's `directory not empty` warning, ignore it.
cd curobo
uv pip install -e . --no-build-isolation
cd ../../..
```

> **Slow build:** cuRobo takes about 30min to build on RTX-4090 + i7-14700KF 

---

## 5. Download the Model

Download the post-trained RoboTwin checkpoint into the lingbot-va repo:

```bash
huggingface-cli download robbyant/lingbot-va-posttrain-robotwin \
    --local-dir /path/to/lingbot-va/models/lingbot-va-posttrain-robotwin
```

Do not use the default HuggingFace cache location (`~/.cache/huggingface`) — the model path
must be set explicitly in the config (see next step).

---

## 6. Configure the Model Path

Edit `wan_va/configs/va_robotwin_cfg.py` line 9 and set the path to the downloaded model:

```python
va_robotwin_cfg.wan22_pretrained_model_name_or_path = \
    "/path/to/lingbot-va/models/lingbot-va-posttrain-robotwin"
```

---

## 7. Set `attn_mode` for Inference

The model's `transformer/config.json` ships with `"attn_mode": "flex"`, which is required for
training but **causes errors at inference time**. You must change it before running the server:

```bash
# Open the file and change "attn_mode": "flex"
# to either "torch" or "flashattn" (if flash-attn is installed)
nano /path/to/lingbot-va/models/lingbot-va-posttrain-robotwin/transformer/config.json
```

| Mode      | `attn_mode` value          |
| --------- | -------------------------- |
| Training  | `"flex"`                   |
| Inference | `"torch"` or `"flashattn"` |

---

## 8. Running Inference

### 8a. Launch the Inference Server

Run from the lingbot-va repo root. The server listens on port 29056 by default.

```bash
cd /path/to/lingbot-va
bash evaluation/robotwin/launch_server.sh
```

Wait until the server prints a ready message before starting the client. The server requires
approximately **24 GB VRAM** (VAE and text encoder are offloaded to CPU automatically).

### 8b. Launch the Eval Client

Open a second terminal and run from the lingbot-va repo root:

```bash
cd /path/to/lingbot-va
task_name="adjust_bottle"
save_root="results/"
bash evaluation/robotwin/launch_client.sh ${save_root} ${task_name}
```

Replace `adjust_bottle` with any of the 50 RoboTwin task names. Results (per-episode videos
and JSON success metrics) are saved under `robotwin/${save_root}`.

> **Slow warmup** on **first run** cuRobo performs a `warmup()` call that JIT-compiles
> further CUDA kernels. This causes the client to appear frozen for around 10 minutes after
> printing the config block — this is normal. Do not interrupt it.

> **Note:** An `eval_result/` folder is also created inside the RoboTwin directory — this is
> native RoboTwin output and duplicates the contents of `results/`. It can be ignored.

---

## 9. Server Address Configuration

By default the client connects to `ws://127.0.0.1:29056` (localhost). If you run the server
on a different machine or need a different host, edit
`evaluation/robotwin/websocket_client_policy.py` line 18:

```python
def __init__(self, host: str = "127.0.0.1", ...):
```

Change `"127.0.0.1"` to the server's IP address or hostname.

> **Important:** Do not use `"0.0.0.0"` as the client host. While `0.0.0.0` is a valid bind
> address for the server (meaning "listen on all interfaces"), it is **not** a valid destination
> address for outgoing connections — Linux will immediately close the connection, producing a
> `websockets.exceptions.InvalidMessage` error.

---

## 10. Optional: Real-Time Viewer

By default all configs run headless (`render_freq: 0`). To open a live SAPIEN viewer window
during policy inference, edit `robotwin/task_config/demo_clean.yml`:

```yaml
render_freq: 5   # refresh viewer every 5 simulation steps (0 = headless)
```

A `$DISPLAY` environment variable pointing to a running X session must be set.

---

## Troubleshooting

| Symptom                                                                       | Cause                                                                  | Fix                                                                                                                       |
| ----------------------------------------------------------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `ImportError: cannot import name 'CONFIGS_PATH' from 'envs'`                  | Python imports the unrelated PyPI `envs` package instead of RoboTwin's | Ensure `robotwin/` symlink exists at repo root; the eval client uses `Path(__file__).resolve()` to build an absolute path |
| `ModuleNotFoundError: No module named 'open3d'`                               | RoboTwin deps not installed in lingbot-va venv                         | `uv pip install -r robotwin/script/requirements.txt`                                                                      |
| `ImportError: cannot import name 'CuroboPlanner'`                             | cuRobo not installed                                                   | Build and install cuRobo from source (Step 4c)                                                                            |
| `websockets.exceptions.InvalidMessage: did not receive a valid HTTP response` | Client connecting to `0.0.0.0`                                         | Change client host to `127.0.0.1` or the server's actual IP (Step 10)                                                     |
| Client appears frozen after printing config                                   | cuRobo first-run CUDA kernel warmup                                    | Wait 3–10 minutes; this is normal on first launch                                                                         |
| `ImportError: /path/to/pytorch3d/_C.cpython-310...so: undefined symbol`       | pytorch3d built against wrong PyTorch ABI                              | Rebuild pytorch3d from source with `--no-build-isolation` (Step 4b)                                                       |
| `"flex" attn_mode` error at server startup                                    | `transformer/config.json` not updated                                  | Change `attn_mode` to `"torch"` or `"flashattn"` (Step 7)                                                                 |
