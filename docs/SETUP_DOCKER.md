# LingBot-VA Setup Guide (Docker, Recommended)

This guide documents the full setup process for running LingBot-VA inference with RoboTwin 2.0 using a **docker image**. 
The original README omits several non-obvious steps that are required to get the system working. Follow this guide in order.

## Prerequisites

- CUDA GPU(s)
- Base Ubuntu Image w/ CUDA 12.6 + matching CUDA toolkit headers

## 1. Build the Image and Mount the Project

This is straight forward. Find [docker file](../Dockerfile), and pass it to whatever build toolkit you use.
RoboTwin is included as a git submodule under `robotwin/`. Clone the repo with submodules and mount only `lingbot-va`:

```bash
git clone --recurse-submodules https://github.com/Robbyant/lingbot-va.git
```

If you already cloned without `--recurse-submodules`, run:

```bash
git submodule update --init --recursive
```

Then mount your `lingbot-va` directory to the PVC of your docker.

> Note: To let the lingbot-va find RoboTwin installation properly you want to set `ROBOTWIN_ROOT` as the **absolute path** to your RoboTwin installation.

## 2. Build cuRobo Manually

The codebase of cuRobo is not included in the Dockerfile so you need to build it manually. Do it by "`cd`" into a running docker and execute the following:

```bash
cd /path/to/lingbot-va/robotwin/envs
git clone https://github.com/NVlabs/curobo.git # If there's `directory not empty` warning, ignore it.
cd curobo
uv pip install -e . --no-build-isolation
cd ../../..
```

## 3. Post Installation Steps

### Download the Model

Download the post-trained RoboTwin checkpoint into the lingbot-va repo:

```bash
huggingface-cli download robbyant/lingbot-va-posttrain-robotwin \
    --local-dir /path/to/lingbot-va/models/lingbot-va-posttrain-robotwin
```

### Configure the Model Path

Edit `wan_va/configs/va_robotwin_cfg.py` line 9 and set the path to the downloaded model:

```python
va_robotwin_cfg.wan22_pretrained_model_name_or_path = \
    "/path/to/lingbot-va/models/lingbot-va-posttrain-robotwin"
```

### Set `attn_mode` for Inference

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

### Download RoboTwin Assets
You need to download RoboTwin Assets before using any of them. `cd` to your RoboTwin installation and run:

```bash
bash script/_download_assets.sh
```

### Audit Hard-Coded Path In RoboTwin Installation:
You might see file not found errors when using RoboTwin. This is a common issue, you should fix the path everytime you move the RoboTwin codebase to a new environment by running:

```bash
python /path/to/RoboTwin/script/update_embodiment_config_path.py
```

## 3. Trouble Shooting

There are some known issues with *some* building toolkits when using this Dockerfile. Here's the trouble shooting guide:

### Broken Type Check in OpenCV

If you encounter error about `LayerId` when importing `cv2`, open `/usr/local/lib/python3.10/dist-packages/cv2/typing/__init__.py` and comment out this line:

```python
# Change this:
LayerId = cv2.dnn.DictValue

# To this
# LayerId = cv2.dnn.DictValue
```

### Missing `pkg_resources`

Install it manually via

```bash
apt-get update && apt install python3-setuptools
```

### Can't Import `flash-attn`:

Try re-installing via:

```bash
pip install flash-attn --no-build-isolation
```

If this does not work, go to `/path/to/lingbot-va/models/transformer/config.json` and change `attn_mode` to `torch`.
