# LingBot-VA Setup Guide (Docker, Recommended)

This guide documents the full setup process for running LingBot-VA inference with RoboTwin 2.0 using a **docker image**. 
The original README omits several non-obvious steps that are required to get the system working. Follow this guide in order.

## Prerequisites

- CUDA GPU(s)
- Base Ubuntu Image w/ CUDA 12.6 + matching CUDA toolkit headers

## 1. Build the Image and Mount the Project
This is straight forward. Find [docker file](../Dockerfile), and pass it to whatever build toolkit you use.
After this please mount your `lingbot-va` and `RoboTwin` project to the PVC of your docker.

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

### Infinite Looped Error in cuRobo
Try re-install curobo. If this does not work, delete `/path/to/RoboTwin/envs/curobo`, re-clone it and re-build it with:
```bash
pip install -e . --no-build-isolation
```

> Note: If you encounter this when running custom tasks, please test if your task is valid. Refence [related doc](./CREATE_OOD_TASK.md) for more details.

### Hard Coded Path in RoboTwin installation
If you migrate your local project to cloud by simply copy it, you will probably see file not found errors when using RoboTwin. 
This is intended behavior of RoboTwin, you should fix the path everytime you move the RoboTwin codebase to a new environment by running:
```bash
python /path/to/RoboTwin/script/update_embodiment_config_path.py
```

### The Client Freezes Without Error Message
Check the server starting log. Are clients posting requests to correct **ports**? For example, after launching the server you will probably see this:
```bash
root@node007:/data/job/lingbot-va# bash evaluation/robotwin/launch_server_multigpus.sh
[Task ] GPU: 0 | PORT: 49556 | MASTER_PORT: 49661 | Log: ./logs/server_0_20260312_122839.log
[Task ] GPU: 1 | PORT: 49557 | MASTER_PORT: 49662 | Log: ./logs/server_1_20260312_122839.log
All 2 instances have been launched in the background.
```
Then your client should listen to `49556` and `49557`, not `49661` or `49662`!

