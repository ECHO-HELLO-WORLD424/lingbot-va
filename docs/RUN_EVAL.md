# Run Evaluation With RoboTwin 2.0

## Prerequisites

Please set up your working dir following either [docker set up guide](./SETUP_DOCKER.md) (recommended) or [virtual env based set up guide](./SETUP_VENV.md)

## Key Files

These are the key files for evaluating `lingbot-va` in a simulated environment:

- `evaluation/robotwin/launch_server_multigpus.sh`: This script launches the `wan_va` model backend server.
- `evaluation/robotwin/launch_client_multigpus.sh`: This script is the client of `wan_va` server. It connects the model to RoboTwin
- `evaluation/robotwin/launch_ood_eval.sh`: This is specifically designed for out-of-distribution test. It 

## Run Evaluation

In terminal 1: 

```bash
NGPU=4 bash evaluation/robotwin/launch_server_multigpus.sh
```

The log will be saved to `./logs` by default. Note that you might want to edit the script and change the GPU count to your actual GPU count.

In terminal 2:

```bash
bash evaluation/robotwin/launch_ood_eval.sh
```

The log will be saved to `./logs` by default. Note that you might want to modify these accordingly:

```bash
start_port=29556 # Line 17. This should be the same as your server's starting port
num_gpus=8 # Change accordingly.
```

Then results will be saved to `./results` and generated video will appears in `./visualization/real`
To analyze raw accuracy based on the `.log` files in `/path/to/lingbot-va/logs`, run the python filter script:

```bash
python evaluation/robotwin/filter_logs.py > filter_logs.txt
```

## Trouble Shooting

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