#!/bin/bash
# Evaluate the 10 OOD composed tasks on multiple GPUs.
# Usage: bash launch_ood_eval.sh [save_root] [seed] [test_num]
export LD_LIBRARY_PATH=/usr/lib64:/usr/lib:$LD_LIBRARY_PATH

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"

save_root=${1:-'./results'}
seed=${2:-0}
test_num=${3:-100}

policy_name=ACT
task_config=demo_clean
train_config_name=0
model_name=0
start_port=29556
num_gpus=8

task_names=(
  "handover_then_place_phone_stand"
  "stack_then_scan"
  "shake_then_place_bottle"
  "rotate_qrcode_then_scan"
  "open_laptop_then_place_object_inside"
  "dump_bin_then_sort_by_color"
  "press_stapler_while_holding"
  "unpack_then_rank"
  "place_dual_shoes_then_hang_mug"
  "fill_then_shake_then_move_to_pot"
)

log_dir="./logs"
mkdir -p "$log_dir"

echo -e "\033[32mLaunching ${#task_names[@]} OOD tasks. GPUs assigned by mod ${num_gpus}, ports starting from ${start_port}.\033[0m"

pid_file="ood_pids.txt"
> "$pid_file"

batch_time=$(date +%Y%m%d_%H%M%S)

for i in "${!task_names[@]}"; do
    task_name="${task_names[$i]}"
    gpu_id=$(( i % num_gpus ))
    port=$(( start_port + i ))

    export CUDA_VISIBLE_DEVICES=${gpu_id}

    log_file="${log_dir}/${task_name}_${batch_time}.log"

    echo -e "\033[33m[Task $i] Task: ${task_name}, GPU: ${gpu_id}, PORT: ${port}, Log: ${log_file}\033[0m"

    PYTHONWARNINGS=ignore::UserWarning \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python -m evaluation.robotwin.eval_polict_client_openpi --config policy/$policy_name/deploy_policy.yml \
        --overrides \
        --task_name ${task_name} \
        --task_config ${task_config} \
        --train_config_name ${train_config_name} \
        --model_name ${model_name} \
        --ckpt_setting ${model_name} \
        --seed ${seed} \
        --policy_name ${policy_name} \
        --save_root ${save_root} \
        --video_guidance_scale 5 \
        --action_guidance_scale 1 \
        --test_num ${test_num} \
        --port ${port} > "$log_file" 2>&1 &

    pid=$!
    echo "${pid}" | tee -a "$pid_file"
done

echo -e "\033[32mAll OOD tasks launched. PIDs saved to ${pid_file}\033[0m"
echo -e "\033[36mTo terminate all processes, run: kill \$(cat ${pid_file})\033[0m"
