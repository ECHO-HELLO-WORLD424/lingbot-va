START_PORT=${START_PORT:-29056}
MASTER_PORT=${MASTER_PORT:-29061}

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"

save_root='visualization/'
mkdir -p $save_root

"$PYTHON" -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port $MASTER_PORT \
    wan_va/wan_va_server.py \
    --config-name robotwin \
    --port $START_PORT \
    --save_root $save_root


