#!/usr/bin/env bash
set -x
set -euo pipefail
source /mnt/shared-storage-user/marti/miniconda3/etc/profile.d/conda.sh
conda activate marti_vllm
which conda
# ray stop

# ---------- environment env -----------
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
# export NCCL_DEBUG=WARN
export RAY_PICKLE_VERBOSE_DEBUG=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
# export RAY_ENABLE_RECORD_ACTOR_TASK_LOGGING=1
# export TORCH_COMPILE_CACHE_DIR="/mnt/shared-storage-user/marti/lipengfei/.cache/torch_compile"
export NCCL_DEBUG=INFO
# export VLLM_ALLOW_INSECURE_SERIALIZATION=1

# ---------- Ray Configuration ----------
# ------------------------------
# config
# ------------------------------
MASTER_PORT=6379
DASHBOARD_PORT=8265
COMMAND="${COMMAND:-bash}"
SCRIPT="${SCRIPT:-/mnt/shared-storage-user/marti/OpenRLHF/examples/mas-scripts/train_multi_agent_ray_hybrid_engine.sh}"
TIMESTAMP=$(date +"%Y%m%d_%H%M")

SHARED_DIR=${ROOT_DIR}
mkdir -p "${SHARED_DIR}"

MASTER_ADDR_FILE="$SHARED_DIR/master_addr.txt"
READY_FLAG_FILE="$SHARED_DIR/ray_head_ready"
MASTER_DONE_FILE="$SHARED_DIR/flag"
# WORKER_READY_FILE="$SHARED_DIR/ray_worker_ready_${TIMESTAMP}.txt"
RANK=${KUBEBRAIN_REPLICA:-0}
WORLD_SIZE=${KUBEBRAIN_REPLICA_TOTAL:-1}
log() {
    echo "[multi_node_rjob.sh] $*" >&2
}
get_my_ip() {
    hostname -i
}
# ------------------------------
# Head
# ------------------------------
# ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265 --include-dashboard=true
start_ray_head() {
    local my_ip
    my_ip=$(get_my_ip)
    log "Starting Ray head on $my_ip ..."
    echo "$my_ip" > "$MASTER_ADDR_FILE"
    ray start --head \
        --port="$MASTER_PORT" \
        --node-ip-address="$my_ip" \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=8265 \
        --disable-usage-stats \
        # --temp-dir="${RAY_TMPDIR}" \

        # --include-dashboard=true \
    touch "$READY_FLAG_FILE"
    log "Ray head started at $my_ip:$MASTER_PORT"
    lsof -i:8265
    export RAY_ADDRESS="$my_ip:$DASHBOARD_PORT"
    echo "$my_ip"
    # ray status
}
# ------------------------------
# Worker
# ------------------------------
connect_ray_worker() {
    local master_addr=$1
    local my_ip
    my_ip=$(get_my_ip)
    log "Connecting Ray worker to $master_addr from $my_ip ..."
    ray start \
        --address="$master_addr" \
        --node-ip-address="$my_ip" \
        --disable-usage-stats \
        # --num-gpus=2 \
    log "Ray worker started"
    WORKER_READY_FILE="$SHARED_DIR/ray_worker_ready_${my_ip}"
    touch "$WORKER_READY_FILE"
    log "Worker ready file created: $WORKER_READY_FILE"
}
# ------------------------------
# waiting all node ready
# ------------------------------
wait_for_workers() {
    local expected_workers=$((WORLD_SIZE - 1))
    local timeout=${1:-300}
    local start=$(date +%s)

    log "Waiting for $expected_workers workers to connect..."
    while true; do
        local now=$(date +%s)
        if (( now - start > timeout )); then
            log "Timeout waiting for workers"
            return 1
        fi
        local count
        count=$(ls "$SHARED_DIR"/ray_worker_ready_* 2>/dev/null | wc -l || true)
        log "Detected $count / $expected_workers workers ready"
        if (( count >= expected_workers )); then
            return 0
        fi
        sleep 5
    done
}
# ------------------------------
# entry command
# ------------------------------
execute_entry_command() {
    # ray_address=$1
    log "Executing: $COMMAND $SCRIPT"
    $COMMAND -exc "$SCRIPT"
    log "Entry command finished with exit code $?"
}
# ------------------------------
# main logic
# ------------------------------
log "RANK=$RANK, WORLD_SIZE=$WORLD_SIZE"
log "COMMAND=$COMMAND"
log "SCRIPT=$SCRIPT"

if [[ "$RANK" -eq 0 ]]; then
    MASTER_ADDR=$(start_ray_head)
    echo "master TIMESTAMP is :${TIMESTAMP}"
    if wait_for_workers 600; then
        log "All workers ready. Running main script."
        ray status
        # echo "job submmit address: ${RAY_ADDRESS}"
        execute_entry_command
	log "Master script finished successfully"
	touch "$MASTER_DONE_FILE"
    else
        log "Not all workers ready, exiting"
        exit 1
    fi
else
    log "Waiting for master address file..."
    echo "worker TIMESTAMP is :${TIMESTAMP}"
    for i in {1..120}; do
        if [[ -f "$MASTER_ADDR_FILE" ]]; then
            MASTER_ADDR=$(cat "$MASTER_ADDR_FILE")
            log "Got master address: $MASTER_ADDR"
            break
        fi
        sleep 1
    done
    if [[ -z "${MASTER_ADDR:-}" ]]; then
        log "Timed out waiting for master"
        exit 1
    fi
    connect_ray_worker "$MASTER_ADDR:$MASTER_PORT"
    sleep 1
    ray status
    log "Worker connected. Waiting for tasks..."

    # -----------------
    # waiting for task finish
    # -----------------
    while [[ ! -f "$MASTER_DONE_FILE" ]]; do
        sleep 1
    done
    log "Worker job finished successfully"
fi
