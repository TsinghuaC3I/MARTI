#!/usr/bin/env bash
set -x
set -euo pipefail
eval $(curl -s http://deploy.i.h.pjlab.org.cn/infra/scripts/nccl_auto_config.py | python3 - --shell-export)
# 验证
echo "NCCL_IB_HCA=$NCCL_IB_HCA"
echo "NCCL_IB_GID_INDEX=$NCCL_IB_GID_INDEX"
# source /mnt/shared-storage-user/marti/lipengfei/miniconda3/etc/profile.d/conda.sh
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
# 配置部分
# ------------------------------
# 写对应gpfs地址 
# 用于 落盘 head地址，收集日志
MASTER_PORT=6379
DASHBOARD_PORT=8265
COMMAND="${COMMAND:-bash}"
SCRIPT="${SCRIPT:-/mnt/shared-storage-user/marti/OpenRLHF/examples/scripts-lpf/train_multi_gspo_qwen3_ray_hybrid_engine_agent_fyk_qwen_8B_areal_8B.sh}"
TIMESTAMP=$(date +"%Y%m%d_%H%M")

SHARED_DIR="/mnt/shared-storage-user/marti/lipengfei/multigputest/multi_agent_test"
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
# Head 节点启动
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
# Worker 节点启动
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
    # worker ready 落盘
    WORKER_READY_FILE="$SHARED_DIR/ray_worker_ready_${my_ip}"
    touch "$WORKER_READY_FILE"
    log "Worker ready file created: $WORKER_READY_FILE"
}
# ------------------------------
# 等待所有节点就绪
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
# 执行 entry command
# ------------------------------
execute_entry_command() {
    # ray_address=$1
    log "Executing: $COMMAND $SCRIPT"
    $COMMAND -exc "$SCRIPT"
    log "Entry command finished with exit code $?"
}
# ------------------------------
# 主逻辑
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
    # Worker 逻辑
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
    # 阻塞等待任务完成
    # -----------------
    # 方式 1：通过文件信号等待 master 结束
    while [[ ! -f "$MASTER_DONE_FILE" ]]; do
        sleep 1
    done
    log "Worker job finished successfully"
fi
