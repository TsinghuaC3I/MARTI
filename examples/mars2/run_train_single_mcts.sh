#!/bin/bash
# gspo + tis + datafilter + overlong
set -x
source /mnt/shared-storage-user/marti/miniconda3/etc/profile.d/conda.sh
conda activate marti_vllm
which conda
which python
cd /mnt/shared-storage-user/marti/MARTI-v2

# basic config
MODEL_DIR="/mnt/shared-storage-user/marti/models"
SHORT_NAME=${1:-"Qwen3-8B"}
# SHORT_NAME=${1:-"Qwen3-4B-Instruct-2507"}
PRETRAIN="${MODEL_DIR}/${SHORT_NAME}"
PROMPT_MAX_LEN=4096
GENERATE_MAX_LEN=32768
EVAL_GENERATE_MAX_LEN=32768
OVERLONG_BUFFER_LEN=2048
MAX_LEN=40000
ADVANTAGE="group_norm"

ROOT_DIR="/mnt/shared-storage-user/marti/MARTI-v2"
TASK="CODE_8B_FILTER_FT"
PROMPT_DATA="json@/mnt/shared-storage-user/marti/lipengfei/MARTI_DEV/data/${TASK}"

# Workflow config
MCTS_NODES=8 # 8
NUM_TASKS=128 # 128
EXP=all_tricks

WORKFLOW_SAVE_PATH="${ROOT_DIR}/outputs/workflow/${ADVANTAGE}-${SHORT_NAME}-${TASK}-db-${EXP}"

TENSORBOARD="${ROOT_DIR}/logs/tensorboard/${ADVANTAGE}-${SHORT_NAME}-${TASK}-db-${EXP}"
LOG_DIR=/mnt/shared-storage-user/marti/MARTI-v2/logs/${ADVANTAGE}-${SHORT_NAME}-${TASK}-db-${EXP}.log

# set port and env var
export MASTER_PORT=$(shuf -i 10000-65535 -n 1)
export OPENRLHF_ASYNC_NUM_TASKS=${NUM_TASKS}

# default agent config
# DEFAULT_AGENT="{
    # \"is_reasoning_model\": true
# }"
DEFAULT_AGENT="{
    \"is_reasoning_model\": false
}"

# Workflow additional param config
WORKFLOW_ARGS="{
    \"max_num_nodes\": ${MCTS_NODES},
    \"eval_max_num_nodes\": 1,
    \"save_path\": \"${WORKFLOW_SAVE_PATH}\",
    \"algo\": {
        \"class_name\": \"AsyncABMCTSA\",
        \"params\": {}
    }
}"

AGENT0="{
    \"0\": {
        \"role\": \"generator\",
        \"pretrain\": \"${PRETRAIN}\",
        \"save_path\": \"/mnt/shared-storage-user/marti/MARTI-v2/outputs/final/${ADVANTAGE}-${SHORT_NAME}-${TASK}-db-${EXP}-agent0\",
        \"ckpt_path\": \"/mnt/shared-storage-user/marti/MARTI-v2/outputs/ckpt/${ADVANTAGE}-${SHORT_NAME}-${TASK}-db-${EXP}-agent0\",
        \"is_tuning\": true
    }
}"

export NCCL_DEBUG=WARN

mkdir -p "${ROOT_DIR}/logs"
mkdir -p "${WORKFLOW_SAVE_PATH}"
mkdir -p "${TENSORBOARD}"
mkdir -p "/mnt/shared-storage-user/marti/MARTI-v2/outputs/final/${ADVANTAGE}-${SHORT_NAME}-${TASK}-db-${EXP}-agent0"
mkdir -p "/mnt/shared-storage-user/marti/MARTI-v2/outputs/final/${ADVANTAGE}-${SHORT_NAME}-${TASK}-db-${EXP}-agent1"
mkdir -p "/mnt/shared-storage-user/marti/MARTI-v2/outputs/ckpt/${ADVANTAGE}-${SHORT_NAME}-${TASK}-db-${EXP}-agent0"
mkdir -p "/mnt/shared-storage-user/marti/MARTI-v2/outputs/ckpt/${ADVANTAGE}-${SHORT_NAME}-${TASK}-db-${EXP}-agent1"

python3 -m examples.mars2.multi_agent_train_ppo_ray \
    --default_agent "$DEFAULT_AGENT" \
    --agents "$AGENT0" \
    --workflow_args "$WORKFLOW_ARGS" \
    --workflow_func_path /mnt/shared-storage-user/marti/MARTI-v2/openrlhf/agent_workflows/ab_mcts_workflow.py \
    --processor_func_path /mnt/shared-storage-user/marti/MARTI-v2/openrlhf/agent_workflows/ab_mcts_processor.py \
    --parallel_loading \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 4 \
    --reward_num_nodes 1 \
    --reward_num_gpus_per_node 4 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 4 \
    --vllm_num_engines 4 \
    --vllm_tensor_parallel_size 1 \
    --colocate_all_models \
    --vllm_gpu_memory_utilization 0.6 \
    --micro_train_batch_size 1 \
    --train_batch_size 32 \
    --micro_rollout_batch_size 1 \
    --rollout_batch_size 32 \
    --n_samples_per_prompt 1 \
    --max_epochs 1 \
    --seed 42 \
    --prompt_max_len ${PROMPT_MAX_LEN} \
    --generate_max_len ${GENERATE_MAX_LEN} \
    --eval_generate_max_len ${EVAL_GENERATE_MAX_LEN} \
    --max_len ${MAX_LEN} \
    --advantage_estimator ${ADVANTAGE} \
    --zero_stage 3 \
    --bf16 \
    --actor_learning_rate 1e-6 \
    --critic_learning_rate 9e-6 \
    --init_kl_coef 1e-3 \
    --gamma 1.0 \
    --use_kl_loss \
    --kl_estimator k3 \
    --normalize_reward \
    --gradient_checkpointing \
    --packing_samples \
    --vllm_sync_backend nccl \
    --enforce_eager \
    --vllm_enable_sleep \
    --deepspeed_enable_sleep \
    --dynamic_filtering_reward_range 0 1 \
    --overlong_buffer_len ${OVERLONG_BUFFER_LEN} \
    --policy_loss_type gspo \
    --enable_vllm_is_correction \
    --vllm_is_truncated_threshold 2 \
    --temperature 1.0 \
    --top_p 1.0 \
    --save_hf_ckpt \
    --save_steps 4 \
    --eval_steps 4 \
    --num_episodes 2 \
    --max_samples 100000 \
    --prompt_data ${PROMPT_DATA} \
    --prompt_split "train" \
    --eval_dataset ${PROMPT_DATA} \
    --eval_split "test" \
    --eval_temperature 1.0 \
    --eval_n_samples_per_prompt 1 \
    --input_key="prompt" \
    --label_key="label" \
    --load_checkpoint \
    --dynamic_filtering_for_agents \
    --use_tensorboard "${TENSORBOARD}" 2>&1 | tee ${LOG_DIR}
