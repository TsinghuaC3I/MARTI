#!/bin/bash
# gspo + tis + datafilter + overlong
set -x
source /mnt/shared-storage-user/marti/miniconda3/etc/profile.d/conda.sh
conda activate marti_vllm
which conda
which python
cd /mnt/shared-storage-user/marti/OpenRLHF

# 基础配置
MODEL_DIR="/mnt/shared-storage-user/marti/models"
#使用areal 14B
SHORT_NAME=${1:-"Qwen3-14B"}
PRETRAIN="${MODEL_DIR}/${SHORT_NAME}"
PROMPT_MAX_LEN=4096
GENERATE_MAX_LEN=32768
EVAL_GENERATE_MAX_LEN=32768
OVERLONG_BUFFER_LEN=2048
MAX_LEN=40000
ADVANTAGE="group_norm"

ROOT_DIR="/mnt/shared-storage-user/marti/OpenRLHF"
TASK="CODE_8B_FILTER_FT"
PROMPT_DATA="json@/mnt/shared-storage-user/marti/lipengfei/MARTI_DEV/data/${TASK}"

# Workflow 配置
#改成8
MCTS_NODES=8
NUM_TASKS=128  # 异步任务并发数量，替代原来的 tools_config.num_workers
EXP=all-tricks

WORKFLOW_SAVE_PATH="${ROOT_DIR}/outputs/workflow/${ADVANTAGE}-${SHORT_NAME}-${TASK}-db-${EXP}"

TENSORBOARD="${ROOT_DIR}/logs/tensorboard/${ADVANTAGE}-${SHORT_NAME}-${TASK}-db-${EXP}"
LOG_DIR=/mnt/shared-storage-user/marti/OpenRLHF/logs/${ADVANTAGE}-${SHORT_NAME}-${TASK}-db-${EXP}.log

# 设置动态端口和环境变量
export MASTER_PORT=$(shuf -i 10000-65535 -n 1)
export OPENRLHF_ASYNC_NUM_TASKS=${NUM_TASKS}

# 定义默认智能体配置
DEFAULT_AGENT="{
    \"is_reasoning_model\": true
}"

# 定义 Workflow 参数配置
WORKFLOW_ARGS="{
    \"max_num_nodes\": ${MCTS_NODES},
    \"eval_max_num_nodes\": 1,
    \"save_path\": \"${WORKFLOW_SAVE_PATH}\",
    \"algo\": {
        \"class_name\": \"AsyncABMCTSA\",
        \"params\": {}
    }
}"

# 定义智能体1配置（generator角色）
AGENT0="{
    \"0\": {
        \"role\": \"generator\",
        \"pretrain\": \"${PRETRAIN}\",
        \"save_path\": \"/mnt/shared-storage-user/marti/OpenRLHF/outputs/final/${ADVANTAGE}-${SHORT_NAME}-${TASK}-db-${EXP}-agent0\",
        \"ckpt_path\": \"/mnt/shared-storage-user/marti/OpenRLHF/outputs/ckpt/${ADVANTAGE}-${SHORT_NAME}-${TASK}-db-${EXP}-agent0\",
        \"is_tuning\": true
    }
}"

# 定义智能体2配置（generator角色）
# AGENT2="{
#     \"agent2\": {
#         \"role\": \"generator\",
#         \"pretrain\": \"${PRETRAIN}\",
#         \"save_path\": \"/mnt/shared-storage-user/marti/OpenRLHF/outputs/final/${SHORT_NAME}-${EXP}-agent2\",
#         \"ckpt_path\": \"/mnt/shared-storage-user/marti/OpenRLHF/outputs/ckpt/${SHORT_NAME}-${EXP}-agent2\",
#         \"is_tuning\": true
#     }
# }"
export NCCL_DEBUG=WARN

# 确保所有必要的目录存在
mkdir -p "${ROOT_DIR}/logs"
mkdir -p "${WORKFLOW_SAVE_PATH}"
mkdir -p "${TENSORBOARD}"
mkdir -p "/mnt/shared-storage-user/marti/OpenRLHF/outputs/final/${ADVANTAGE}-${SHORT_NAME}-${TASK}-db-${EXP}-agent0"
mkdir -p "/mnt/shared-storage-user/marti/OpenRLHF/outputs/final/${ADVANTAGE}-${SHORT_NAME}-${TASK}-db-${EXP}-agent1"
mkdir -p "/mnt/shared-storage-user/marti/OpenRLHF/outputs/ckpt/${ADVANTAGE}-${SHORT_NAME}-${TASK}-db-${EXP}-agent0"
mkdir -p "/mnt/shared-storage-user/marti/OpenRLHF/outputs/ckpt/${ADVANTAGE}-${SHORT_NAME}-${TASK}-db-${EXP}-agent1"

# 运行训练脚本
#--vllm_generate_batch_size 32

python3 -m openrlhf.cli.multi_agent_train_ppo_ray \
    --default_agent "$DEFAULT_AGENT" \
    --agents "$AGENT0" \
    --workflow_args "$WORKFLOW_ARGS" \
    --workflow_func_path /mnt/shared-storage-user/marti/OpenRLHF/openrlhf/agent_workflows/ab_mcts_workflow.py \
    --parallel_loading \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 8 \
    --reward_num_nodes 1 \
    --reward_num_gpus_per_node 8 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 8 \
    --vllm_num_engines 8 \
    --vllm_tensor_parallel_size 1 \
    --colocate_all_models \
    --vllm_gpu_memory_utilization 0.75 \
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
    --eval_temperature 0.6 \
    --eval_n_samples_per_prompt 1 \
    --input_key="prompt" \
    --label_key="label" \
    --load_checkpoint \
    --dynamic_filtering_for_agents \
    --use_tensorboard "${TENSORBOARD}" 2>&1 | tee ${LOG_DIR}
