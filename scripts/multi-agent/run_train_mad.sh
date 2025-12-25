#!/bin/bash
# gspo + tis + datafilter + overlong
# Debate workflow with 4 GPUs
# set -x

# basic config
MODEL_DIR="/your_model_path"
SHORT_NAME=${1:-"Qwen3-4B-Instruct-2507"}
PRETRAIN="${MODEL_DIR}${SHORT_NAME}"
PROMPT_MAX_LEN=4096
GENERATE_MAX_LEN=5000
EVAL_GENERATE_MAX_LEN=5000
OVERLONG_BUFFER_LEN=2048
MAX_LEN=30000
ADVANTAGE="group_norm"

ROOT_DIR=""
TASK="MATH"
PROMPT_DATA="json@${ROOT_DIR}/data/${TASK}"

# Workflow config
NUM_TASKS=16
EXP=debate

WORKFLOW_SAVE_PATH="${ROOT_DIR}/outputs/workflow/${ADVANTAGE}-${SHORT_NAME}-${TASK}-db-${EXP}"

TENSORBOARD="${ROOT_DIR}/logs/tensorboard/${ADVANTAGE}-${SHORT_NAME}-${TASK}-db-${EXP}"
LOG_DIR=${ROOT_DIR}/logs/multi_agent/${ADVANTAGE}-${SHORT_NAME}-${TASK}-db-${EXP}.log

# port&env
export PYTHONNOUSERSITE=1
export MASTER_PORT=$(shuf -i 10000-65535 -n 1)
export OPENRLHF_ASYNC_NUM_TASKS=${NUM_TASKS}
# export MASTER_PORT=8266
# export OPENRLHF_ASYNC_NUM_TASKS=${NUM_TASKS}

# default agent config
DEFAULT_AGENT="{
    \"is_reasoning_model\": false
}"

# Workflow params config
WORKFLOW_ARGS="{
    \"task\": \"math\",
    \"num_rounds\": 2,
    \"max_others\": 5,
    \"contain_self\": true,
    \"shuffle_responses\": true
}"

TOOLS_ARGS="{
    \"num_workers\": 128,
    \"max_concurrent_calls\": 8,
    \"enable_metrics\": true,
    \"enable_rate_limiting\": true,
    \"rate_limit\": 8
}"

REWARD_ALLOC_ARGS="{
    \"name\": \"margin\",
    \"alpha\": 0.5,
    \"beta\": 0.5,
    \"use_ttrl\": false
}"

AGENT0="{
    \"0\": {
        \"role\": \"generator\",
        \"pretrain\": \"${PRETRAIN}\",
        \"save_path\": \"${ROOT_DIR}/outputs/final/${ADVANTAGE}-${SHORT_NAME}-${TASK}-db-${EXP}-agent0\",
        \"ckpt_path\": \"${ROOT_DIR}/outputs/ckpt/${ADVANTAGE}-${SHORT_NAME}-${TASK}-db-${EXP}-agent0\",
        \"is_tuning\": true
    }
}"

AGENT1="{
    \"1\": {
        \"role\": \"generator\",
        \"pretrain\": \"${PRETRAIN}\",
        \"save_path\": \"${ROOT_DIR}/outputs/final/${ADVANTAGE}-${SHORT_NAME}-${TASK}-db-${EXP}-agent1\",
        \"ckpt_path\": \"${ROOT_DIR}/outputs/ckpt/${ADVANTAGE}-${SHORT_NAME}-${TASK}-db-${EXP}-agent1\",
        \"is_tuning\": true
    }
}"

WANDB_KEY="your_wandb_key"  # your wandb API key
# WANDB_PROJECT="openrlhf_math_ppo"
# # WANDB_ORG="your-wandb-org"  # optional
# # WANDB_GROUP="${ADVANTAGE}-${SHORT_NAME}-${TASK}"  # optional
# WANDB_RUN_NAME="${ADVANTAGE}-${SHORT_NAME}-${TASK}-${EXP}"  # optional

export NCCL_DEBUG=WARN

mkdir -p "${ROOT_DIR}/logs/multi_agent"
mkdir -p "${WORKFLOW_SAVE_PATH}"
mkdir -p "${TENSORBOARD}"
mkdir -p "${ROOT_DIR}/outputs/final/${ADVANTAGE}-${SHORT_NAME}-${TASK}-db-${EXP}-agent0"
mkdir -p "${ROOT_DIR}/outputs/final/${ADVANTAGE}-${SHORT_NAME}-${TASK}-db-${EXP}-agent1"
mkdir -p "${ROOT_DIR}/outputs/ckpt/${ADVANTAGE}-${SHORT_NAME}-${TASK}-db-${EXP}-agent0"
mkdir -p "${ROOT_DIR}/outputs/ckpt/${ADVANTAGE}-${SHORT_NAME}-${TASK}-db-${EXP}-agent1"

python3 -m openrlhf.cli.multi_agent_train_ppo_ray \
    --default_agent "$DEFAULT_AGENT" \
    --agents "$AGENT0" "$AGENT1" \
    --workflow_args "$WORKFLOW_ARGS" \
    --tools_config "$TOOLS_ARGS" \
    --reward_alloc "$REWARD_ALLOC_ARGS" \
    --workflow_func_path openrlhf/agent_workflows/debate_workflow.py \
    --processor_func_path openrlhf/agent_workflows/debate_processor.py \
    --parallel_loading \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 1 \
    --reward_num_nodes 1 \
    --reward_num_gpus_per_node 1 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 1 \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
    --colocate_all_models \
    --vllm_gpu_memory_utilization 0.6 \
    --micro_train_batch_size 1 \
    --train_batch_size 16 \
    --micro_rollout_batch_size 1 \
    --rollout_batch_size 16 \
    --n_samples_per_prompt 2 \
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
    --save_steps 20 \
    --eval_steps 20 \
    --num_episodes 2 \
    --max_samples 100000 \
    --prompt_data ${PROMPT_DATA} \
    --prompt_split "train" \
    --eval_dataset ${PROMPT_DATA} \
    --eval_split "test" \
    --eval_temperature 1.0 \
    --eval_n_samples_per_prompt 1 \
    --input_key="prompt" \
    --label_key="answer" \
    --load_checkpoint \
    --dynamic_filtering_for_agents 2>&1 | tee ${LOG_DIR}
    # --use_wandb "${WANDB_KEY}" \
    # --wandb_project "MARTI" \
    # --wandb_run_name "${EXP}"
    # --use_tensorboard "${TENSORBOARD}" \
    # ${WANDB_API_KEY:+--use_wandb "${WANDB_API_KEY}"} \
    # ${WANDB_PROJECT:+--wandb_project "${WANDB_PROJECT}"} \
    # ${WANDB_RUN_NAME:+--wandb_run_name "${WANDB_RUN_NAME}"} \
    # 2>&1 | tee ${LOG_DIR}