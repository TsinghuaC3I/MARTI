#!/bin/bash
# gspo + tis + datafilter + overlong
# set -x
ROOT_DIR=""
cd "${ROOT_DIR}"

# basic config
# set local model path
MODEL_DIR=""
SHORT_NAME=${1:-"Qwen3-1.7B"}
PRETRAIN="${MODEL_DIR}/${SHORT_NAME}"
PROMPT_MAX_LEN=4096
GENERATE_MAX_LEN=32768
EVAL_GENERATE_MAX_LEN=32768
OVERLONG_BUFFER_LEN=2048
MAX_LEN=40000
ADVANTAGE="group_norm"

# set correct root dir

TASK="CODE"
PROMPT_DATA="json@${ROOT_DIR}/data/${TASK}"

# Workflow config
MCTS_NODES=2
NUM_TASKS=16
EXP=all_tricks

WORKFLOW_SAVE_PATH="${ROOT_DIR}/outputs/workflow/${ADVANTAGE}-${SHORT_NAME}-${TASK}-db-${EXP}"

TENSORBOARD="${ROOT_DIR}/logs/tensorboard/${ADVANTAGE}-${SHORT_NAME}-${TASK}-db-${EXP}"
LOG_DIR=./logs/${ADVANTAGE}-${SHORT_NAME}-${TASK}-db-${EXP}.log
export MASTER_PORT=8266
export OPENRLHF_ASYNC_NUM_TASKS=${NUM_TASKS}

DEFAULT_AGENT="{
    \"is_reasoning_model\": true
}"

WORKFLOW_ARGS="{
    \"task\": \"code\"
}"


AGENT0="{
    \"0\": {
        \"role\": \"generator\",
        \"pretrain\": \"${PRETRAIN}\",
        \"save_path\": \"./outputs/final/${ADVANTAGE}-${SHORT_NAME}-${TASK}-db-${EXP}-agent0\",
        \"ckpt_path\": \"./outputs/ckpt/${ADVANTAGE}-${SHORT_NAME}-${TASK}-db-${EXP}-agent0\",
        \"is_tuning\": true
    }
}"

export NCCL_DEBUG=WARN


mkdir -p "${ROOT_DIR}/logs"
mkdir -p "${WORKFLOW_SAVE_PATH}"
mkdir -p "${TENSORBOARD}"
mkdir -p "./outputs/final/${ADVANTAGE}-${SHORT_NAME}-${TASK}-db-${EXP}-agent0"
mkdir -p "./outputs/final/${ADVANTAGE}-${SHORT_NAME}-${TASK}-db-${EXP}-agent1"
mkdir -p "./outputs/ckpt/${ADVANTAGE}-${SHORT_NAME}-${TASK}-db-${EXP}-agent0"
mkdir -p "./outputs/ckpt/${ADVANTAGE}-${SHORT_NAME}-${TASK}-db-${EXP}-agent1"



python3 -m marti.cli.multi_agent_train_ppo_ray \
    --default_agent "$DEFAULT_AGENT" \
    --agents "$AGENT0" \
    --workflow_args "$WORKFLOW_ARGS" \
    --workflow_func_path marti/agent_workflows/single_codeworkflow.py \
    --parallel_loading \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 2 \
    --reward_num_nodes 1 \
    --reward_num_gpus_per_node 1 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 2 \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 2 \
    --colocate_all_models \
    --vllm_gpu_memory_utilization 0.6 \
    --micro_train_batch_size 1 \
    --train_batch_size 32 \
    --micro_rollout_batch_size 1 \
    --rollout_batch_size 32 \
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
    --dynamic_filtering_for_agents 

