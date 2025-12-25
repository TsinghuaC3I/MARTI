#!/bin/bash
# ReviewRL training script migrated to new framework
set -x

# source ${ROOT_DIR}/miniconda3/etc/profile.d/conda.sh
# conda activate marti_vllm
# which conda
# which python
ROOT_DIR=""
cd ${ROOT_DIR}

# basic config
MODEL_DIR="/your_model_path"
SHORT_NAME0=${1:-"Qwen2.5-7B-Instruct"}
SHORT_NAME1=${2:-"Qwen3-4B-Instruct-2507"}
PRETRAIN0="${MODEL_DIR}/${SHORT_NAME0}"
PRETRAIN1="${MODEL_DIR}/${SHORT_NAME1}"
SHORT_NAME="${SHORT_NAME0}_${SHORT_NAME1}"
PROMPT_MAX_LEN=24000
GENERATE_MAX_LEN=16000
EVAL_GENERATE_MAX_LEN=16000
OVERLONG_BUFFER_LEN=2048
MAX_LEN=32768
ADVANTAGE="reinforce"

DATE=$(date +%m%d)
TASK="REVIEW"
PROMPT_DATA="json@${ROOT_DIR}/data/${TASK}"

# Workflow config
EXP="ReviewRL-async"

WORKFLOW_SAVE_PATH="${ROOT_DIR}/outputs/workflow/${ADVANTAGE}-${SHORT_NAME}-${TASK}-${EXP}"

TENSORBOARD="${ROOT_DIR}/logs/tensorboard/${ADVANTAGE}-${SHORT_NAME}-${TASK}-${EXP}"
LOG_DIR=${ROOT_DIR}/logs/${ADVANTAGE}-${SHORT_NAME}-${TASK}-${EXP}.log

# set port and env var
export MASTER_PORT=$(shuf -i 10000-65535 -n 1)
export OPENRLHF_ASYNC_NUM_TASKS=16
export WANDB_API_KEY="1a81e954eb0305fff7b73e92388dd3f3086c1380"

# default agent config
DEFAULT_AGENT="{
    \"is_reasoning_model\": true
}"

# Workflow additional param config - using judge workflow for ReviewRL
WORKFLOW_ARGS=$(cat <<EOF
{
    "save_path": "${WORKFLOW_SAVE_PATH}",
    "num_rounds": 1,
    "score_parser": "keywords",
    "judge_weight": 0.5,
    "label_separator": "||DIV REVIEW SCORE||",
    "judge_template": "You are an expert academic peer reviewer. Compare these two peer reviews for the same research paper.\n\n[Paper Abstract]\n{prompt}\n\n[Review 1 - Generated]\n{generated_answer}\n\n[Review 2 - Reference]\n{label}\n\nEvaluate both reviews based on:\n1. Technical depth and accuracy\n2. Completeness of coverage\n3. Specificity of feedback\n4. Constructiveness for authors\n\nYour response MUST be EXACTLY one of these two options (no other text):\nREVIEW_1_BETTER\nREVIEW_2_BETTER\n\nResponse:"
}
EOF
)

# Agent 0 configuration (main generator/actor)
AGENT0="{
    \"0\": {
        \"role\": \"generator\",
        \"pretrain\": \"${PRETRAIN0}\",
        \"is_reasoning_model\": false,
        \"save_path\": \"${ROOT_DIR}/outputs/final/${ADVANTAGE}-${SHORT_NAME}-${TASK}-${EXP}-agent0\",
        \"ckpt_path\": \"${ROOT_DIR}/outputs/ckpt/${ADVANTAGE}-${SHORT_NAME}-${TASK}-${EXP}-agent0\",
        \"is_tuning\": true
    }
}"


AGENT1="{
    \"1\": {
        \"role\": \"judge\",
        \"pretrain\": \"${PRETRAIN1}\",
        \"is_reasoning_model\": false,
        \"generate_max_len\": 256,
        \"save_path\": \"${ROOT_DIR}/outputs/final/${ADVANTAGE}-${SHORT_NAME}-${TASK}-${EXP}-agent1\",
        \"ckpt_path\": \"${ROOT_DIR}/outputs/ckpt/${ADVANTAGE}-${SHORT_NAME}-${TASK}-${EXP}-agent1\",
        \"is_tuning\": false
    }
}"

export NCCL_DEBUG=WARN
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

mkdir -p "${ROOT_DIR}/logs"
mkdir -p "${WORKFLOW_SAVE_PATH}"
mkdir -p "${TENSORBOARD}"
mkdir -p "${ROOT_DIR}/outputs/final/${ADVANTAGE}-${SHORT_NAME}-${TASK}-${EXP}-agent0"
mkdir -p "${ROOT_DIR}/outputs/ckpt/${ADVANTAGE}-${SHORT_NAME}-${TASK}-${EXP}-agent0"


python3 -m openrlhf.cli.multi_agent_train_ppo_ray \
    --default_agent "$DEFAULT_AGENT" \
    --agents "$AGENT0" "$AGENT1" \
    --workflow_args "$WORKFLOW_ARGS" \
    --workflow_func_path ${ROOT_DIR}/openrlhf/agent_workflows/judge_workflow.py \
    --parallel_loading \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 2 \
    --reward_num_nodes 1 \
    --reward_num_gpus_per_node 2 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 2 \
    --vllm_num_engines 2 \
    --vllm_tensor_parallel_size 1 \
    --colocate_all_models \
    --vllm_gpu_memory_utilization 0.8 \
    --vllm_sync_backend nccl \
    --vllm_enable_sleep \
    --deepspeed_enable_sleep \
    --micro_train_batch_size 1 \
    --train_batch_size 32 \
    --micro_rollout_batch_size 1 \
    --rollout_batch_size 32 \
    --n_samples_per_prompt 8 \
    --max_epochs 1 \
    --seed 42 \
    --prompt_max_len ${PROMPT_MAX_LEN} \
    --generate_max_len ${GENERATE_MAX_LEN} \
    --eval_generate_max_len ${EVAL_GENERATE_MAX_LEN} \
    --max_len ${MAX_LEN} \
    --overlong_buffer_len ${OVERLONG_BUFFER_LEN} \
    --advantage_estimator ${ADVANTAGE} \
    --temperature 0.7 \
    --top_p 0.95 \
    --lambd 1.0 \
    --gamma 1.0 \
    --zero_stage 3 \
    --bf16 \
    --actor_learning_rate 1e-6 \
    --critic_learning_rate 9e-6 \
    --init_kl_coef 0.00 \
    --use_kl_loss \
    --kl_estimator k3 \
    --normalize_reward \
    --adam_offload \
    --gradient_checkpointing \
    --packing_samples \
    --enforce_eager \
    --save_hf_ckpt \
    --save_steps 1 \
    --eval_steps 1 \
    --logging_steps 1 \
    --num_episodes 1 \
    --max_samples 20 \
    --max_ckpt_num 3 \
    --prompt_data ${PROMPT_DATA} \
    --prompt_split "train" \
    --input_key="prompt" \
    --label_key="cleaned_output" \
    --verify_task="review_group" \
    --verify_task_eval="review_group" \
    --load_checkpoint \
    --dynamic_filtering_for_agents \
    --use_tensorboard "${TENSORBOARD}" 2>&1 | tee ${LOG_DIR}

#     --wandb_project "MARTI" \
#     --wandb_run_name "${DATE}-${TASK}-${SHORT_NAME}-${ADVANTAGE}-${EXP}" \
# echo "Model Training Finished. Shutting down..."

