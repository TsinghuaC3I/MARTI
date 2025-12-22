#!/bin/bash
# ReviewRL training script migrated to new framework
set -x
source /mnt/shared-storage-user/marti/miniconda3/etc/profile.d/conda.sh
conda activate marti_vllm
which conda
which python
cd /mnt/shared-storage-user/marti/MARTI-v2

# basic config
MODEL_DIR="/mnt/shared-storage-user/marti/models"
SHORT_NAME=${1:-"qwen2.5-7b"}
PRETRAIN="${MODEL_DIR}/${SHORT_NAME}"
PROMPT_MAX_LEN=24000
GENERATE_MAX_LEN=16000
EVAL_GENERATE_MAX_LEN=16000
OVERLONG_BUFFER_LEN=2048
MAX_LEN=40000
ADVANTAGE="reinforce"

ROOT_DIR="/mnt/shared-storage-user/marti/MARTI-v2"
DATE=$(date +%m%d)
TASK="REVIEW"
PROMPT_DATA="json@/mnt/workspace/qibiqing/openreviewer/rl_marti/data_preprocess/rl_data_deepreview"

# Workflow config
EXP="ReviewRL-async"

WORKFLOW_SAVE_PATH="${ROOT_DIR}/outputs/workflow/${ADVANTAGE}-${SHORT_NAME}-${TASK}-${EXP}"

TENSORBOARD="${ROOT_DIR}/logs/tensorboard/${ADVANTAGE}-${SHORT_NAME}-${TASK}-${EXP}"
LOG_DIR=${ROOT_DIR}/logs/${ADVANTAGE}-${SHORT_NAME}-${TASK}-${EXP}.log

# set port and env var
export MASTER_PORT=$(shuf -i 10000-65535 -n 1)
export OPENRLHF_ASYNC_NUM_TASKS=128
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
    "judge_template": "You are an expert academic peer reviewer. You will be shown the abstract/content of a research paper and two peer reviews for that paper. Your task is to determine which peer review is of higher quality based on the following criteria:\n\n1. **Factual Accuracy & Soundness:** Does the review accurately understand the paper's contributions and limitations? Is the critique based on sound reasoning?\n\n2. **Completeness & Coverage:** Does the review address the core aspects of the paper (e.g., methodology, results, significance)?\n\n3. **Level of Detail & Specificity:** Does the review provide specific examples and detailed comments rather than vague statements?\n\n4. **Comparison with Existing Work:** Does the review appropriately contextualize the paper within the existing literature and compare it to relevant methods?\n\n5. **Constructiveness:** Is the feedback helpful for the authors to improve the paper? Is the tone professional and constructive?\n\n6. **Clarity & Organization:** Is the review well-structured and easy to understand?\n\n[Paper Context (Abstract/Content)]\n{prompt}\n\n[Review 1]\n{generated_answer}\n\n[Review 2]\n{label}\n\nWhich peer review is of higher quality based on the criteria above? Respond with EXACTLY one of these options:\n- REVIEW_1_BETTER\n- REVIEW_2_BETTER\n\nYOU MUST CHOOSE A BETTER REVIEW. A TIE IS NOT ALLOWED."
}
EOF
)

# Agent 0 configuration (main generator/actor)
AGENT0="{
    \"0\": {
        \"role\": \"generator\",
        \"pretrain\": \"${PRETRAIN}\",
        \"save_path\": \"${ROOT_DIR}/outputs/final/${ADVANTAGE}-${SHORT_NAME}-${TASK}-${EXP}-agent0\",
        \"ckpt_path\": \"${ROOT_DIR}/outputs/ckpt/${ADVANTAGE}-${SHORT_NAME}-${TASK}-${EXP}-agent0\",
        \"is_tuning\": true
    }
}"


#这个好像是纯奖励模型 不tune的 所以这些参数都不该有的
AGENT1="{
    \"1\": {
        \"role\": \"judge\",
        \"pretrain\": \"${PRETRAIN}\",
        \"is_reasoning_model\": false
    }
}"

        
        # \"save_path\": \"${ROOT_DIR}/outputs/final/${ADVANTAGE}-${SHORT_NAME}-${TASK}-${EXP}-agent1\",
        # \"ckpt_path\": \"${ROOT_DIR}/outputs/ckpt/${ADVANTAGE}-${SHORT_NAME}-${TASK}-${EXP}-agent1\",
        # \"is_tuning\": true

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
    --workflow_func_path /mnt/shared-storage-user/marti/MARTI-v2/openrlhf/agent_workflows/judge_workflow.py \
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
    --vllm_gpu_memory_utilization 0.8 \
    --vllm_sync_backend nccl \
    --vllm_enable_sleep \
    --deepspeed_enable_sleep \
    --micro_train_batch_size 1 \
    --train_batch_size 32 \
    --micro_rollout_batch_size 2 \
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
    --temperature 1.0 \
    --top_p 1.0 \
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
    --save_steps 3 \
    --eval_steps 1 \
    --logging_steps 1 \
    --num_episodes 1 \
    --max_samples 400000 \
    --max_ckpt_num 3 \
    --prompt_data ${PROMPT_DATA} \
    --prompt_split "train" \
    --input_key="problem" \
    --label_key="answer" \
    --load_checkpoint \
    --wandb_project "MARTI" \
    --wandb_run_name "${DATE}-${TASK}-${SHORT_NAME}-${ADVANTAGE}-${EXP}" \
    --use_tensorboard "${TENSORBOARD}" 2>&1 | tee ${LOG_DIR}

echo "Model Training Finished. Shutting down..."

