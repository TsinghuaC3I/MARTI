#!/bin/bash
# gspo + tis + datafilter + overlong
# Chain workflow with 4 GPUs
# Pattern: Generator -> Verifier -> Refiner
# set -x
# source /mnt/shared-storage-user/marti/miniconda3/etc/profile.d/conda.sh
# conda activate marti_vllm
# which conda
# which python
# cd /mnt/shared-storage-user/marti/OpenRLHF

# 基础配置
MODEL_DIR="/mnt/public/hf_models/Qwen/"
#使用qwen3 8B
SHORT_NAME=${1:-"Qwen2.5-3B-Instruct"}
PRETRAIN="${MODEL_DIR}/${SHORT_NAME}"
PROMPT_MAX_LEN=4096
GENERATE_MAX_LEN=32768
EVAL_GENERATE_MAX_LEN=32768
OVERLONG_BUFFER_LEN=2048
MAX_LEN=30000
ADVANTAGE="group_norm"

ROOT_DIR="/mnt/public/tk/1024/MARTI-Dev-openrlhf-v2"
TASK="MATH"
PROMPT_DATA="json@/mnt/public/tk/1024/MARTI/data/${TASK}"

# Workflow 配置
NUM_TASKS=16  # 异步任务并发数量，替代原来的 tools_config.num_workers
EXP=chain

WORKFLOW_SAVE_PATH="${ROOT_DIR}/outputs/workflow/${ADVANTAGE}-${SHORT_NAME}-${TASK}-ch-${EXP}"

TENSORBOARD="${ROOT_DIR}/logs/tensorboard/${ADVANTAGE}-${SHORT_NAME}-${TASK}-ch-${EXP}"
LOG_DIR=/mnt/public/tk/1024/MARTI-Dev-openrlhf-v2/logs/${ADVANTAGE}-${SHORT_NAME}-${TASK}-ch-${EXP}.log

# 设置动态端口和环境变量
export MASTER_PORT=8266
export OPENRLHF_ASYNC_NUM_TASKS=${NUM_TASKS}

# 定义默认智能体配置
DEFAULT_AGENT="{
    \"is_reasoning_model\": false
}"

# 定义 Workflow 参数配置
WORKFLOW_ARGS="{
    \"task\": \"math\"
}"

# 定义智能体0配置（generator角色）
AGENT0="{
    \"0\": {
        \"role\": \"generator\",
        \"pretrain\": \"${PRETRAIN}\",
        \"save_path\": \"/mnt/public/tk/1024/MARTI-Dev-openrlhf-v2/outputs/final/${ADVANTAGE}-${SHORT_NAME}-${TASK}-ch-${EXP}-agent0\",
        \"ckpt_path\": \"/mnt/public/tk/1024/MARTI-Dev-openrlhf-v2/outputs/ckpt/${ADVANTAGE}-${SHORT_NAME}-${TASK}-ch-${EXP}-agent0\",
        \"is_tuning\": true,
        \"chat_template\": [\"\\$question\\n\\nPlease reason step by step, and put your final answer within \\\\boxed{}.\\n\"]
    }
}"

# 定义智能体1配置（verifier角色）
AGENT1="{
    \"1\": {
        \"role\": \"verifier\",
        \"pretrain\": \"${PRETRAIN}\",
        \"save_path\": \"/mnt/public/tk/1024/MARTI-Dev-openrlhf-v2/outputs/final/${ADVANTAGE}-${SHORT_NAME}-${TASK}-ch-${EXP}-agent1\",
        \"ckpt_path\": \"/mnt/public/tk/1024/MARTI-Dev-openrlhf-v2/outputs/ckpt/${ADVANTAGE}-${SHORT_NAME}-${TASK}-ch-${EXP}-agent1\",
        \"is_tuning\": true,
        \"chat_template\": [\"You are tasked with analyzing an answer to a problem and providing constructive feedback.\\n\\nProblem: \\$question\\n\\nSolution: \\$generator\\n\\nDo NOT provide direct solutions.\\n\"]
    }
}"

# 定义智能体2配置（refiner角色）
AGENT2="{
    \"2\": {
        \"role\": \"refiner\",
        \"pretrain\": \"${PRETRAIN}\",
        \"save_path\": \"/mnt/public/tk/1024/MARTI-Dev-openrlhf-v2/outputs/final/${ADVANTAGE}-${SHORT_NAME}-${TASK}-ch-${EXP}-agent2\",
        \"ckpt_path\": \"/mnt/public/tk/1024/MARTI-Dev-openrlhf-v2/outputs/ckpt/${ADVANTAGE}-${SHORT_NAME}-${TASK}-ch-${EXP}-agent2\",
        \"is_tuning\": true,
        \"chat_template\": [\"You are tasked with revising a draft solution to a problem based on the critique provided. Please provide a revised solution that addresses the feedback and improves the overall quality of the solution.\\n\\nProblem: \\$question\\n\\nSolution: \\$generator\\n\\nCritique: \\$verifier\\n\\nPlease reason step by step, and put your final answer within \\\\boxed{}.\\n\"]
    }
}"

WANDB_KEY="6461769bfefb0c4fa60bde38d55799cf3ee3986e"  # 你的 wandb API key
# WANDB_PROJECT="openrlhf_math_ppo"
# # WANDB_ORG="your-wandb-org"  # 可选
# # WANDB_GROUP="${ADVANTAGE}-${SHORT_NAME}-${TASK}"  # 可选
# WANDB_RUN_NAME="${ADVANTAGE}-${SHORT_NAME}-${TASK}-${EXP}"  # 可选

export NCCL_DEBUG=WARN

# 确保所有必要的目录存在
mkdir -p "${ROOT_DIR}/logs"
mkdir -p "${WORKFLOW_SAVE_PATH}"
mkdir -p "${TENSORBOARD}"
mkdir -p "/mnt/public/tk/1024/MARTI-Dev-openrlhf-v2/outputs/final/${ADVANTAGE}-${SHORT_NAME}-${TASK}-ch-${EXP}-agent0"
mkdir -p "/mnt/public/tk/1024/MARTI-Dev-openrlhf-v2/outputs/final/${ADVANTAGE}-${SHORT_NAME}-${TASK}-ch-${EXP}-agent1"
mkdir -p "/mnt/public/tk/1024/MARTI-Dev-openrlhf-v2/outputs/final/${ADVANTAGE}-${SHORT_NAME}-${TASK}-ch-${EXP}-agent2"
mkdir -p "/mnt/public/tk/1024/MARTI-Dev-openrlhf-v2/outputs/ckpt/${ADVANTAGE}-${SHORT_NAME}-${TASK}-ch-${EXP}-agent0"
mkdir -p "/mnt/public/tk/1024/MARTI-Dev-openrlhf-v2/outputs/ckpt/${ADVANTAGE}-${SHORT_NAME}-${TASK}-ch-${EXP}-agent1"
mkdir -p "/mnt/public/tk/1024/MARTI-Dev-openrlhf-v2/outputs/ckpt/${ADVANTAGE}-${SHORT_NAME}-${TASK}-ch-${EXP}-agent2"

# 运行训练脚本
#--vllm_generate_batch_size 32

python3 -m openrlhf.cli.multi_agent_train_ppo_ray \
    --default_agent "$DEFAULT_AGENT" \
    --agents "$AGENT0" "$AGENT1" "$AGENT2" \
    --workflow_args "$WORKFLOW_ARGS" \
    --workflow_func_path /mnt/public/tk/1024/MARTI-Dev-openrlhf-v2/openrlhf/agent_workflows/chain_workflow.py \
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
    --dynamic_filtering_for_agents \
    --use_wandb "${WANDB_KEY}" \
    --wandb_project "MARTI" \
    --wandb_run_name "${EXP}"
    # --use_tensorboard "${TENSORBOARD}" \
    # ${WANDB_API_KEY:+--use_wandb "${WANDB_API_KEY}"} \
    # ${WANDB_PROJECT:+--wandb_project "${WANDB_PROJECT}"} \
    # ${WANDB_RUN_NAME:+--wandb_run_name "${WANDB_RUN_NAME}"} \
    # 2>&1 | tee ${LOG_DIR}


    # --task "math" \
    # --tool_manager None\

