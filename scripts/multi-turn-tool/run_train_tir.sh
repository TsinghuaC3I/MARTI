#!/bin/bash
# Tool-Integrated Reasoning (TIR) training script for Math tasks with Python code interpreter
set -x

ROOT_DIR=""
cd ${ROOT_DIR}

MODEL_DIR="${MODEL_DIR:-/your_model_path}"
SHORT_NAME="${1:-Qwen3-4B}"
PRETRAIN="${MODEL_DIR}/${SHORT_NAME}"

PROMPT_MAX_LEN=4096
GENERATE_MAX_LEN=32768
EVAL_GENERATE_MAX_LEN=32768
MAX_LEN=40960
ADVANTAGE="group_norm"

TASK="MATH"
ALGO="tool-code-sandbox-fusion"
DATE=$(date +%m%d)

PROMPT_DATA="${PROMPT_DATA:-json@${ROOT_DIR}/data/${TASK}}"

# EXTRA_EVAL_TASKS='["aime","amc"]'
# EXTRA_EVAL_DIR="${ROOT_DIR}/data/Bench"

NUM_TASKS=128
EXP="${DATE}-${TASK}-${SHORT_NAME}-${ADVANTAGE}-${ALGO}"

SAVE_PATH="${ROOT_DIR}/outputs/final/${ADVANTAGE}-${SHORT_NAME}-${TASK}-${EXP}"
CKPT_PATH="${ROOT_DIR}/outputs/ckpt/${ADVANTAGE}-${SHORT_NAME}-${TASK}-${EXP}"
TENSORBOARD="${ROOT_DIR}/logs/tensorboard/${ADVANTAGE}-${SHORT_NAME}-${TASK}-${EXP}"
LOG_DIR="${ROOT_DIR}/logs/${ADVANTAGE}-${SHORT_NAME}-${TASK}-${EXP}.log"


# Sandbox URL for Python code execution
CODE_SANDBOX_URL="${CODE_SANDBOX_URL:-http://100.96.26.35:8080/run_code}"
CODE_SCHEMA_PATH="${ROOT_DIR}/examples/schema/code.json"
MAX_TURNS=2


# 添加 DOCKER_HOST 环境变量，指向远程Docker守护进程（100.103.184.53:2376）
export DOCKER_HOST=tcp://100.96.26.35:2376
export MASTER_PORT=$(shuf -i 10000-65535 -n 1)
export OPENRLHF_ASYNC_NUM_TASKS=${NUM_TASKS}
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=true

# 更新 no_proxy/NO_PROXY，添加远程IP 100.103.184.53，避免代理拦截
export no_proxy=localhost,127.0.0.1,100.96.26.35
export NO_PROXY=$no_proxy 


# Default agent settings
DEFAULT_AGENT="{
    \"is_reasoning_model\": true
}"

# Agent 0: Generator with training enabled
AGENT0="{
    \"0\": {
        \"role\": \"generator\",
        \"pretrain\": \"${PRETRAIN}\",
        \"save_path\": \"${SAVE_PATH}\",
        \"ckpt_path\": \"${CKPT_PATH}\",
        \"is_tuning\": true,
        \"chat_template\": \"Answer the given math problem. You must conduct reasoning inside  and  first every time you get new information or need to solve a subproblem. After reasoning, if you find that a calculation or code is needed, you can use the code interpreter inside <tool_call> and </tool_call>; it will return the result between <tool_response> and </tool_response>. You can run as many code cells as needed. If you find no further computation is necessary, put your final answer within \\\\boxed{{}}. Question: {question}\"
    }
}"

# Workflow arguments for tool-integrated reasoning
WORKFLOW_ARGS="{
    \"task\": \"math\"
}"

TOOL_ARGS="{
    \"max_turns\": ${MAX_TURNS},
    \"num_workers\": 128,
    \"enable_metrics\": true,
    \"enable_rate_limiting\": true,
    \"tools\": {
        \"python\": {
            \"type\": \"sandbox_fusion\",
            \"enable_rate_limiting\": true,
            \"rate_limit\": 30,
            \"timeout\": 30,
            \"base_url\": \"${CODE_SANDBOX_URL}\",
            \"schema_path\": \"${CODE_SCHEMA_PATH}\"
        }
    }
}"

mkdir -p "${ROOT_DIR}/logs"
mkdir -p "${SAVE_PATH}"
mkdir -p "${CKPT_PATH}"
mkdir -p "${TENSORBOARD}"


python3 -m openrlhf.cli.multi_agent_train_ppo_ray \
    --default_agent "$DEFAULT_AGENT" \
    --agents "$AGENT0" \
    --workflow_args "$WORKFLOW_ARGS" \
    --tools_config "$TOOL_ARGS" \
    --workflow_func_path "${ROOT_DIR}/openrlhf/agent_workflows/tool_workflow.py" \
    --parallel_loading \
    --dynamic_filtering_for_agents \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 4 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 4 \
    --vllm_num_engines 4 \
    --vllm_tensor_parallel_size 1 \
    --colocate_all_models \
    --vllm_gpu_memory_utilization 0.6 \
    --micro_train_batch_size 1 \
    --train_batch_size 512 \
    --micro_rollout_batch_size 1 \
    --rollout_batch_size 256 \
    --n_samples_per_prompt 4 \
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
    --init_kl_coef 0.001 \
    --gamma 1.0 \
    --lambd 1.0 \
    --use_kl_loss \
    --kl_estimator k3 \
    --entropy_loss_coef 0 \
    --max_norm 1.0 \
    --normalize_reward \
    --gradient_checkpointing \
    --packing_samples \
    --vllm_sync_backend nccl \
    --enforce_eager \
    --vllm_enable_sleep \
    --deepspeed_enable_sleep \
    --temperature 1.0 \
    --top_p 1.0 \
    --save_hf_ckpt \
    --save_steps 10 \
    --eval_steps 1 \
    --logging_steps 1 \
    --num_episodes 5 \
    --max_samples 1000000 \
    --prompt_data "${PROMPT_DATA}" \
    --prompt_split "train" \
    --eval_dataset "${PROMPT_DATA}" \
    --eval_split "test" \
    --eval_temperature 1.0 \
    --eval_n_samples_per_prompt 2 \
    --input_key="prompt" \
    --label_key="answer" \
    --eval_before_training \
    --use_tensorboard "${TENSORBOARD}" 2>&1 | tee "${LOG_DIR}"

echo "Model Training Finished."