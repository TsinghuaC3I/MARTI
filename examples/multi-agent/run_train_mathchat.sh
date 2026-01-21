#!/bin/bash
# gspo + tis + datafilter + overlong
# Debate workflow with 4 GPUs
# set -x
ROOT_DIR=""
cd ${ROOT_DIR}

# basic config
MODEL_DIR="/your_model_path"
SHORT_NAME=${1:-"Qwen2.5-3B-Instruct"}
PRETRAIN="${MODEL_DIR}/${SHORT_NAME}"
PROMPT_MAX_LEN=16384
GENERATE_MAX_LEN=4096
EVAL_GENERATE_MAX_LEN=4096
OVERLONG_BUFFER_LEN=2048
MAX_LEN=30000
ADVANTAGE="reinforce"

# set your root dir
TASK="MATH"
PROMPT_DATA="json@${ROOT_DIR}/data/${TASK}"

# Workflow config
NUM_TASKS=16
EXP=mathchat-3x2-unshared

WORKFLOW_SAVE_PATH="${ROOT_DIR}/outputs/workflow/${ADVANTAGE}-${SHORT_NAME}-${TASK}-db-${EXP}"

TENSORBOARD="${ROOT_DIR}/logs/tensorboard/${ADVANTAGE}-${SHORT_NAME}-${TASK}-db-${EXP}"
LOG_DIR=${ROOT_DIR}/logs/${ADVANTAGE}-${SHORT_NAME}-${TASK}-db-${EXP}.log

# port&env
export DOCKER_HOST=tcp://100.96.26.35:2376
# 更新 no_proxy/NO_PROXY，添加远程IP 100.103.184.53，避免代理拦截
export no_proxy=localhost,127.0.0.1,100.96.26.35
export NO_PROXY=$no_proxy 
export PYTHONNOUSERSITE=1
export MASTER_PORT=8266
export OPENRLHF_ASYNC_NUM_TASKS=${NUM_TASKS}

# check whether sandbox run successfully
curl 'http://100.96.26.35:8080/run_code' \
  -H 'Content-Type: application/json' \
  --noproxy localhost \
  --data-raw '{"code": "print(\"Hello, world!\")", "language": "python"}'

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
    # \"tool_manager\":

    # \"num_workers\": 512,

TOOL_ARGS="{
    \"max_turns\": 2,
    \"num_workers\": 4,
    \"enable_metrics\": true,
    \"enable_rate_limiting\": true,
    \"tools\": {
        \"code_interpreter\": {
            \"type\": \"sandbox_fusion\",
            \"enable_rate_limiting\": true,
            \"rate_limit\": 256,
            \"timeout\": 30,
            \"base_url\": \"http://100.96.26.35:8080/run_code\",
            \"schema_path\": \"examples/schema/code.json\"
        }
    }
}"


REWARD_ALLOC_ARGS="{
    \"name\": \"margin\",
    \"alpha\": 0.5,
    \"beta\": 0.5,
    \"use_ttrl\": false
}"
# reward_alloc.name="margin" \
# reward_alloc.alpha=0.5 \
# reward_alloc.beta=0.5 \
# reward_alloc.use_ttrl=False \


AGENT0="{
    \"0\": {
        \"role\": \"agent_problem_solver\",
        \"pretrain\": \"${PRETRAIN}\",
        \"save_path\": \"${ROOT_DIR}/outputs/final/${ADVANTAGE}-${SHORT_NAME}-${TASK}-db-${EXP}-agent0\",
        \"ckpt_path\": \"${ROOT_DIR}/outputs/ckpt/${ADVANTAGE}-${SHORT_NAME}-${TASK}-db-${EXP}-agent0\",
        \"is_tuning\": true,
        \"chat_template\": \"You are Agent Problem Solver, and your role is to collaborate with other agents to address various challenges.\nFor each problem, please follow these steps:\n    1. **Document Your Solution**: Write your solution step by step, ensuring it is independent of the solutions provided by other agents.\n    2. **Engage in Discussion**: Once you have outlined your solution, discuss your approach and findings with the other agents.\n\nProblem: $question\n\nPlease reason step by step, and put your final answer within \\boxed{}.\"
    }
}"

AGENT1="{
    \"1\": {
        \"role\": \"agent_code_executor\",
        \"pretrain\": \"${PRETRAIN}\",
        \"save_path\": \"${ROOT_DIR}/outputs/final/${ADVANTAGE}-${SHORT_NAME}-${TASK}-db-${EXP}-agent1\",
        \"ckpt_path\": \"${ROOT_DIR}/outputs/ckpt/${ADVANTAGE}-${SHORT_NAME}-${TASK}-db-${EXP}-agent1\",
        \"is_tuning\": true,
        \"code_execution\": true,
        \"chat_template\": \"You are Agent Code Executor. You can solve problems only writing commented Python code.\nFor each problem, please follow these steps:\n    1. **Develop Your Solution**: Write your solution in Python code, detailing each step independently from the solutions provided by other agents.\n    2. **Utilize SymPy**: Feel free to use the SymPy package to facilitate calculations and enhance your code's efficiency.\n    3. **Display Results**: Ensure that you **print the final result at the end of your Python code** (e.g., \`print(_result_)\`).\n    4. **Engage in Discussion**: After obtaining the result from your Python code, discuss your findings with the other agents.\nAlways format your Python code within:\n\`\`\`python\n# your code here\nprint(_result_)\n\`\`\`\n\nProblem: $question\n\nHere is the output from Agent Problem Solver:\n$agent_problem_solver\"
    }
}"

AGENT2="{
    \"2\": {
        \"role\": \"agent_verifier\",
        \"pretrain\": \"${PRETRAIN}\",
        \"save_path\": \"${ROOT_DIR}/outputs/final/${ADVANTAGE}-${SHORT_NAME}-${TASK}-db-${EXP}-agent2\",
        \"ckpt_path\": \"${ROOT_DIR}/outputs/ckpt/${ADVANTAGE}-${SHORT_NAME}-${TASK}-db-${EXP}-agent2\",
        \"is_tuning\": true,
        \"chat_template\": \"You are Agent Verifier.\nYour role is to critically evaluate the solutions proposed by other agents step by step and provide a final solution.\n    1. **Solution Requirement**: Before making any decisions, ensure you have received solutions from both Agent Code Executor and Agent Problem Solver.\n    2. **Avoid Assumptions**: Pay attention to the variables provided in the original problem statement versus those assumed by the agents. **Assumed values are not valid for the solution** and can lead to inaccuracies. Never base your solution on assumed values. Always base your solution on the explicitly given variables to ensure correctness. If a problem is deemed unsolvable due to missing information, return: **SOLUTION_FOUND \\boxed{'None'}**.\n    3. **Evaluating Conflicting Solutions**: If different answers are presented during the discussion, choose the most appropriate solution based on your evidence or initiate further discussion to clarify.\n    4. **Final Solution Declaration**: When you are confident about the final solution, return it as follows: **SOLUTION_FOUND \\boxed{_solution_value_here_}**. Ensure that only numerical values are placed inside the \\boxed{}; any accompanying text should be outside.\n\nProblem: $question\n\nHere is the output from Agent Problem Solver:\n$agent_problem_solver\n\nHere is the output from Agent Code Executor:\n$agent_code_executor\"
    }
}"

# WANDB_KEY="your_wandb_key"  # your wandb API key
# WANDB_PROJECT="openrlhf_math_ppo"
# # WANDB_ORG="your-wandb-org"  # optional
# # WANDB_GROUP="${ADVANTAGE}-${SHORT_NAME}-${TASK}"  # optional
# WANDB_RUN_NAME="${ADVANTAGE}-${SHORT_NAME}-${TASK}-${EXP}"  # optional

export NCCL_DEBUG=WARN

mkdir -p "${ROOT_DIR}/logs"
mkdir -p "${WORKFLOW_SAVE_PATH}"
mkdir -p "${TENSORBOARD}"
mkdir -p "${ROOT_DIR}/outputs/final/${ADVANTAGE}-${SHORT_NAME}-${TASK}-db-${EXP}-agent0"
mkdir -p "${ROOT_DIR}/outputs/final/${ADVANTAGE}-${SHORT_NAME}-${TASK}-db-${EXP}-agent1"
mkdir -p "${ROOT_DIR}/outputs/ckpt/${ADVANTAGE}-${SHORT_NAME}-${TASK}-db-${EXP}-agent0"
mkdir -p "${ROOT_DIR}/outputs/ckpt/${ADVANTAGE}-${SHORT_NAME}-${TASK}-db-${EXP}-agent1"


# 128 128 8 
python3 -m openrlhf.cli.multi_agent_train_ppo_ray \
    --default_agent "$DEFAULT_AGENT" \
    --agents "$AGENT0" "$AGENT1" "$AGENT2" \
    --workflow_args "$WORKFLOW_ARGS" \
    --tools_config "$TOOL_ARGS" \
    --reward_alloc "$REWARD_ALLOC_ARGS" \
    --workflow_func_path openrlhf/agent_workflows/mathchat_workflow.py \
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
    --micro_train_batch_size 4 \
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
    --init_kl_coef 0.0 \
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
    --dynamic_filtering_for_agents 2>&1 | tee "${LOG_DIR}"
    # --use_wandb "${WANDB_KEY}" \
    # --wandb_project "MARTI" \
    # --wandb_run_name "${EXP}"
    # --use_tensorboard "${TENSORBOARD}" \
    # ${WANDB_API_KEY:+--use_wandb "${WANDB_API_KEY}"} \
    # ${WANDB_PROJECT:+--wandb_project "${WANDB_PROJECT}"} \
    # ${WANDB_RUN_NAME:+--wandb_run_name "${WANDB_RUN_NAME}"} \
    # 2>&1 | tee ${LOG_DIR}