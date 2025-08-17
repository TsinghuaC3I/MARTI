#!/bin/bash
set -euo pipefail
set -x

# ---------- Ray Configuration ----------
export RAY_TMPDIR="${RAY_TMPDIR}"            # Move Ray's temporary directory to a place with sufficient inodes.
export RAY_LOG_TO_STDERR_ONLY=1              # Avoid writing massive log files
export RAY_OBJECT_STORE_MEMORY=1000000000    # Set the explicit size of the object store (can be increased according to machine memory).
export RAY_DISABLE_IMPORT_WARNING=1          # Clear output
export RAY_BACKEND_LOG_LEVEL=warning         # Lower the worker log level (strongly recommended)

# ---------- PyTorch / CUDA / vLLM Configuration ----------
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export NCCL_DEBUG=WARN
# export RAY_ENABLE_RECORD_ACTOR_TASK_LOGGING=1  # To debug the execution details of an actor, you can enable

# ---------- Start Ray ----------
echo "[INFO] Ensuring no existing Ray cluster is conflicting..."
if ray status >/dev/null 2>&1; then
  echo "[WARN] An existing Ray cluster is running. Skipping start (or stop it first with 'ray stop')." >&2
else
  echo "[INFO] Starting Ray head node..."
  ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265 --include-dashboard=true || { echo "[ERROR] Failed to start Ray" >&2; exit 1; }
fi

MODEL_DIR=${1}
WANDB_KEY=${2}
ROOT_DIR=$(pwd)

DATE=$(date +%m%d)
ADVANTAGE="group_norm"
SHORT_NAME="Qwen2.5-3B-Instruct"
TASK="search_r1"
ALGO="multiagent-search"
PRETRAIN="${MODEL_DIR}/${SHORT_NAME}"
EXP="${DATE}-${TASK}-${SHORT_NAME}-${ADVANTAGE}-${ALGO}"

SAVE_PATH="/data/outputs/${ADVANTAGE}-${ALGO}/${DATE}/${SHORT_NAME}/model"

PROMPT_DATA="json@/data/local/${TASK}"
TENSORBOARD="${ROOT_DIR}/logs/tensorboard/${ADVANTAGE}-${ALGO}-${DATE}-${SHORT_NAME}"
CKPT_PATH="/data/outputs/${ADVANTAGE}-${ALGO}/${DATE}/${SHORT_NAME}/ckpt"

mkdir -p "${ROOT_DIR}/logs"
mkdir -p "${ROOT_DIR}/logs/std"
mkdir -p "${ROOT_DIR}/logs/tensorboard"
mkdir -p "${ROOT_DIR}/outputs"

PROMPT_MAX_LEN=1024
GENERATE_MAX_LEN=2048

ENV_JSON=$(cat <<EOF
{
  "working_dir": "${ROOT_DIR}",
  "excludes": ["data/", "outputs/", ".git/", "local/", "logs/"],
  "pip": ["hydra-core", "antlr4-python3-runtime==4.9.3", "shortuuid", "class_registry", "json5", "mcp[cli]"]
}
EOF
)

ray job submit --address="http://localhost:8265" \
    --runtime-env-json="${ENV_JSON}" \
  -- python -m marti.cli.commands.train --config-name "ma_search_w_tool" \
  workflow_version=new \
    async_workflow=True \
    parallel_loading=True \
    default_agent.is_reasoning_model=False \
    default_agent.ref_num_nodes=1 \
    default_agent.ref_num_gpus_per_node=2 \
    default_agent.critic_num_nodes=0 \
    default_agent.critic_num_gpus_per_node=0 \
    default_agent.actor_num_nodes=1 \
    default_agent.actor_num_gpus_per_node=2 \
    default_agent.vllm_num_engines=4 \
    default_agent.vllm_tensor_parallel_size=2 \
    default_agent.vllm_sync_backend="nccl" \
    default_agent.colocate_all_models=False \
    default_agent.vllm_enable_sleep=True \
    default_agent.deepspeed_enable_sleep=True \
    default_agent.vllm_gpu_memory_utilization=0.99 \
    default_agent.pretrain="${PRETRAIN}" \
    default_agent.save_path="${SAVE_PATH}" \
    default_agent.micro_train_batch_size=4 \
    default_agent.train_batch_size=16 \
    default_agent.num_episodes=1 \
    default_agent.save_steps=100 \
    default_agent.eval_steps=5 \
    default_agent.logging_steps=1 \
    default_agent.max_samples=400000 \
    default_agent.micro_rollout_batch_size=8 \
    default_agent.rollout_batch_size=16 \
    default_agent.training_mode="rl" \
    default_agent.n_samples_per_prompt=8 \
    default_agent.max_epochs=1 \
    default_agent.prompt_max_len=${PROMPT_MAX_LEN} \
    default_agent.generate_max_len=${GENERATE_MAX_LEN} \
    default_agent.advantage_estimator=${ADVANTAGE} \
    default_agent.temperature=0.7 \
    default_agent.lambd=1.0 \
    default_agent.gamma=1.0 \
    default_agent.zero_stage=3 \
    default_agent.bf16=True \
    default_agent.actor_learning_rate=1e-6 \
    default_agent.critic_learning_rate=9e-6 \
    default_agent.init_kl_coef=0.01 \
    default_agent.use_kl_loss=True \
    default_agent.max_ckpt_num=100 \
    default_agent.normalize_reward=True \
    default_agent.adam_offload=True \
    default_agent.gradient_checkpointing=True \
    default_agent.ckpt_path="${CKPT_PATH}" \
    workflow_args.num_rounds=2 \
    workflow_func_path="marti/worlds/workflows/masearch_workflow.py" \
    processor_func_path="marti/worlds/workflows/default_processor.py" \
    tools_config.num_workers=512 \
    tools_config.tools.search.base_url="http://127.0.0.1:8080/retrieve" \
    tools_config.max_turns=2 \
    reward_alloc.name="margin" \
    reward_alloc.alpha=0.5 \
    reward_alloc.beta=0.5 \
    reward_alloc.use_ttrl=True \
    eval_before_training=True \
    eval_only=False \
    eval_workers=-1 \
    mask_truncated_completions=True \
    shared_agents=False \
    packing_samples=True \
    prompt_data="${PROMPT_DATA}" \
    input_key="question" \
    label_key="golden_answers" \
    apply_chat_template=False \
    add_prompt_suffix=null \
    use_wandb="${WANDB_KEY}" \
    wandb_project="MARTI" \
    wandb_run_name="${EXP}" \
    extra_eval_tasks=["musique","bamboogle"] \
    extra_eval_dir="/data/local/bench" \
    use_tensorboard="${TENSORBOARD}" 2>&1 | tee "${ROOT_DIR}/logs/std/${DATE}-${EXP}.log"

echo "Model Training Finished. Shutting down..."