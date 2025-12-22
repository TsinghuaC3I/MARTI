#!/bin/bash
set -x
ulimit -n 65535

# ---------- Ray Configuration ----------
RAY_TMPDIR="/fs-computility/mabasic/qibiqing/ray_temp"
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

echo "[INFO] Starting Ray head node..."
ray start --head \
  --port=6379 \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265 \
  --include-dashboard=true \
  --temp-dir="${RAY_TMPDIR}"

MODEL_DIR="/fs-computility/mabasic/shared/models/"
WANDB_KEY="43cfd199988bf094dfde21086e23d3dbfd1269e4"
ROOT_DIR=$(pwd)

DATE=$(date +%m%d)
ADVANTAGE="reinforce"
SHORT_NAME="Qwen2.5-3B-Instruct"
TASK="MATH"
ALGO="mathchat-3x2-unshared"
PRETRAIN="${MODEL_DIR}/${SHORT_NAME}"
EXP="${DATE}-${TASK}-${SHORT_NAME}-${ADVANTAGE}-${ALGO}"

SAVE_PATH="${ROOT_DIR}/outputs/${ADVANTAGE}-${ALGO}/${DATE}/${SHORT_NAME}/model"

PROMPT_DATA="json@${ROOT_DIR}/data/${TASK}"
TENSORBOARD="${ROOT_DIR}/logs/tensorboard/${ADVANTAGE}-${ALGO}-${DATE}-${SHORT_NAME}"
CKPT_PATH="${ROOT_DIR}/outputs/${ADVANTAGE}-${ALGO}//${DATE}/${SHORT_NAME}/ckpt"

mkdir -p "${ROOT_DIR}/logs"
mkdir -p "${ROOT_DIR}/logs/std"
mkdir -p "${ROOT_DIR}/logs/tensorboard"
mkdir -p "${ROOT_DIR}/outputs"

PROMPT_MAX_LEN=16384
GENERATE_MAX_LEN=4096

export PYTORCH_NVML_BASED_CUDA_CHECK=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=WARN
export RAY_PICKLE_VERBOSE_DEBUG=1

ENV_JSON=$(cat <<EOF
{
  "working_dir": "${ROOT_DIR}",
  "excludes": ["data/", "outputs/", ".git/", "local/", "logs/"],
  "pip": ["hydra-core", "antlr4-python3-runtime==4.9.3", "shortuuid", "class_registry", "json5", "mcp[cli]", "func_timeout"]
}
EOF
)

ray job submit --address="http://localhost:8265" \
    --runtime-env-json="${ENV_JSON}" \
    -- python -m marti.cli.commands.train --config-name "ma_mathchat_w_tool" \
    async_workflow=True \
    parallel_loading=True \
    default_agent.is_reasoning_model=False \
    default_agent.ref_num_nodes=1 \
    default_agent.ref_num_gpus_per_node=2 \
    default_agent.critic_num_nodes=1 \
    default_agent.critic_num_gpus_per_node=2 \
    default_agent.actor_num_nodes=1 \
    default_agent.actor_num_gpus_per_node=2 \
    default_agent.vllm_num_engines=2 \
    default_agent.vllm_tensor_parallel_size=1 \
    default_agent.vllm_sync_backend="nccl" \
    default_agent.colocate_all_models=True \
    default_agent.vllm_enable_sleep=True \
    default_agent.deepspeed_enable_sleep=True \
    default_agent.vllm_gpu_memory_utilization=0.6 \
    default_agent.pretrain="${PRETRAIN}" \
    default_agent.save_path="${SAVE_PATH}" \
    default_agent.micro_train_batch_size=4 \
    default_agent.train_batch_size=128 \
    default_agent.num_episodes=5 \
    default_agent.save_steps=1000 \
    default_agent.eval_steps=5 \
    default_agent.logging_steps=1 \
    default_agent.max_samples=400000 \
    default_agent.micro_rollout_batch_size=8 \
    default_agent.rollout_batch_size=128 \
    default_agent.training_mode="rl" \
    default_agent.n_samples_per_prompt=8 \
    default_agent.max_epochs=1 \
    default_agent.prompt_max_len=${PROMPT_MAX_LEN} \
    default_agent.generate_max_len=${GENERATE_MAX_LEN} \
    default_agent.advantage_estimator=${ADVANTAGE} \
    default_agent.temperature=1.0 \
    default_agent.lambd=1.0 \
    default_agent.gamma=1.0 \
    default_agent.zero_stage=3 \
    default_agent.bf16=True \
    default_agent.actor_learning_rate=1e-6 \
    default_agent.critic_learning_rate=9e-6 \
    default_agent.init_kl_coef=0.00 \
    default_agent.use_kl_loss=True \
    default_agent.max_ckpt_num=100 \
    default_agent.normalize_reward=True \
    default_agent.adam_offload=True \
    default_agent.gradient_checkpointing=True \
    default_agent.ckpt_path="${CKPT_PATH}" \
    workflow_args.num_rounds=2 \
    workflow_func_path="marti/worlds/workflows/mathchat_workflow.py" \
    processor_func_path="marti/worlds/workflows/default_processor.py" \
    tools_config.num_workers=512 \
    tools_config.tools.code_interpreter.base_url="http://101.6.64.188:10086/run_code" \
    tools_config.tools.code_interpreter.rate_limit=256 \
    reward_alloc.name="margin" \
    reward_alloc.alpha=0.5 \
    reward_alloc.beta=0.5 \
    reward_alloc.use_ttrl=False \
    eval_before_training=False \
    eval_only=False \
    eval_workers=-1 \
    mask_truncated_completions=True \
    shared_agents=False \
    packing_samples=True \
    extra_eval_tasks=["aime","amc"] \
    extra_eval_dir="${ROOT_DIR}/data/Bench" \
    prompt_data="${PROMPT_DATA}" \
    input_key="prompt" \
    label_key="answer" \
    add_prompt_suffix=null \
    wandb_project="MARTI" \
    wandb_run_name="${EXP}" \
    use_tensorboard="${TENSORBOARD}" 2>&1 | tee "${ROOT_DIR}/logs/std/${DATE}-${EXP}.log"

    # use_wandb="${WANDB_KEY}" \

echo "Model Training Finished. Shutting down..."