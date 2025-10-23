#!/bin/bash
# gspo + tis + datafilter + overlong
set -x
source /mnt/shared-storage-user/marti/miniconda3/etc/profile.d/conda.sh
conda activate marti_vllm
cd /mnt/shared-storage-user/marti/lipengfei/OpenRLHF-0.8.9


MODEL_DIR="/mnt/shared-storage-user/marti/models"
# /mnt/shared-storage-user/marti/models/models--Qwen--Qwen2.5-Coder-7B-Instruct/snapshots/c03e6d358207e414f1eca0bb1891e29f1db0e242
SHORT_NAME=${1:-"Qwen3-8B"}
PRETRAIN="${MODEL_DIR}/${SHORT_NAME}"
PROMPT_MAX_LEN=3072
GENERATE_MAX_LEN=32768
EVAL_GENERATE_MAX_LEN=32768
OVERLONG_BUFFER_LEN=2048
MAX_LEN=40000
ADVANTAGE="group_norm"

ROOT_DIR="/mnt/shared-storage-user/marti/lipengfei/OpenRLHF-0.8.9"
TASK="CODE_8B_FILTER_FT"
PROMPT_DATA="json@/mnt/shared-storage-user/marti/lipengfei/MARTI_DEV/data/${TASK}"

EXP=gspo-tis-df-overlong

TENSORBOARD="${ROOT_DIR}/logs/tensorboard/${ADVANTAGE}-${SHORT_NAME}-${TASK}-db-${EXP}-lfy"

   # --advantage_estimator $ADVANTAGE \

# Set a dynamic MASTER_PORT to avoid port conflict
export MASTER_PORT=$(shuf -i 10000-65535 -n 1)
# export WORLD_SIZE=8
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

   # --dynamic_filtering \

python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 8 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 8 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 8 \
   --vllm_num_engines 4 \
   --vllm_tensor_parallel_size 2 \
   --colocate_all_models \
   --vllm_gpu_memory_utilization 0.6 \
   --init_kl_coef 1e-3 \
   --gamma 1.0 \
   --use_kl_loss \
   --kl_estimator k3 \
   --pretrain $PRETRAIN \
   --dynamic_filtering_reward_range 0 1 \
   --overlong_buffer_len ${OVERLONG_BUFFER_LEN} \
   --enable_vllm_is_correction \
   --policy_loss_type "gspo" \
   --advantage_estimator $ADVANTAGE \
   --agent_func_path /mnt/shared-storage-user/marti/OpenRLHF/examples/python/agent_func_lpf.py \
   --save_path /mnt/shared-storage-user/marti/OpenRLHF/outputs/final/${SHORT_NAME}-${EXP} \
   --ckpt_path /mnt/shared-storage-user/marti/OpenRLHF/outputs/ckpt/${SHORT_NAME}-${EXP} \
   --save_hf_ckpt \
   --micro_train_batch_size 1 \
   --train_batch_size 32 \
   --micro_rollout_batch_size 1 \
   --rollout_batch_size 32 \
   --n_samples_per_prompt 8 \
   --save_steps 10 \
   --eval_steps 5 \
   --max_epochs 1 \
   --num_episodes 2 \
   --max_len ${MAX_LEN} \
   --prompt_max_len ${PROMPT_MAX_LEN} \
   --max_samples 100000 \
   --generate_max_len ${GENERATE_MAX_LEN} \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --prompt_data ${PROMPT_DATA} \
   --eval_dataset ${PROMPT_DATA} \
   --eval_split "test" \
   --eval_temperature 0.6 \
   --eval_n_samples_per_prompt 4 \
   --input_key="prompt" \
   --label_key="label" \
   --apply_chat_template \
   --normalize_reward \
   --gradient_checkpointing \
   --packing_samples \
   --load_checkpoint \
   --vllm_sync_backend nccl \
   --enforce_eager \
   --vllm_enable_sleep \
   --deepspeed_enable_sleep \
   --use_tensorboard "${TENSORBOARD}" 2>&1 | tee /mnt/shared-storage-user/marti/lipengfei/OpenRLHF-0.8.9/logs/gspo_tis_datafilter_overlong_openrhf_qwen3-8B.log \

# You could also try
#   --kl_estimator k2 \
