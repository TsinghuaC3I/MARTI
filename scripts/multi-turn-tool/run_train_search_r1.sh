#!/bin/bash
# Search-R1 training script for QA tasks with web search tool (1/1000 data subset)
set -x

source /mnt/shared-storage-user/marti/miniconda3/etc/profile.d/conda.sh
conda activate marti_vllm
which conda
which python

ROOT_DIR=""
cd ${ROOT_DIR}

# local retrieve server
SEARCH_INDEX_FILE="${SEARCH_INDEX_FILE:-/mnt/shared-storage-user/marti/sxw/Index/e5_Flat.index}"
SEARCH_CORPUS_FILE="${SEARCH_CORPUS_FILE:-/mnt/shared-storage-user/marti/sxw/Index/wiki-18.jsonl}"
RETRIEVER_NAME="${RETRIEVER_NAME:-e5}"
RETRIEVER_PATH="${RETRIEVER_PATH:-/mnt/shared-storage-user/marti/sxw/model/e5-base-v2}"
RETRIEVAL_SERVER_SCRIPT="${RETRIEVAL_SERVER_SCRIPT:-/mnt/shared-storage-user/marti/sxw/slime/examples/search-r1/local_dense_retriever/retrieval_server.py}"
SEARCH_PORT="${SEARCH_PORT:-8000}"

# Check if search service is already running
if curl -s --connect-timeout 2 --max-time 5 "http://127.0.0.1:${SEARCH_PORT}/retrieve" > /dev/null 2>&1; then
    echo "✅ Search service is already running on port ${SEARCH_PORT}"
else
    echo "Starting search service on port ${SEARCH_PORT}..."
    
    # Activate retriever environment and start search service in background
    (
        source /mnt/shared-storage-user/marti/sxw/miniconda3/etc/profile.d/conda.sh
        conda activate retriever
        
        python "${RETRIEVAL_SERVER_SCRIPT}" \
            --index_path "${SEARCH_INDEX_FILE}" \
            --corpus_path "${SEARCH_CORPUS_FILE}" \
            --topk 3 \
            --retriever_name "${RETRIEVER_NAME}" \
            --retriever_model "${RETRIEVER_PATH}" \
            > "${ROOT_DIR}/logs/search_service.log" 2>&1
    ) &
    
    SEARCH_PID=$!
    echo "Search service started with PID: ${SEARCH_PID}"
    
    # Wait for service to be ready 
    echo "Waiting for search service to be ready..."
    for i in {1..120}; do
        if curl -s --connect-timeout 2 --max-time 5 "http://127.0.0.1:${SEARCH_PORT}/retrieve" > /dev/null 2>&1; then
            echo "✅ Search service is ready!"
            break
        fi
        if [ $i -eq 120 ]; then
            echo "❌ Search service failed to start within 120 seconds"
            echo "Check logs at: ${ROOT_DIR}/logs/search_service.log"
            exit 1
        fi
        echo -n "."
        sleep 1
    done
    echo ""
fi

# Test search service with a sample query
echo "Testing search service..."
TEST_RESPONSE=$(curl -s --max-time 10 -X POST "http://127.0.0.1:${SEARCH_PORT}/retrieve" \
    -H "Content-Type: application/json" \
    -d '{"queries": ["test"], "topk": 1, "return_scores": false}')

if echo "$TEST_RESPONSE" | grep -q '"result"'; then
    echo "✅ Search service test passed"
else
    echo "⚠️  Search service test warning: unexpected response format"
    echo "Response: $TEST_RESPONSE"
fi

echo "=========================================="
echo ""

# Switch back to training environment
conda activate marti_vllm

MODEL_DIR="${MODEL_DIR:-/your_model_path}"
SHORT_NAME="${1:-Qwen3-4B}"
PRETRAIN="${MODEL_DIR}/${SHORT_NAME}"

PROMPT_MAX_LEN=8192
GENERATE_MAX_LEN=32768
EVAL_GENERATE_MAX_LEN=32768
MAX_LEN=40960
ADVANTAGE="group_norm"

TASK="search_r1"
ALGO="tool-search-r1-async-10"
DATE=$(date +%m%d)


PROMPT_DATA="${PROMPT_DATA:-${ROOT_DIR}/search_r1_data/nq_hotpotqa}"

# EXTRA_EVAL_TASKS='["nq","musique","bamboogle"]'
# EXTRA_EVAL_DIR="${ROOT_DIR}/data/Bench"


NUM_TASKS=128
EXP="${DATE}-${TASK}-${SHORT_NAME}-${ADVANTAGE}-${ALGO}"


SAVE_PATH="${ROOT_DIR}/outputs/final/${ADVANTAGE}-${SHORT_NAME}-${TASK}-${EXP}"
CKPT_PATH="${ROOT_DIR}/outputs/ckpt/${ADVANTAGE}-${SHORT_NAME}-${TASK}-${EXP}"
TENSORBOARD="${ROOT_DIR}/logs/tensorboard/${ADVANTAGE}-${SHORT_NAME}-${TASK}-${EXP}"
LOG_DIR="${ROOT_DIR}/logs/${ADVANTAGE}-${SHORT_NAME}-${TASK}-${EXP}.log"


# Search API URL - matches retrieval_server.py default port 8000
SEARCH_SERVICE_URL="${SEARCH_SERVICE_URL:-http://127.0.0.1:8000/retrieve}"
SEARCH_SCHEMA_PATH="${ROOT_DIR}/examples/schema/search.json"
MAX_TURNS=2


export MASTER_PORT=$(shuf -i 10000-65535 -n 1)
export OPENRLHF_ASYNC_NUM_TASKS=${NUM_TASKS}
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=true

export no_proxy=localhost,127.0.0.1 
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
        \"chat_template\": \"Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine inside <tool_call> and </tool_call> with JSON format {{\\\"name\\\": \\\"search\\\", \\\"arguments\\\": {{\\\"query_list\\\": [\\\"your query\\\"]}}}}, it will return the top searched results between <tool_response> and </tool_response>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\"
    }
}"

# Workflow arguments for search-augmented reasoning
WORKFLOW_ARGS="{
    \"task\": \"${TASK}\"
}"

TOOL_ARGS="{
    \"max_turns\": ${MAX_TURNS},
    \"num_workers\": 128,
    \"enable_metrics\": true,
    \"enable_rate_limiting\": true,
    \"max_model_len\": ${MAX_LEN},
    \"max_tokens_per_turn\": 16384,
    \"tools\": {
        \"search\": {
            \"type\": \"search_r1\",
            \"enable_rate_limiting\": true,
            \"rate_limit\": 30,
            \"timeout\": 90,
            \"topk\": 3,
            \"base_url\": \"${SEARCH_SERVICE_URL}\",
            \"schema_path\": \"${SEARCH_SCHEMA_PATH}\"
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
    --train_batch_size 1024 \
    --micro_rollout_batch_size 1 \
    --rollout_batch_size 512 \
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
    --entropy_loss_coef 0 \
    --gamma 1.0 \
    --lambd 1.0 \
    --use_kl_loss \
    --kl_estimator k3 \
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
    --eval_steps 5 \
    --logging_steps 1 \
    --num_episodes 1 \
    --max_samples 1000000 \
    --prompt_data "${PROMPT_DATA}" \
    --prompt_split "train" \
    --eval_dataset "${PROMPT_DATA}" \
    --eval_split "test" \
    --eval_temperature 1.0 \
    --eval_n_samples_per_prompt 4 \
    --input_key="question" \
    --label_key="golden_answers" \
    --eval_before_training \
    --use_tensorboard "${TENSORBOARD}" 2>&1 | tee "${LOG_DIR}"

echo "Search-R1 Model Training Finished."


