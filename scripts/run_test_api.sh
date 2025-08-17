#!/bin/bash
# Inference script for API-based multi‑agent debate via marti.cli.commands.test_api
# Reference style: run_test_mas.sh
#
# Usage:
#   ./run_test_api.sh <API_KEY> [API_MODEL_NAME] [API_BASE_URL] [CONFIG_1 CONFIG_2 ...]
# Example:
#   ./run_test_api.sh "$OPENAI_API_KEY" gpt-4o-mini https://api.openai.com/v1 ma_chain_api
# If no configs are provided, a default list will be used.

set -euo pipefail

if [ ${#} -lt 1 ]; then
  echo "Usage: $0 <API_KEY> [API_MODEL_NAME] [API_BASE_URL] [CONFIGS...]" >&2
  exit 1
fi

API_KEY=${1}
API_MODEL_NAME=${2:-gpt-4o-mini}
API_BASE_URL=${3:-https://api.openai.com/v1}

# Determine how many of the first three optional fixed args were provided to shift accordingly
FIXED_ARGS=1
[ ${#} -ge 2 ] && FIXED_ARGS=2
[ ${#} -ge 3 ] && FIXED_ARGS=3
shift ${FIXED_ARGS}

if [ $# -gt 0 ]; then
  CFG_LIST=("$@")
  echo "Using provided config list: ${CFG_LIST[*]}"
else
  # Provide a sensible default config list (can adjust as new API-oriented configs are added)
  CFG_LIST=("ma_chain_api")
  echo "No configs specified. Using default configs: ${CFG_LIST[*]}"
fi

ROOT_DIR=$(pwd)
PROMPT_FILE="examples/test_api_prompts.json"
INPUT_KEY="problem"
OUTPUT_KEY="model_output"
TEMPERATURE=0.6
PROMPT_MAX_LEN=8192
GENERATE_MAX_LEN=8192
SAVE_ROOT="outputs/test_api"
SAVE_FILE="results.json"

if [ ! -f "$PROMPT_FILE" ]; then
  echo "Prompt file $PROMPT_FILE not found (expected relative to repo root)." >&2
  exit 2
fi


for config in "${CFG_LIST[@]}"; do
  SAVE_PATH="${SAVE_ROOT}/${config}/${API_MODEL_NAME}"
  mkdir -p "$SAVE_PATH"
  echo "Running config=$config model=$API_MODEL_NAME -> $SAVE_PATH/$SAVE_FILE"

  # NOTE: We override default_agent.save_path (existing key) and add +save_file (new key)
  python3 -m marti.cli.commands.test_api \
    --config-name "$config" \
    api_key="$API_KEY" \
    api_base_url="$API_BASE_URL" \
    api_model_name="$API_MODEL_NAME" \
    default_agent.prompt_max_len=$PROMPT_MAX_LEN \
    default_agent.generate_max_len=$GENERATE_MAX_LEN \
    default_agent.temperature=$TEMPERATURE \
    prompt_data="$PROMPT_FILE" \
    input_key="$INPUT_KEY" \
    output_key="$OUTPUT_KEY" \
    default_agent.save_path="$SAVE_PATH" \
    +save_file="$SAVE_FILE"
done

echo "All runs completed. Outputs in ${SAVE_ROOT}"
