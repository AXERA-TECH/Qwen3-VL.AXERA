#!/usr/bin/env bash
set -euo pipefail

# Set these variables to directories visible from the conversion environment.
# The adapter directory name becomes the runtime task_id, so keep each name
# unique and use only letters, digits, '.', '_' or '-'.
#
# Note: FLOAT_MATMUL_USE_CONV_EU is not supported for this LoRA build and must
# stay unset. Enabling it fails while building the LoRA branch.
MODEL_DIR="${MODEL_DIR:-../Qwen/Qwen3-VL-4B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-../Qwen/Qwen3-VL-4B-Instruct-LoRA-AX650-P1536-C2048}"
ADAPTER_CHARTQA_DIR="${ADAPTER_CHARTQA_DIR:-../Qwen/qwen3-vl-lora-chartqa}"
ADAPTER_DESIGN_DIR="${ADAPTER_DESIGN_DIR:-../Qwen/qwen3-vl-lora-design}"

pulsar2 llm_build2 \
  --input_path "$MODEL_DIR" \
  --output_path "$OUTPUT_DIR" \
  --hidden_state_type bf16 \
  --weight_type s8 \
  --post_weight_type s8 \
  --prefill_len 1536 \
  --prefill_step_size 128 \
  --max_context 2048 \
  --decode_step_size -1 \
  --chip AX650 \
  --parallel 8 \
  --tensor_parallel_size 0 \
  --check_level 0 \
  --lora_adapter_path "$ADAPTER_CHARTQA_DIR" \
  --lora_adapter_path "$ADAPTER_DESIGN_DIR"

# Generate the runtime BF16 token embedding expected by axllm.
# embed_process.sh resolves tools/ relatively, so run this script from
# model_convert/ (the default MODEL_DIR/OUTPUT_DIR above are relative too).
./tools/embed_process.sh "$MODEL_DIR" "$OUTPUT_DIR"
