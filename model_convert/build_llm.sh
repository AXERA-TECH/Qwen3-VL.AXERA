set -e

INPUT_DIR=../Qwen/Qwen3-VL-2B-Instruct
OUTPUT_DIR=../Qwen3-VL-2B-Instruct--AX650-C128_P1152_CTX2047

pulsar2 llm_build2 --input_path $INPUT_DIR \
                --output_path  $OUTPUT_DIR \
                --hidden_state_type bf16 \
                --prefill_len 1280 \
                --prefill_step_size 128 \
                --max_context 2048 \
                --chip AX650 \
                --parallel 8 
                
./tools/embed_process.sh $INPUT_DIR $OUTPUT_DIR