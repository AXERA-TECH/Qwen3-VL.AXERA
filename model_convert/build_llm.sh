set -e

pulsar2 llm_build --input_path ../../Qwen/Qwen3-VL-4B-Instruct/ \
                --output_path ../../Qwen/Qwen3-VL-4B-Instruct-AX650-c128_p1152 \
                --model_type qwen3_vl_text \
                --kv_cache_len 2047 \
                --hidden_state_type bf16 \
                --prefill_len 128 \
                --last_kv_cache_len 128 \
                --last_kv_cache_len 256 \
                --last_kv_cache_len 384 \
                --last_kv_cache_len 512 \
                --last_kv_cache_len 640 \
                --last_kv_cache_len 768 \
                --last_kv_cache_len 896 \
                --last_kv_cache_len 1024 \
                --last_kv_cache_len 1152 \
                --chip AX650 \
                --parallel 16


./tools/embed_process.sh ../../Qwen/Qwen3-VL-4B-Instruct/ ../../Qwen/Qwen3-VL-4B-Instruct-AX650-c128_p1152