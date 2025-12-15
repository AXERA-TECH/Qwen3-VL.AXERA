pulsar2 build --input Qwen3-VL-2B-Instruct_vision.onnx \
                --config config.json \
                --output_dir build-output-image-2b \
                --output_name Qwen3-VL-2B-Instruct_vision.axmodel \
                --target_hardware AX650 \
                --compiler.check 0