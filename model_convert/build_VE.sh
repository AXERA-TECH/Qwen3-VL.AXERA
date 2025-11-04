# nohup pulsar2 build --input Qwen3-VL-2B-Instruct_vision_1280x736.onnx --config config.json --output_dir build-output-image-1280x736 --output_name Qwen3-VL-4B-Instruct_vision_1280x736.axmodel --target_hardware AX650 --compiler.check 0 > build_VE.log &

# nohup pulsar2 build --input Qwen3-VL-2B-Instruct_vision_640x640.onnx --config config_640.json --output_dir build-output-image-640x640 --output_name Qwen3-VL-4B-Instruct_vision_640x640.axmodel --target_hardware AX650 --compiler.check 0 > build_VE——640.log &

nohup pulsar2 build --input Qwen3-VL-2B-Instruct_vision_640x640_p1.onnx --config config_640.json --output_dir build-output-image-640x640-p1 --output_name Qwen3-VL-4B-Instruct_vision_640x640_p1.axmodel --target_hardware AX650 --compiler.check 0 > build_VE——640_p1.log &

nohup pulsar2 build --input Qwen3-VL-2B-Instruct_vision_640x640_p2.onnx --config config_640.json --output_dir build-output-image-640x640-p2 --output_name Qwen3-VL-4B-Instruct_vision_640x640_p2.axmodel --target_hardware AX650 --compiler.check 0 > build_VE——640_p2.log &