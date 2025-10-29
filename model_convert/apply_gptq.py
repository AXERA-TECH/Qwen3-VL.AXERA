from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig
from transformers import AutoTokenizer
import os

# model_id = "meta-llama/Llama-3.2-1B-Instruct"
# quant_path = "Llama-3.2-1B-Instruct-gptqmodel-4bit"

model_id = "../../Qwen/Qwen3-VL-2B-Instruct/" # "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
quant_path = "../../Qwen/Qwen3-VL-2B-Instruct-GPTQ-4bit"



# calibration_dataset = load_dataset(
#     "allenai/c4",
#     data_files="en/c4-train.00001-of-01024.json.gz",
#     split="train"
#   ).select(range(1024))["text"]

# tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code = True)
# calibration_dataset = [
#         tokenizer(
#             "gptqmodel is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
#         )
#     ]

calibration_dataset = [
        [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "../demo.jpeg",
                },
                {"type": "text", "text": "描述图片内容"},
            ],
        }
    ]
]

quant_config = QuantizeConfig(
        bits=4,  # quantize model to 4-bit
        group_size=128,  # it is recommended to set the value to 128
        desc_act=False,
        static_groups=True,
        sym=True,
    )

model = GPTQModel.load(model_id, quant_config)

# increase `batch_size` to match gpu/vram specs to speed up quantization
model.quantize(calibration_dataset, batch_size=1)

model.save(quant_path)