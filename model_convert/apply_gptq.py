from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig
from transformers import AutoTokenizer
import os
import json
import random 
from PIL import Image

model_id = "../../Qwen/Qwen3-VL-2B-Instruct/" 
quant_path = "../../Qwen/Qwen3-VL-2B-Instruct-GPTQ-Int4"


# https://huggingface.co/datasets/lmms-lab/COCO-Caption
dataset = load_dataset("parquet", data_files="val-00001-of-00013.parquet", split="train").shuffle()

calibration_dataset = []

for i in range(512):
    item = dataset[i]
    d = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": item["image"].resize((384,384))
                        },
                        {"type": "text", "text": item['question']},
                    ],
                }
            ]
    calibration_dataset.append(d)

quant_config = QuantizeConfig(
        bits=4,  # quantize model to 4-bit
        group_size=128,  # it is recommended to set the value to 128
        desc_act=False,
        static_groups=True,
        sym=True,
        v2=True,
        mse=2.5,
        v2_memory_device="auto"  
    )

model = GPTQModel.load(model_id, quant_config)

# increase `batch_size` to match gpu/vram specs to speed up quantization
model.quantize(calibration_dataset, batch_size=8)

model.save(quant_path)