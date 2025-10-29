from transformers import  AutoProcessor
from gptqmodel import GPTQModel, QuantizeConfig

model_path="../../Qwen/Qwen3-VL-2B-Instruct/"
quant_path = "../../Qwen/Qwen3-VL-2B-Instruct-GPTQ-4bit"

device = "cpu"

processor = AutoProcessor.from_pretrained(model_path)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "描述图片内容"},
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(device)
print("inputs_ids",inputs['input_ids'].tolist(),inputs['input_ids'].shape)
# keys: 'input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw'
# Inference: Generation of the output
print("inpuinputs['input_ids']ts.device",inputs['input_ids'].device)
# test post-quant inference
model = GPTQModel.load(quant_path, device_map={"": "cpu"} )

generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
print("generated_ids_trimmed",generated_ids_trimmed)
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
