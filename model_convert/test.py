from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image 

model_path="../../Qwen/Qwen3-VL-2B-Instruct/"
# model_path="../../Qwen/Qwen3-VL-4B-Thinking/"
# default: Load the model on the available device(s)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_path, dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen3VLForConditionalGeneration.from_pretrained(
#     model_path,
#     dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

processor = AutoProcessor.from_pretrained(model_path)
path = "demo_720p.jpg"
img = Image.open(path).resize((1280,640))
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": img,
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
inputs = inputs.to(model.device)
print("inputs_ids",inputs['input_ids'].tolist(),inputs['input_ids'].shape)
# keys: 'input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw'
# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
print("generated_ids_trimmed",generated_ids_trimmed)
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
