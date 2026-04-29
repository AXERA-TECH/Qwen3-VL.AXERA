import os

import torch
from gptqmodel import GPTQModel
from PIL import Image
from transformers import AutoProcessor


MODEL_PATH = os.environ.get("MODEL_PATH", "../../Qwen/Qwen3-VL-2B-Instruct/")
QUANT_PATH = os.environ.get("QUANT_PATH", "../../Qwen/Qwen3-VL-2B-Instruct-GPTQ-Int4")
IMAGE_PATH = os.environ.get("IMAGE_PATH", "../demo.jpeg")
QUESTION = os.environ.get("QUESTION", "请只根据图片中清楚可见的内容，客观描述。不要猜测人物关系、地点、时间、情绪或看不见的细节。")
DEVICE = os.environ.get("DEVICE", "cuda:0")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "2048"))

def contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def system_prompt(question: str) -> str:
    if contains_cjk(question):
        return "请使用和用户问题相同的语言回答。本轮用户使用中文提问，所以只用简体中文回答；不要夹杂英文，不要重复同一句话。"
    return "Answer in the same language as the user's question. The user asks in English, so answer only in English. Do not mix languages or repeat the same sentence."


processor = AutoProcessor.from_pretrained(MODEL_PATH)
img = Image.open(IMAGE_PATH).convert("RGB").resize((384, 384))

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": system_prompt(QUESTION)}],
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": QUESTION},
        ],
    },
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
).to(DEVICE)

print("input_ids", inputs["input_ids"].tolist(), inputs["input_ids"].shape)
print("input_ids.device", inputs["input_ids"].device)

model = GPTQModel.load(QUANT_PATH, device_map={"": DEVICE})

with torch.inference_mode():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        repetition_penalty=1.12,
        no_repeat_ngram_size=8,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.eos_token_id,
    )

generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
print("generated_ids_trimmed", generated_ids_trimmed)
output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)
print(output_text)
