import torch
from transformers import  AutoProcessor
from PIL import Image
from preprocess import Qwen2VLImageProcessorExport
from modeling_qwen3_vl_export import Qwen3VLForConditionalGenerationONNX
from PIL import Image
from transformers.image_utils import PILImageResampling
from preprocess import Qwen2VLImageProcessorExport

model_path="../../Qwen/Qwen3-VL-2B-Instruct/"
# model_path="../../Qwen/Qwen3-VL-4B-Thinking/"
# default: Load the model on the available device(s)

device="cuda"

model = Qwen3VLForConditionalGenerationONNX.from_pretrained(
    model_path, dtype="auto", device_map=device
)
model.model.visual.init_onnx_session("Qwen3-VL-2B-Instruct_vision.onnx")
model.model.visual.forward = model.model.visual.forward_image

processor = AutoProcessor.from_pretrained(model_path)

path = "../demo.jpeg"
img = Image.open(path).resize((384,384))
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

images = [img]

img_processor = Qwen2VLImageProcessorExport(max_pixels=384*384, patch_size=16, temporal_patch_size=2, merge_size=2)

image_mean = [
    0.5,
    0.5,
    0.5
  ]

image_std =  [
    0.5,
    0.5,
    0.5
  ]
# pixel_values, grid_thw = img_processor._preprocess(images, do_resize=True, resample=PILImageResampling.BICUBIC, 
#                                     do_rescale=True, rescale_factor=1/255, do_normalize=True, 
#                                     image_mean=image_mean, image_std=image_std,do_convert_rgb=True)
pixel_values, grid_thw = img_processor._preprocess(images, do_resize=True, resample=PILImageResampling.BICUBIC, 
                                        do_rescale=False, do_normalize=False, 
                                        do_convert_rgb=True)
print("pixel_values.shape",pixel_values.shape)
print("grid_thw",grid_thw)
t,seq_len,tpp,_ = pixel_values.shape

pixel_values = torch.from_numpy(pixel_values).to(device)
mean = torch.tensor(image_mean,dtype=torch.float32).reshape([1,1,1,3])*255
mean = mean.to(device)
std = torch.tensor(image_std,dtype=torch.float32).reshape([1,1,1,3])*255
std = std.to(device)
pixel_values = (pixel_values-mean)/std

pixel_values = pixel_values.permute(0,3,1,2).to(device)


# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

inputs["image_grid_thw"] = torch.tensor(grid_thw).reshape(1,3)
inputs['pixel_values'] = pixel_values

# keys: 'input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw'
# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
