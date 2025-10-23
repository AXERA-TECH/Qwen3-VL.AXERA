import torch
from transformers import   AutoProcessor, AutoConfig
from qwen_vl_utils import process_vision_info
import sys 
from PIL import Image
from glob import glob
import numpy as np 
from modeling_qwen3_vl_export import Qwen3VLForConditionalGenerationONNX
from preprocess import Qwen2VLImageProcessorExport
from transformers.image_utils import PILImageResampling

checkpoint_dir = sys.argv[1] if len(sys.argv)>=2 else "../../Qwen/Qwen3-VL-2B-Instruct/"
# default: Load the model on the available device(s)
model = Qwen3VLForConditionalGenerationONNX.from_pretrained(
    checkpoint_dir, torch_dtype=torch.float32, device_map="cuda"
)

model.model.visual.init_onnx_session("Qwen3-VL-2B-Instruct_vision.onnx")
model.model.visual.forward = model.model.visual.forward_video

paths = sorted(glob("../video/*.jpg"))
print(paths)
frames = [ Image.open(p).resize((384,384)) for p in paths ]


text = "描述这个视频."
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": frames,
                "max_pixels": 384 * 384,
                "sample_fps": 4,
            },
            {"type": "text", "text": text},
        ],
    }
]

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
pixel_values, grid_thw = img_processor._preprocess(frames, do_resize=True, resample=PILImageResampling.BICUBIC, 
                                        do_rescale=False, do_normalize=False, 
                                        do_convert_rgb=True)

pixel_values = torch.from_numpy(pixel_values).to("cpu")
mean = torch.tensor(image_mean,dtype=torch.float32).reshape([1,1,1,3])*255
mean = mean.to("cpu")
std = torch.tensor(image_std,dtype=torch.float32).reshape([1,1,1,3])*255
std = std.to("cpu")
pixel_values = (pixel_values-mean)/std
pixel_values = pixel_values.permute(0,3,1,2)

#In Qwen 2.5 VL, frame rate information is also input into the model to align with absolute time.
# Preparation for inference
cfg = AutoConfig.from_pretrained(
        checkpoint_dir, trust_remote_code=True
    )
processor = AutoProcessor.from_pretrained(checkpoint_dir) 
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs, video_kwargs = process_vision_info([messages],  return_video_kwargs=True, image_patch_size= 16, return_video_metadata=True)
if video_inputs is not None:
    video_inputs, video_metadatas = zip(*video_inputs)
    video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
else:
    video_metadatas = None
    
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
    **video_kwargs,
    do_resize=False,
)
inputs = inputs.to("cuda")
print("input_ids",inputs['input_ids'])
inputs['pixel_values_videos'] = pixel_values
# Inference
generated_ids = model.generate(**inputs, max_new_tokens=2048)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)