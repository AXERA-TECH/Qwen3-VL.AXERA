import torch
import onnx
from onnx.shape_inference import infer_shapes
import onnxsim
import onnx
from onnx import helper
from transformers import  AutoTokenizer, AutoProcessor
from modeling_qwen3_vl_export import Qwen3VLForConditionalGenerationExport
from qwen_vl_utils import process_vision_info
import numpy as np 
import os
import sys 

def export_onnx(model, input, input_names, output_names, onnx_output):

    torch.onnx.export(
        model,
        input,
        onnx_output,
        input_names=input_names,
        output_names=output_names,
        opset_version=16,
    )

    # onnx_model = onnx.load(onnx_output)
    # print("IR 版本:", onnx_model.ir_version)
    # print("操作集:", onnx_model.opset_import)
    # onnx_model = infer_shapes(onnx_model)
    # # convert model
    # model_simp, check = onnxsim.simplify(onnx_model)
    # assert check, "Simplified ONNX model could not be validated"
    # onnx.save(model_simp, onnx_output)
    # print("onnx simpilfy successed, and model saved in {}".format(onnx_output))


checkpoint_dir = sys.argv[1] if len(sys.argv)>=2 else "../../Qwen/Qwen3-VL-2B-Instruct/"
which = sys.argv[2] if len(sys.argv)>=3 else "image"
onnx_output = sys.argv[3] if len(sys.argv)>=4 else "Qwen3-VL-2B-Instruct_vision.onnx"
# default: Load the model on the available device(s)
model = Qwen3VLForConditionalGenerationExport.from_pretrained(
    checkpoint_dir, torch_dtype=torch.float32, device_map="cpu"
)

export_model = model.model.visual

if which=="image":
    export_model.forward = export_model.forward_chunk1
elif which=="video":
    export_model.forward = export_model.forward_export_by_second_nchw
else:
    raise NotImplementedError
device = torch.device("cpu")


# hidden_states = torch.load("hidden_states.pth",weights_only=True).to(torch.float32).to(device)
hidden_states = torch.ones((1,3, 1600,512) ).to(torch.float32).to(device)
print("hidden_states",hidden_states.shape)

input = ( hidden_states)

input_names = ["hidden_states"]



output_names = [f"hidden_states_out", "deepstack_feature_0", "deepstack_feature_1", "deepstack_feature_2"]


export_onnx(export_model, input, input_names, output_names, "Qwen3-VL-2B-Instruct_vision_640x640_p1.onnx")    



if which=="image":
    export_model.forward = export_model.forward_chunk2
elif which=="video":
    export_model.forward = export_model.forward_export_by_second_nchw
else:
    raise NotImplementedError
device = torch.device("cpu")




input = ( hidden_states)

input_names = ["hidden_states"]



output_names = [f"hidden_states_out", "deepstack_feature_0", "deepstack_feature_1", "deepstack_feature_2"]


export_onnx(export_model, input, input_names, output_names, "Qwen3-VL-2B-Instruct_vision_640x640_p2.onnx")    

