from transformers import AutoTokenizer, AutoConfig
import numpy as np
import math
from ml_dtypes import bfloat16
from axengine import InferenceSession
from PIL import Image
from torchvision import transforms
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import torch
from transformers import  AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import onnxruntime
import gc
from glob import glob
from utils import get_rope_index
from transformers.image_utils import PILImageResampling
from preprocess import Qwen2VLImageProcessorExport

def post_process(data, topk=20, topp=0.8, temperature=0.7):
    def top_p(l: np.ndarray, p: float) -> np.ndarray:
        index = np.argsort(l)
        res = l.copy()
        sum_p = 0
        for i in index[::-1]:
            if sum_p >= p:
                res[i] = 0
            sum_p += res[i]
        return res / sum_p

    def softmax(l: np.ndarray) -> np.ndarray:
        l_max = l - l.max()
        l_exp = np.exp(l_max)
        res = l_exp / np.sum(l_exp)
        return res.astype(np.float64)

    r = data.astype(np.float32)
    r = r.flatten()
    # topk
    candidate_index = np.argpartition(r, -topk)[-topk:]
    candidate_value = r[candidate_index]
    # temperature
    candidate_value /= temperature
    # softmax
    candidate_soft = softmax(candidate_value)
    # topp
    candidate_soft = top_p(candidate_soft, topp)
    candidate_soft = candidate_soft.astype(np.float64) / candidate_soft.sum()
    pos = np.random.multinomial(1, candidate_soft).argmax()
    next_token = candidate_index[pos]
    return next_token, candidate_index, candidate_soft



if __name__ == "__main__":

    prefill_len = 1152
    chunk_len = 128
    checkpoint_dir=f"../../Qwen/Qwen3-VL-4B-Instruct-AX650-c128_p1152/"
    cfg = AutoConfig.from_pretrained(
        "../../Qwen/Qwen3-VL-4B-Instruct", trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "../../Qwen/Qwen3-VL-4B-Instruct", trust_remote_code=True
    )
        
    processor = AutoProcessor.from_pretrained("../../Qwen/Qwen3-VL-4B-Instruct") 
    
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

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    position_ids,_ = get_rope_index(cfg, inputs["input_ids"], video_grid_thw=inputs['video_grid_thw'])
    print("position_ids",position_ids.shape)
    print(position_ids)
    # pixel_values = inputs['pixel_values_videos']
    # print("pixel_values",pixel_values.shape)
    # extract img feature by vit
    vit_session = InferenceSession(f'{checkpoint_dir}/Qwen3-VL-4B-Instruct_vision.axmodel')

    img_processor = Qwen2VLImageProcessorExport(max_pixels=384*384, patch_size=16, temporal_patch_size=2, merge_size=2)
    pixel_values, grid_thw = img_processor._preprocess(frames, do_resize=True, resample=PILImageResampling.BICUBIC, 
                                        do_rescale=False, do_normalize=False, 
                                        do_convert_rgb=True)

    # seq_len, dim = pixel_values.shape
    # ht = pixel_values.reshape(t, seq_len//t, dim)
    print("pixel_values.shape",pixel_values.shape)
    t, seq_len,_,_ = pixel_values.shape
    vit_output = []
    for i in range(t):
        out = vit_session.run(None, {"hidden_states": pixel_values[i:i+1]})[0]  
        vit_output.append(out.astype(bfloat16))
    
    del vit_session
    gc.collect()

    vit_output = np.concatenate(vit_output, axis=0)
    vit_output = vit_output[None,:,:]
    
    print("vit feature extract done!")

    token_ids = inputs['input_ids'].squeeze().numpy().tolist()
    print("token_ids",token_ids)
    print("np.where(np.array(token_ids) == 151652)[0]",np.where(np.array(token_ids) == 151652)[0])
    image_start_index = np.where(np.array(token_ids) == 151652)[0].tolist()[0]
    image_insert_index = image_start_index + 1
    embeds = np.load(f"{checkpoint_dir}/model.embed_tokens.weight.npy")
    prefill_data = np.take(embeds, token_ids, axis=0)
    prefill_data = prefill_data.astype(bfloat16)
    
    prefill_data[ image_insert_index : image_insert_index + vit_output.shape[1]] = vit_output[0, :, :]
    token_len = len(token_ids)


    lastN = 2047
    cfg = cfg.text_config
    
    d = cfg.hidden_size
    head = cfg.num_attention_heads
    epsilon = cfg.rms_norm_eps
    assert d % head == 0
    sub_d = cfg.head_dim if cfg.head_dim else d // head
    kv_dim = sub_d * cfg.num_key_value_heads
    k_caches = [
        np.zeros((1, lastN, kv_dim), dtype=bfloat16)
        for _ in range(cfg.num_hidden_layers)
    ]
    v_caches = [
        np.zeros((1, lastN, kv_dim), dtype=bfloat16)
        for _ in range(cfg.num_hidden_layers)
    ]

    prefill_decoder_sessins = []
    for i in range(cfg.num_hidden_layers):
        if chunk_len > 0:
            session = InferenceSession(
                f"{checkpoint_dir}/qwen3_vl_text_p{chunk_len}_l{i}_together.axmodel"
            )
        else:
            session = InferenceSession(
                f"{checkpoint_dir}/qwen3_vl_text_p{prefill_len}_l{i}_together.axmodel"
            )
        prefill_decoder_sessins.append(session)
    post_process_session = InferenceSession(
        f"{checkpoint_dir}/qwen3_vl_text_post.axmodel"
    )
    print("model load done!")

    """
        prefill
    """

    if prefill_len > 0:
        indices = np.zeros((3, prefill_len), dtype=np.uint32)

        indices[:, 0:token_len] = position_ids.squeeze(1).numpy().astype(np.uint32)

        mask = np.zeros((1, prefill_len, prefill_len)) - 65536
        data = np.zeros((1, prefill_len, cfg.hidden_size)).astype(bfloat16)
        
        data[:, 0:token_len] = prefill_data
        for i, t in enumerate(token_ids):
            mask[:, i, : i + 1] = 0
        mask = mask.astype(bfloat16)

        chunk_num = math.ceil(token_len / chunk_len)
        for i in range(cfg.num_hidden_layers):
            if chunk_len <= 0:
                input_feed = {
                    "K_cache": np.zeros((1, 1, cfg.hidden_size), dtype=bfloat16),
                    "V_cache": np.zeros((1, 1, cfg.hidden_size), dtype=bfloat16),
                    "indices": indices,
                    "input": data,
                    "mask": mask,
                }
                outputs = prefill_decoder_sessins[i].run(None, input_feed, shape_group=1)

                k_caches[i][:, :token_len, :] = outputs[0][:, :token_len, :]
                v_caches[i][:, :token_len, :] = outputs[1][:, :token_len, :]
                data[:, 0:token_len] = outputs[2][:, :token_len, :]
            else:
                last_layer_output = []
                for ck in range(chunk_num):
                    
                    gid = ck + 1
                    if ck==0:
                        input_feed = {
                            "K_cache": np.zeros((1, 1, cfg.hidden_size), dtype=bfloat16),
                            "V_cache": np.zeros((1, 1, cfg.hidden_size), dtype=bfloat16),
                            "indices": indices[:, 0:chunk_len],
                            "input": data[:, 0:chunk_len],
                            "mask": mask[:, 0:chunk_len, 0:chunk_len],
                        }
                        outputs = prefill_decoder_sessins[i].run(None, input_feed, shape_group=gid)
                        k_caches[i][:, :chunk_len, :] = outputs[0][:, :chunk_len, :]
                        v_caches[i][:, :chunk_len, :] = outputs[1][:, :chunk_len, :]

                    else:
                        input_feed = {
                            "K_cache": k_caches[i][:, :ck*chunk_len, :],
                            "V_cache": v_caches[i][:, :ck*chunk_len, :],
                            "indices": indices[:, ck*chunk_len:(ck+1)*chunk_len],
                            "input": data[:, ck*chunk_len:(ck+1)*chunk_len],
                            "mask": mask[:, ck*chunk_len:(ck+1)*chunk_len, 0:(ck+1)*chunk_len],
                        }
                        outputs = prefill_decoder_sessins[i].run(None, input_feed, shape_group=gid)
                        k_caches[i][:, ck*chunk_len:(ck+1)*chunk_len, :] = outputs[0][:, :chunk_len, :]
                        v_caches[i][:, ck*chunk_len:(ck+1)*chunk_len, :] = outputs[1][:, :chunk_len, :]

                    last_layer_output.append(outputs[2][:, :chunk_len, :])
                
                data = np.concatenate(last_layer_output, axis=1)

    post_out = post_process_session.run(None, {"input": data[:, token_len - 1:token_len, :]})[0]
    next_token, posssible_tokens, possible_soft = post_process(post_out, topk=1)
    posibles = [tokenizer.decode([t]) for t in posssible_tokens]
    posible_soft = [str((t, s)) for t, s in zip(posibles, possible_soft)]
    token_ids.append(next_token)
    print("prefill done!")
  

    # lastN = np.max(indices)
    start_ids = np.max(indices) + 1
    mask = np.zeros((1, 1, lastN + 1), dtype=np.float32).astype(bfloat16)
    mask[:, :, :lastN] -= 65536
    mask[:, :, :token_len] = 0
    for start_indice in range(lastN + 1):
        if prefill_len > 0 and start_indice < token_len:
            continue
        next_token = token_ids[start_indice]
        indices = np.array([start_ids], np.uint32).reshape((1, 1))
        start_ids += 1
        data = embeds[next_token, :].reshape((1, 1, cfg.hidden_size)).astype(bfloat16)

        for i in range(cfg.num_hidden_layers):
            input_feed = {
                "K_cache": k_caches[i],
                "V_cache": v_caches[i],
                "indices": indices,
                "input": data,
                "mask": mask,
            }
            outputs = prefill_decoder_sessins[i].run(None, input_feed, shape_group=0)
            k_caches[i][:, start_indice, :] = outputs[0][:, :, :]
            v_caches[i][:, start_indice, :] = outputs[1][:, :, :]
            data = outputs[2]
        mask[..., start_indice] = 0
        if start_indice < token_len - 1:
            pass
        else:
            post_out = post_process_session.run(None, {"input": data})[0]
            next_token, posssible_tokens, possible_soft = post_process(post_out)
            token_ids.append(next_token)
            print(tokenizer.decode(next_token))
        if next_token == tokenizer.eos_token_id:
            # print("hit eos!")
            break
    print(tokenizer.decode(token_ids[token_len:]))
    
    
    