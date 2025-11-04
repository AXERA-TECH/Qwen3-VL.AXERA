import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, Callable, Optional, Union
import math
import numpy as np
import onnxruntime as ort
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLModel, Qwen3VLVisionModel


class Qwen3VLVisionModelInfer(Qwen3VLVisionModel):
    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)

    def forward_image(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        torch.save(hidden_states, "hidden_states.pth")
        t, channel, seq_len, tpp = hidden_states.shape
        assert t==1 
        hidden_states = hidden_states.permute(0,2,1,3).reshape(t,seq_len, channel*tpp)

        hidden_states = self.patch_embed(hidden_states)

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        torch.save(pos_embeds, "pos_embeds.pth")
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        torch.save(position_embeddings, "position_embeddings.pth")

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        torch.save(cu_seqlens, "cu_seqlens.pth")

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
                    hidden_states
                )
                deepstack_feature_lists.append(deepstack_feature)

        hidden_states = self.merger(hidden_states)

        return hidden_states, deepstack_feature_lists

    def forward_chunk(self, hidden_states, pos_embeds, pos_emb_cos, pos_emb_sin,  **kwargs):
        
        t, channel, seq_len, tpp = hidden_states.shape
        assert t==1 
        hidden_states = hidden_states.permute(0,2,1,3).reshape(t,seq_len, channel*tpp)

        hidden_states = self.patch_embed(hidden_states)

        hidden_states = hidden_states + pos_embeds

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        # rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        # emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        # position_embeddings = (emb.cos(), emb.sin())
        position_embeddings = (pos_emb_cos, pos_emb_sin)
        cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32).to(hidden_states.device)

    
        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
                    hidden_states
                )
                deepstack_feature_lists.append(deepstack_feature)

        hidden_states = self.merger(hidden_states)

        return hidden_states, deepstack_feature_lists
    
    def forward_dynamic(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        # chunk_len = int(640/16 * 640/16)
        # print("chunk_len",chunk_len)
        t, channel, seq_len, tpp = hidden_states.shape
        chunk_len = seq_len//2
        assert seq_len %2 == 0
        assert t==1 
        # hidden_states = hidden_states.permute(0,2,1,3).reshape(t,seq_len, channel*tpp)
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        torch.save(pos_embeds[0:chunk_len], "pos_embeds_p1.pth")
        torch.save(pos_embeds[chunk_len:], "pos_embeds_p2.pth")

        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        emb_cos = emb.cos()
        emb_sin = emb.sin()
        position_embeddings = (emb_cos, emb_sin)
        position_embeddings_p1 = (emb_cos[0:chunk_len], emb_sin[0:chunk_len])
        position_embeddings_p2 = (emb_cos[chunk_len:], emb_sin[chunk_len:])
        torch.save(position_embeddings_p1, "position_embeddings_p1.pth")
        torch.save(position_embeddings_p2, "position_embeddings_p2.pth")

        
        device = hidden_states.device
        dtype = hidden_states.dtype
        all_hidden_states = []
        all_deepstack_features_0 = []
        all_deepstack_features_1 = []
        all_deepstack_features_2 = []

        # grid_thw_one = torch.tensor([[1, 384//16, 384//16]], dtype=torch.int32).to(hidden_states.device)
        chunk_num = math.ceil(seq_len/chunk_len)
        for i in range(chunk_num):

            start = chunk_len * i 
            end = min(start + chunk_len, seq_len)
            
            ht = hidden_states[:,:, start:end]
            pos_emb = pos_embeds[start:end]
            # r_pos_emb = rotary_pos_emb[start:end]

            # r_pos_emb = r_pos_emb.reshape(end-start, -1)
            # emb = torch.cat((r_pos_emb, r_pos_emb), dim=-1)
            pos_emb_cos = emb_cos[start:end]
            pos_emb_sin = emb_sin[start:end]
            
            ht, deepstack_feature_lists = self.forward_chunk(ht,  
                                                            pos_emb, 
                                                            pos_emb_cos,  
                                                            pos_emb_sin, 
                                                            **kwargs)

            all_hidden_states.append(ht)
            all_deepstack_features_0.append(deepstack_feature_lists[0])
            all_deepstack_features_1.append(deepstack_feature_lists[1])
            all_deepstack_features_2.append(deepstack_feature_lists[2])

        all_hidden_states = torch.cat(all_hidden_states, 0)
        all_deepstack_features_0 = torch.cat(all_deepstack_features_0, 0)
        all_deepstack_features_1 = torch.cat(all_deepstack_features_1, 0)
        all_deepstack_features_2 = torch.cat(all_deepstack_features_2, 0)

        return all_hidden_states, [all_deepstack_features_0, all_deepstack_features_1, all_deepstack_features_2]

class Qwen3VLModelInfer(Qwen3VLModel):
    def __init__(self, config):
        super().__init__(config)
        config.vision_config._attn_implementation = "eager"
        self.visual = Qwen3VLVisionModelInfer._from_config(config.vision_config)
class Qwen3VLForConditionalGenerationInfer(Qwen3VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3VLModelInfer(config)


class Qwen3VLVisionModelExport(Qwen3VLVisionModel):
    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        
        # self.pos_embeds = torch.load("pos_embeds.pth", "cpu", weights_only=True)
        # self.position_embeddings = torch.load("position_embeddings.pth", "cpu", weights_only=True)
        # self.cu_seqlens = torch.load("cu_seqlens.pth","cpu", weights_only=True)

        self.pos_embeds_p1 = torch.load("pos_embeds_p1.pth", "cpu", weights_only=True)
        self.position_embeddings_p1 = torch.load("position_embeddings_p1.pth", "cpu", weights_only=True)
        self.pos_embeds_p2 = torch.load("pos_embeds_p2.pth", "cpu", weights_only=True)
        self.position_embeddings_p2 = torch.load("position_embeddings_p2.pth", "cpu", weights_only=True)

    def forward_image(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        device = hidden_states.device
        t, channel, seq_len, tpp = hidden_states.shape
        assert t==1 
        hidden_states = hidden_states.permute(0,2,1,3).reshape(t,seq_len, channel*tpp)

        hidden_states = self.patch_embed(hidden_states)        
        hidden_states = hidden_states + self.pos_embeds.to(device)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        
        position_embeddings = (self.position_embeddings[0].to(device), self.position_embeddings[1].to(device))
        cu_seqlens = self.cu_seqlens.to(device)

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
                    hidden_states
                )
                deepstack_feature_lists.append(deepstack_feature)

        hidden_states = self.merger(hidden_states)

        return hidden_states, deepstack_feature_lists
    
    def forward_chunk1(self, hidden_states):
        device = hidden_states.device
        t, channel, seq_len, tpp = hidden_states.shape
        assert t==1 
        hidden_states = hidden_states.permute(0,2,1,3).reshape(t,seq_len, channel*tpp)

        hidden_states = self.patch_embed(hidden_states)

        hidden_states = hidden_states + self.pos_embeds_p1.to(device)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        # rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        # emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        # position_embeddings = (emb.cos(), emb.sin())
        position_embeddings = (self.position_embeddings_p1[0].to(device), self.position_embeddings_p1[1].to(device))
        cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32).to(hidden_states.device)

    
        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
                    hidden_states
                )
                deepstack_feature_lists.append(deepstack_feature)

        hidden_states = self.merger(hidden_states)

        return hidden_states, deepstack_feature_lists
    
    def forward_chunk2(self, hidden_states):
        
        device = hidden_states.device
        t, channel, seq_len, tpp = hidden_states.shape
        assert t==1 
        hidden_states = hidden_states.permute(0,2,1,3).reshape(t,seq_len, channel*tpp)

        hidden_states = self.patch_embed(hidden_states)

        hidden_states = hidden_states + self.pos_embeds_p2.to(device)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        # rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        # emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        # position_embeddings = (emb.cos(), emb.sin())
        position_embeddings = (self.position_embeddings_p2[0].to(device), self.position_embeddings_p2[1].to(device))
        cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32).to(hidden_states.device)

    
        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings
            )
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
                    hidden_states
                )
                deepstack_feature_lists.append(deepstack_feature)

        hidden_states = self.merger(hidden_states)

        return hidden_states, deepstack_feature_lists
    
    def forward_dynamic(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        # chunk_len = int(640/16 * 640/16)
        # print("chunk_len",chunk_len)
        t, channel, seq_len, tpp = hidden_states.shape
        chunk_len = seq_len//2
        assert seq_len %2 == 0
        assert t==1 
        
        emb_cos, emb_sin = self.position_embeddings
        
        device = hidden_states.device
        dtype = hidden_states.dtype
        all_hidden_states = []
        all_deepstack_features_0 = []
        all_deepstack_features_1 = []
        all_deepstack_features_2 = []

        # grid_thw_one = torch.tensor([[1, 384//16, 384//16]], dtype=torch.int32).to(hidden_states.device)
        chunk_num = math.ceil(seq_len/chunk_len)
        for i in range(chunk_num):

            start = chunk_len * i 
            end = min(start + chunk_len, seq_len)
            
            ht = hidden_states[:,:, start:end]
          
            if i==0:  
                ht, deepstack_feature_lists = self.forward_chunk1(ht)
            else:
                ht, deepstack_feature_lists = self.forward_chunk2(ht)

            all_hidden_states.append(ht)
            all_deepstack_features_0.append(deepstack_feature_lists[0])
            all_deepstack_features_1.append(deepstack_feature_lists[1])
            all_deepstack_features_2.append(deepstack_feature_lists[2])

        all_hidden_states = torch.cat(all_hidden_states, 0)
        all_deepstack_features_0 = torch.cat(all_deepstack_features_0, 0)
        all_deepstack_features_1 = torch.cat(all_deepstack_features_1, 0)
        all_deepstack_features_2 = torch.cat(all_deepstack_features_2, 0)

        return all_hidden_states, [all_deepstack_features_0, all_deepstack_features_1, all_deepstack_features_2]
    
class Qwen3VLModelExport(Qwen3VLModel):
    def __init__(self, config):
        super().__init__(config)
        config.vision_config._attn_implementation = "eager"
        self.visual = Qwen3VLVisionModelExport._from_config(config.vision_config)

class Qwen3VLForConditionalGenerationExport(Qwen3VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3VLModelExport(config)


class Qwen3VLVisionModelONNX(Qwen3VLVisionModel):
    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)

    def init_onnx_session(self, onnx_path1, onnx_path2):
        print(f"init onnx model:{onnx_path1}")
        # self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

        self.session1 = ort.InferenceSession(onnx_path1, providers=["CPUExecutionProvider"])
        self.session2 = ort.InferenceSession(onnx_path2, providers=["CPUExecutionProvider"])


    def forward_image(self, hidden_states: torch.Tensor,  grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        device = hidden_states.device
        inputs = {"hidden_states": hidden_states.to(torch.float32).cpu().numpy()}
        outputs = self.session.run(None, inputs)
        outputs = [torch.from_numpy(out).to(device) for out in outputs]
        print("outputs[1:]",outputs[1:])
        return outputs[0], outputs[1:]

    def forward_sliced_image(self, hidden_states: torch.Tensor,  grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        device = hidden_states.device
        length = hidden_states.shape[2]
        len_onnx = 576
        chunk_num = math.ceil(length/len_onnx)
        print("chunk_num", chunk_num)
        outputs0 = []
        outputs1 = []
        outputs2 = []
        outputs3 = []
        for i in range(chunk_num):
            start = i*len_onnx
            end = min(start+len_onnx, length)

            x = np.zeros((1, 3, len_onnx, 512), dtype=np.float32)    
            x[:,:,0:end-start] = hidden_states.to(torch.float32).cpu().numpy()[:,:,start:end]

            inputs = {"hidden_states": x}
            outputs = self.session.run(None, inputs)
            outputs0.append(  torch.from_numpy(outputs[0][: (end-start)//4]).to(device) )
            outputs1.append(  torch.from_numpy(outputs[1][: (end-start)//4]).to(device) )
            outputs2.append(  torch.from_numpy(outputs[2][: (end-start)//4]).to(device) )
            outputs3.append(  torch.from_numpy(outputs[3][: (end-start)//4]).to(device) )

        outputs0 = torch.cat(outputs0, 0)
        outputs1 = torch.cat(outputs1, 0)
        outputs2 = torch.cat(outputs2, 0)
        outputs3 = torch.cat(outputs3, 0)
        return outputs0, [outputs1, outputs2, outputs3]
    
    def forward_video(self, hidden_states: torch.Tensor,  grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        device = hidden_states.device
        t = hidden_states.shape[0]        

        outputs0 = []
        outputs1 = []
        outputs2 = []
        outputs3 = []
        for i in range(t):
            ht = hidden_states[i:i+1]
            inputs = {"hidden_states": ht.to(torch.float32).cpu().numpy()}
            outputs = self.session.run(None, inputs)

            outputs0.append(  torch.from_numpy(outputs[0]).to(device) )
            outputs1.append(  torch.from_numpy(outputs[1]).to(device) )
            outputs2.append(  torch.from_numpy(outputs[2]).to(device) )
            outputs3.append(  torch.from_numpy(outputs[3]).to(device) )

        outputs0 = torch.cat(outputs0, 0)
        outputs1 = torch.cat(outputs1, 0)
        outputs2 = torch.cat(outputs2, 0)
        outputs3 = torch.cat(outputs3, 0)
        return outputs0, [outputs1, outputs2, outputs3]

    def forward_chunk1(self, hidden_states, grid_thw):
        
        device = hidden_states.device
        inputs = {"hidden_states": hidden_states.to(torch.float32).cpu().numpy()}
        outputs = self.session1.run(None, inputs)
        outputs = [torch.from_numpy(out).to(device) for out in outputs]
        print("outputs[1:]",outputs[1:])
        return outputs[0], outputs[1:]
    
    def forward_chunk2(self, hidden_states, grid_thw):
        
        device = hidden_states.device
        inputs = {"hidden_states": hidden_states.to(torch.float32).cpu().numpy()}
        outputs = self.session2.run(None, inputs)
        outputs = [torch.from_numpy(out).to(device) for out in outputs]
        print("outputs[1:]",outputs[1:])
        return outputs[0], outputs[1:]
    
    def forward_dynamic(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        # chunk_len = int(640/16 * 640/16)
        # print("chunk_len",chunk_len)
        t, channel, seq_len, tpp = hidden_states.shape
        chunk_len = seq_len//2
        assert seq_len %2 == 0
        assert t==1 
        
        all_hidden_states = []
        all_deepstack_features_0 = []
        all_deepstack_features_1 = []
        all_deepstack_features_2 = []

        # grid_thw_one = torch.tensor([[1, 384//16, 384//16]], dtype=torch.int32).to(hidden_states.device)
        chunk_num = math.ceil(seq_len/chunk_len)
        for i in range(chunk_num):

            start = chunk_len * i 
            end = min(start + chunk_len, seq_len)
            
            ht = hidden_states[:,:, start:end]
          
            if i==0:  
                ht, deepstack_feature_lists = self.forward_chunk1(ht, grid_thw)
            else:
                ht, deepstack_feature_lists = self.forward_chunk2(ht, grid_thw)

            all_hidden_states.append(ht)
            all_deepstack_features_0.append(deepstack_feature_lists[0])
            all_deepstack_features_1.append(deepstack_feature_lists[1])
            all_deepstack_features_2.append(deepstack_feature_lists[2])

        all_hidden_states = torch.cat(all_hidden_states, 0)
        all_deepstack_features_0 = torch.cat(all_deepstack_features_0, 0)
        all_deepstack_features_1 = torch.cat(all_deepstack_features_1, 0)
        all_deepstack_features_2 = torch.cat(all_deepstack_features_2, 0)

        return all_hidden_states, [all_deepstack_features_0, all_deepstack_features_1, all_deepstack_features_2]
class Qwen3VLModelONNX(Qwen3VLModel):
    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen3VLVisionModelONNX._from_config(config.vision_config)
        
class Qwen3VLForConditionalGenerationONNX(Qwen3VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3VLModelONNX(config)