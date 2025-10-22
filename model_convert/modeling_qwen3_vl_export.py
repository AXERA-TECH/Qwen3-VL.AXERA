import torch
import torch.nn.functional as F
import numpy as np
import onnxruntime as ort
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLModel, Qwen3VLVisionModel


class Qwen3VLVisionModelInfer(Qwen3VLVisionModel):

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
class Qwen3VLModelInfer(Qwen3VLModel):
    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen3VLVisionModelInfer._from_config(config.vision_config)
class Qwen3VLForConditionalGenerationInfer(Qwen3VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3VLModelInfer(config)


class Qwen3VLVisionModelExport(Qwen3VLVisionModel):
    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        
        self.pos_embeds = torch.load("pos_embeds.pth", "cpu", weights_only=True)
        self.position_embeddings = torch.load("position_embeddings.pth", "cpu", weights_only=True)
        self.cu_seqlens = torch.load("cu_seqlens.pth","cpu", weights_only=True)

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

    def init_onnx_session(self, onnx_path):
        print(f"init onnx model:{onnx_path}")
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

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
        return outputs[0], outputs[1:]
    
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
            outputs1.append(  torch.from_numpy(outputs[0]).to(device) )
            outputs2.append(  torch.from_numpy(outputs[0]).to(device) )
            outputs3.append(  torch.from_numpy(outputs[0]).to(device) )

        outputs0 = torch.cat(outputs0, 0)
        outputs1 = torch.cat(outputs1, 0)
        outputs2 = torch.cat(outputs2, 0)
        outputs3 = torch.cat(outputs3, 0)
        return outputs0, [outputs1, outputs2, outputs3]

class Qwen3VLModelONNX(Qwen3VLModel):
    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen3VLVisionModelONNX._from_config(config.vision_config)
        
class Qwen3VLForConditionalGenerationONNX(Qwen3VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3VLModelONNX(config)