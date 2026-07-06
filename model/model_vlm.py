import os
import torch
import warnings
from .model_minimind import *
from typing import Optional, Tuple, List, Union
from torch import nn
from transformers import AutoImageProcessor, AutoModel
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

warnings.filterwarnings('ignore')


class VLMConfig(MiniMindConfig):
    model_type = "minimind-v"

    def __init__(self, image_special_token='<|image_pad|>', image_ids=[12], **kwargs):
        self.image_special_token = image_special_token
        self.image_ids = image_ids
        self.image_hidden_size = kwargs.get("image_hidden_size", 768)
        self.image_token_len = kwargs.get("image_token_len", 64)
        super().__init__(**kwargs)

class MMVisionProjector(nn.Module):
    def __init__(self, in_dim, out_dim, source_tokens=256, target_tokens=64):
        super().__init__()
        if source_tokens % target_tokens != 0:
            raise ValueError(f"source_tokens={source_tokens} 必须能被 target_tokens={target_tokens} 整除")
        self.target_tokens = target_tokens
        self.merge = source_tokens // target_tokens
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * self.merge, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
    def forward(self, x):
        b, n, d = x.shape
        x = x.reshape(b, self.target_tokens, d * self.merge)
        return self.mlp(x)

# 继承自语言模型
class MiniMindVLM(MiniMindForCausalLM):
    config_class = VLMConfig

    def __init__(self, config: VLMConfig = None, vision_model_path="./model/siglip2-base-p16-ve"):
        print("MiniMindVLM.__init__ started")
        self.config = config or VLMConfig()
        print("Calling super().__init__...")
        super().__init__(self.config)
        print("super().__init__ completed, loading vision model...")
        import sys
        sys.stdout.flush()
        self.vision_encoder, self.processor = self.__class__.get_vision_model(vision_model_path)
        print("Vision model loaded, creating vision_proj...")
        sys.stdout.flush()
        source_tokens = (
            (self.vision_encoder.config.image_size // self.vision_encoder.config.patch_size) ** 2
            if self.vision_encoder is not None else 256
        )
        self.vision_proj = MMVisionProjector(
            self.config.image_hidden_size,
            self.config.hidden_size,
            source_tokens=source_tokens,
            target_tokens=self.config.image_token_len,
        )
        print("MiniMindVLM.__init__ completed")
        sys.stdout.flush()

    @staticmethod
    def get_vision_model(model_path: str):
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()
        if not os.path.exists(model_path):
            return None, None
        print(f"Loading vision model from {model_path}...")
        import sys
        sys.stdout.flush()
        model = AutoModel.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=False
        )
        print("Vision model loaded, loading processor...")
        sys.stdout.flush()
        processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
        print("Processor loaded, freezing parameters...")
        sys.stdout.flush()
        # 冻结 vision_encoder 的所有参数
        for param in model.parameters():
            param.requires_grad = False
        print("Parameters frozen, calling model.eval()...")
        sys.stdout.flush()
        model_eval = model.eval()
        print("model.eval() completed, returning...")
        sys.stdout.flush()
        return model_eval, processor

    @staticmethod
    def image2tensor(image, processor):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        inputs = processor(images=image, return_tensors="pt")
        return inputs

    @staticmethod
    def get_image_embeddings(image_inputs, vision_model):
        if hasattr(image_inputs, 'keys'):
            device = vision_model.device
            cleaned_inputs = {}
            for k, v in image_inputs.items():
                if k == 'spatial_shapes' and isinstance(v, list):
                    # spatial_shapes 需要转为 tensor: [[h, w]] -> tensor([[h, w]])
                    v = torch.tensor(v, dtype=torch.long, device=device)
                elif isinstance(v, list):
                    # 其他 list 转为 tensor
                    v = torch.tensor(v, device=device)
                elif isinstance(v, torch.Tensor):
                    v = v.to(device)
                    if v.ndim > 2 and v.shape[1] == 1:
                        v = v.squeeze(1)
                cleaned_inputs[k] = v
            image_inputs = cleaned_inputs
        with torch.no_grad():
            outputs = vision_model(**image_inputs)
        return outputs.last_hidden_state

    @torch.compiler.disable
    def count_vision_proj(self, tokens, h, vision_tensors=None, seqlen=512):
        if vision_tensors is None or not self.config.image_ids:
            return h
        marker, vf = self.config.image_ids[0], vision_tensors
        if vf.dim() == 3:
            vf = vf.unsqueeze(1)
        out = []
        for b in range(h.size(0)):
            hb, seq, k, i = h[b], tokens[b].tolist(), 0, 0
            while i < len(seq):
                if seq[i] == marker:
                    start = i
                    while i < len(seq) and seq[i] == marker:
                        i += 1
                    if k < vf.size(1):
                        hb = torch.cat((hb[:start], vf[b][k][:i - start], hb[i:]), dim=0)[:seqlen]
                        k += 1
                else:
                    i += 1
            out.append(hb)
        return torch.stack(out)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                labels: Optional[torch.Tensor] = None,
                loss_weights: Optional[torch.Tensor] = None,
                pixel_values: Optional[torch.FloatTensor] = None,
                **args):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.model.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.model.dropout(self.model.embed_tokens(input_ids))

        if pixel_values is not None and start_pos == 0:
            if hasattr(pixel_values, 'keys'):
                # Handle batched dict inputs: [batch, num_imgs, ...] -> [batch*num_imgs, ...]
                first_key = list(pixel_values.keys())[0]
                need_flatten = (
                    'spatial_shapes' in pixel_values
                    and isinstance(pixel_values['spatial_shapes'], torch.Tensor)
                    and pixel_values['spatial_shapes'].ndim == 3
                )
                if need_flatten or pixel_values[first_key].ndim > 4:
                    if need_flatten:
                        batch_size, num_imgs = pixel_values['spatial_shapes'].shape[:2]
                    else:
                        batch_size = pixel_values[first_key].shape[0]
                        num_imgs = pixel_values[first_key].shape[1]
                    flat_pixel_values = {
                        k: (v.reshape(-1, *v.shape[2:]) if isinstance(v, torch.Tensor) and v.ndim >= 3 else v)
                        for k, v in pixel_values.items()
                    }
                    first_tensor = next(v for v in flat_pixel_values.values() if isinstance(v, torch.Tensor))
                    total_imgs = first_tensor.shape[0]
                    spatial_shapes = flat_pixel_values.get('spatial_shapes', None)
                    if isinstance(spatial_shapes, torch.Tensor) and spatial_shapes.ndim == 2:
                        valid_mask = (spatial_shapes[:, 0] > 0) & (spatial_shapes[:, 1] > 0)
                    else:
                        valid_mask = torch.ones(total_imgs, dtype=torch.bool, device=first_tensor.device)

                    proj_dtype = next(self.vision_proj.parameters()).dtype
                    if bool(valid_mask.any().item()):
                        filtered_pixel_values = {
                            k: (v[valid_mask] if isinstance(v, torch.Tensor) and v.ndim > 0 and v.shape[0] == total_imgs else v)
                            for k, v in flat_pixel_values.items()
                        }
                        img_emb = MiniMindVLM.get_image_embeddings(filtered_pixel_values, self.vision_encoder)
                        img_emb = img_emb.to(dtype=proj_dtype)
                        valid_vision_tensors = self.vision_proj(img_emb)
                        if bool(valid_mask.all().item()):
                            vision_tensors = valid_vision_tensors
                        else:
                            vision_tensors = valid_vision_tensors.new_zeros((total_imgs, *valid_vision_tensors.shape[1:]))
                            vision_tensors[valid_mask] = valid_vision_tensors
                    else:
                        vision_tensors = hidden_states.new_zeros(
                            (total_imgs, self.config.image_token_len, self.config.hidden_size),
                            dtype=proj_dtype,
                        )
                    # Reshape back: [batch*num_imgs, seq, hidden] -> [batch, num_imgs, seq, hidden]
                    vision_tensors = vision_tensors.reshape(batch_size, num_imgs, *vision_tensors.shape[1:])
                else:
                    img_emb = MiniMindVLM.get_image_embeddings(pixel_values, self.vision_encoder)
                    proj_dtype = next(self.vision_proj.parameters()).dtype
                    img_emb = img_emb.to(dtype=proj_dtype)
                    vision_tensors = self.vision_proj(img_emb)
            else:
                if len(pixel_values.shape) == 6:
                    pixel_values = pixel_values.squeeze(2)
                bs, num, c, im_h, im_w = pixel_values.shape
                stack_dim = 1 if bs > 1 else 0
                proj_dtype = next(self.vision_proj.parameters()).dtype
                vision_tensors = torch.stack([
                    self.vision_proj(MiniMindVLM.get_image_embeddings(pixel_values[:, i, :, :, :], self.vision_encoder).to(dtype=proj_dtype))
                    for i in range(num)
                ], dim=stack_dim)
            hidden_states = self.count_vision_proj(tokens=input_ids, h=hidden_states, vision_tensors=vision_tensors, seqlen=input_ids.shape[1])

        position_embeddings = (
            self.model.freqs_cos[start_pos:start_pos + seq_length],
            self.model.freqs_sin[start_pos:start_pos + seq_length]
        )

        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.model.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        hidden_states = self.model.norm(hidden_states)

        aux_loss = sum([l.mlp.aux_loss for l in self.model.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            if (shift_labels != -100).any():
                token_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction='none'
                )
                valid = shift_labels.view(-1).ne(-100)
                if loss_weights is not None:
                    weights = loss_weights[..., 1:].contiguous().view(-1).to(token_loss.dtype)
                    weights = weights * valid.to(weights.dtype)
                    loss = (token_loss * weights).sum() / weights.sum().clamp_min(1.0)
                else:
                    loss = token_loss[valid].mean()
            else:
                loss = shift_logits.new_zeros(())

        output = MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=presents, hidden_states=hidden_states)
        return output
