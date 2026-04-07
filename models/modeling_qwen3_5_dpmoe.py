"""Qwen 3.5 model with dynamic pruning virtual operations for ToMoE training.

This file modifies the HuggingFace Qwen 3.5 text model to inject virtual pruning
operations into MLP layers (all 32) and attention layers (8 full-attention layers only).
Gated Delta Network (linear attention) layers remain completely untouched.

Requires: transformers >= 4.57.0, flash-linear-attention (FLA)
"""

import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

# Import unmodified Qwen 3.5 components from transformers
try:
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5GatedDeltaNet,
        Qwen3_5PreTrainedModel as _Qwen3_5PreTrainedModel,
    )
    from transformers import Qwen3_5Config
except ImportError:
    raise ImportError(
        "Qwen 3.5 support requires transformers >= 4.57.0. "
        "Install with: pip install 'transformers>=4.57.0'"
    )

# Import ToMoE virtual operations
from tomoe.hypernetwork import (
    virtual_mlp_operation,
    virtual_basic_operation,
    virtual_block_attn_operation,
    virtual_block_basic_operation,
    virtual_dynamic_operation,
    virtual_vo_operation,
)

logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# Utility layers (copied from HF for standalone use)
# ---------------------------------------------------------------------------

class Qwen3_5RMSNorm(nn.Module):
    """Qwen 3.5 uses zero-initialized weight with (1 + weight) scaling."""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, hidden_states):
        output = self._norm(hidden_states.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(hidden_states)


class Qwen3_5TextRotaryEmbedding(nn.Module):
    """M-RoPE with partial rotary factor for Qwen 3.5."""

    def __init__(self, config: Qwen3_5Config, device=None):
        super().__init__()
        self.config = config
        # Qwen 3.5: head_dim=256, partial_rotary_factor=0.25 -> rotary_dim=64
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        dim = int(head_dim * partial_rotary_factor)
        self.rotary_dim = dim

        rope_theta = getattr(config, "rope_theta", 10000.0)
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # position_ids: [batch_size, seq_len] or [4, batch_size, seq_len] for M-RoPE
        device = x.device

        if position_ids.dim() == 2:
            # Text-only: expand to M-RoPE format [4, batch, seq]
            position_ids = position_ids.unsqueeze(0).expand(4, -1, -1)

        # M-RoPE sections from config (e.g. [11, 11, 10] for mrope_sections)
        mrope_sections = getattr(self.config, "mrope_section", None) or getattr(self.config, "mrope_sections", [self.rotary_dim // 2])

        inv_freq = self.inv_freq.to(device)
        # Compute frequencies for each M-RoPE section
        cos_parts = []
        sin_parts = []
        freq_offset = 0
        for i, section_size in enumerate(mrope_sections):
            section_inv_freq = inv_freq[freq_offset:freq_offset + section_size]
            pos = position_ids[i].float()  # [batch, seq]
            freqs = torch.einsum("bs,d->bsd", pos, section_inv_freq)  # [batch, seq, section_size]
            cos_parts.append(freqs.cos())
            sin_parts.append(freqs.sin())
            freq_offset += section_size

        # Concatenate sections and duplicate for rotate_half pattern
        cos = torch.cat(cos_parts, dim=-1)  # [batch, seq, rotary_dim/2]
        sin = torch.cat(sin_parts, dim=-1)
        cos = torch.cat([cos, cos], dim=-1)  # [batch, seq, rotary_dim]
        sin = torch.cat([sin, sin], dim=-1)

        return cos.to(x.dtype), sin.to(x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Apply partial rotary position embedding to query and key tensors.

    For Qwen 3.5, cos/sin have shape [batch, seq, rotary_dim] where rotary_dim < head_dim.
    Only the first rotary_dim dimensions of q/k get rotated.
    """
    cos = cos.unsqueeze(unsqueeze_dim)  # [batch, 1, seq, rotary_dim]
    sin = sin.unsqueeze(unsqueeze_dim)

    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# ---------------------------------------------------------------------------
# Modified MLP with virtual pruning operations (ALL 32 layers)
# ---------------------------------------------------------------------------

class Qwen3_5MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

        # === ToMoE virtual pruning operations ===
        ex_dict = {
            'dim_1': config.intermediate_size,
            'dim_2': config.hidden_size,
            'num_weight': 3,
        }
        self.use_gate = True
        self.virtual_gate = virtual_mlp_operation(dim=config.intermediate_size, ex_dict=ex_dict)
        self.dynamic_router = virtual_dynamic_operation(middle_dim=config.intermediate_size)

    def forward(self, x):
        if self.use_gate:
            vectors = self.dynamic_router(x)
            self.virtual_gate.set_vector_value(vectors)
            down_proj = self.down_proj(
                self.virtual_gate(self.act_fn(self.gate_proj(x)))
                * self.virtual_gate(self.up_proj(x))
            )
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


# ---------------------------------------------------------------------------
# Modified Attention with virtual pruning operations (8 full-attention layers)
# ---------------------------------------------------------------------------

class Qwen3_5Attention(nn.Module):
    """Qwen 3.5 full attention with ToMoE virtual pruning.

    Qwen 3.5 specifics:
    - q_proj outputs num_heads * head_dim * 2 (query + sigmoid output gate)
    - q_norm / k_norm (per-head RMSNorm) before RoPE
    - Partial RoPE (partial_rotary_factor=0.25, only 64 of 256 dims)
    - Output gate: attn_output * sigmoid(gate) before o_proj
    """

    def __init__(self, config: Qwen3_5Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True

        # Qwen 3.5: q_proj is doubled for output gate
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim * 2, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        # Qwen 3.5: per-head QK norms
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # === ToMoE virtual pruning operations ===
        ex_dict = {
            'dim_1': self.hidden_size,
            'dim_2': self.num_key_value_heads * self.head_dim,
            'head_dim': self.head_dim,
            'num_groups': self.num_key_value_groups,
            'num_kv_heads': self.num_key_value_heads,
            'num_heads': self.num_heads,
            'num_weight': 4,
        }

        self.use_att_gate = True
        self.apply_qk_gate = False

        self.virtual_attn_gate_1 = virtual_block_attn_operation(
            dim=config.hidden_size, ex_dict=ex_dict
        )
        self.virtual_attn_gate_2 = virtual_basic_operation(dim=config.hidden_size)

        ex_dict_vo = {
            'dim_1': self.hidden_size,
            'dim_2': self.num_key_value_heads * self.head_dim,
            'head_dim': self.head_dim,
            'num_groups': self.num_key_value_groups,
            'num_kv_heads': self.num_key_value_heads,
            'num_heads': self.num_heads,
        }

        self.virtual_gate = virtual_vo_operation(dim=self.head_dim, ex_dict=ex_dict_vo)
        self.dynamic_router = virtual_dynamic_operation(
            middle_dim=self.head_dim + self.head_dim
        )
        self.qk_sample_rate = 0.6
        self.grad_w = 2.0
        self.pv_detach_flag = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # Route through hypernetwork for virtual gate
        if self.use_att_gate:
            vectors = self.dynamic_router(hidden_states)
            self.virtual_gate.set_vector_value(vectors)

        # Q projection (doubled: query + gate)
        qkv = self.q_proj(hidden_states)
        qkv = qkv.view(bsz, q_len, self.num_heads, self.head_dim * 2)
        query_states, gate = qkv.split(self.head_dim, dim=-1)
        gate = gate.reshape(bsz, q_len, -1)  # [bsz, q_len, num_heads * head_dim]

        # K, V projections
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # QK norms (per-head)
        query_states = self.q_norm(query_states)  # [bsz, q_len, num_heads, head_dim]
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        key_states = self.k_norm(key_states)

        # Transpose to [bsz, heads, q_len, head_dim]
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply virtual pruning gates to Q, K, V
        if self.use_att_gate:
            key_states = self.virtual_gate(key_states, pv_detach=False, mode='inner_head')
            query_states = self.virtual_gate(query_states, pv_detach=False, mode='inner_head')
            value_states = self.virtual_gate(value_states, pv_detach=False, mode='inner_head_v')

        # Partial RoPE
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # KV cache
        if past_key_value is not None:
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # GQA expansion
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Standard scaled dot-product attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        # Qwen 3.5 output gate
        attn_output = attn_output * torch.sigmoid(gate)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# ---------------------------------------------------------------------------
# Decoder Layer (hybrid: routes between GDN and full attention)
# ---------------------------------------------------------------------------

class Qwen3_5DecoderLayer(nn.Module):
    """Qwen 3.5 decoder layer with ToMoE virtual pruning.

    Based on config.layer_types[layer_idx]:
    - "linear_attention": uses Qwen3_5GatedDeltaNet (UNTOUCHED, no virtual ops)
    - "full_attention": uses Qwen3_5Attention (WITH virtual ops)
    - All layers: use Qwen3_5MLP (WITH virtual ops)
    """

    def __init__(self, config: Qwen3_5Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # Determine layer type
        layer_types = getattr(config, "layer_types", None)
        if layer_types is not None:
            self.layer_type = layer_types[layer_idx]
        else:
            # Fallback: every 4th layer is full attention
            full_attention_interval = getattr(config, "full_attention_interval", 4)
            if (layer_idx + 1) % full_attention_interval == 0:
                self.layer_type = "full_attention"
            else:
                self.layer_type = "linear_attention"

        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)
        else:
            self.self_attn = Qwen3_5Attention(config=config, layer_idx=layer_idx)

        self.mlp = Qwen3_5MLP(config)
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # ToMoE training regularization
        self.resid_dropout = nn.Dropout(0.15)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, ...]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "linear_attention":
            # GDN layer: pass through untouched
            # GatedDeltaNet uses cache_params (not past_key_value)
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_value,
                attention_mask=attention_mask,
            )
            # GDN may return a tuple or just hidden_states
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
            self_attn_weights = None
            present_key_value = None
        else:
            # Full attention layer with virtual pruning
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.resid_dropout(hidden_states)
        hidden_states = residual + hidden_states

        # MLP (all layers have virtual pruning on MLP)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.resid_dropout(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


# ---------------------------------------------------------------------------
# Pre-trained model base
# ---------------------------------------------------------------------------

class Qwen3_5PreTrainedModel(PreTrainedModel):
    config_class = Qwen3_5Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3_5DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        std = self.config.initializer_range if hasattr(self.config, "initializer_range") else 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


# ---------------------------------------------------------------------------
# Text Model (backbone)
# ---------------------------------------------------------------------------

class Qwen3_5TextModel(Qwen3_5PreTrainedModel):
    def __init__(self, config: Qwen3_5Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3_5DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config)
        self.gradient_checkpointing = False

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Handle position_ids
        if cache_position is None:
            past_seen_tokens = 0
            if past_key_values is not None and hasattr(past_key_values, "get_seq_length"):
                past_seen_tokens = past_key_values.get_seq_length()
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Compute rotary embeddings for full-attention layers
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        # Build causal mask
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position
        )

        hidden_states = inputs_embeds

        # Decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(self, attention_mask, input_tensor, cache_position):
        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        target_length = cache_position[-1].item() + 1 if cache_position is not None else sequence_length

        causal_mask = torch.full(
            (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
        )
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask.to(torch.float32), diagonal=1).to(dtype)

        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)

        if attention_mask is not None and attention_mask.dim() == 2:
            causal_mask = causal_mask.clone()
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
            causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)

        return causal_mask


# ---------------------------------------------------------------------------
# Causal LM wrapper
# ---------------------------------------------------------------------------

class Qwen3_5ForCausalLM(Qwen3_5PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        # Qwen 3.5 uses a multimodal config; extract text_config
        text_config = getattr(config, "text_config", config)
        self.model = Qwen3_5TextModel(text_config)
        self.vocab_size = text_config.vocab_size
        self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)
        self.text_config = text_config

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.text_config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
