"""Qwen 3.5 MoE final model for inference after ToMoE pruning.

This replaces virtual pruning operations with actual expert routing via
single_experts_module. MLP layers use token-level MoE routing on all 32 layers.
Attention layers use binary head-dimension masking on 8 full-attention layers.
GDN (linear attention) layers are completely untouched.
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
    MoeModelOutputWithPast,
    MoeCausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

try:
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5GatedDeltaNet,
    )
    from transformers import Qwen3_5Config
except ImportError:
    raise ImportError(
        "Qwen 3.5 support requires transformers >= 4.57.0. "
        "Install with: pip install 'transformers>=4.57.0'"
    )

logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# Gumbel sampling utilities (from LLaMA MoE final)
# ---------------------------------------------------------------------------

def gumbel_sigmoid_function(logits, tau=1, hard=False, sample=True, offset=0):
    if sample:
        device = logits.get_device()
        gumbels = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format, device=device).exponential_().log()
        gumbels = (logits + gumbels + offset) / tau
    else:
        gumbels = (logits + offset) / tau
    y_soft = gumbels.sigmoid()
    if hard:
        y_hard = torch.round(y_soft)
        ret = (y_hard - y_soft).detach() + y_soft
    else:
        ret = y_soft
    return ret


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, T, sample=True):
    gumbel_sample = sample_gumbel(logits.size())
    if logits.get_device() == -1:
        gumbel_sample = gumbel_sample.cpu()
    else:
        gumbel_sample = gumbel_sample.to(logits.get_device())
    if sample:
        y = logits + gumbel_sample
    else:
        y = logits
    return F.softmax(y / T, dim=-1)


def gumbel_softmax(logits, T, hard_sample=False, return_soft=False, sample=True):
    shape = logits.size()
    logits = logits.view(-1, shape[-1])
    y = gumbel_softmax_sample(logits, T, sample)
    if not hard_sample:
        return y.view(shape)
    else:
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        y = y.view(shape)
        y_hard = (y_hard - y).detach() + y
        if return_soft:
            return y_hard.view(shape), y.view(shape)
        else:
            return y_hard.view(shape)


def hard_sample(out):
    binary_out = torch.round(out)
    binary_out = (binary_out - out).detach() + out
    return binary_out


def hard_topk(out, k):
    topk = torch.topk(out, k, dim=-1)
    indices = topk.indices
    mask = torch.zeros_like(out)
    mask.scatter_(-1, indices, 1.0)
    mask = (mask - out).detach() + out
    return mask


# ---------------------------------------------------------------------------
# Expert routing module (shared between MLP and attention)
# ---------------------------------------------------------------------------

class single_experts_module(nn.Module):
    def __init__(self, mlp_dim, model_dim, experts=8, head_dim=128, attn_flag=False, num_kv_heads=0.5):
        super().__init__()
        self.experts = experts
        self.T = 0.4
        self.base = 3.0
        self.emb_dim = 128
        self.mlp_dim = mlp_dim
        self.model_dim = model_dim
        self.actual_moe = True

        if attn_flag:
            self.head_dim = head_dim
            self.linear_router = nn.Linear(self.model_dim, self.emb_dim, bias=False)
            self.linear_decoder = nn.Linear(self.emb_dim, head_dim, bias=False)
            self.ln = nn.LayerNorm([self.emb_dim])
            self.register_buffer('experts_for_eval', torch.zeros(1, self.head_dim).to(torch.uint8))
            self.register_buffer('qk_index', torch.zeros(self.head_dim).to(torch.int64))
            self.register_buffer('rnn_state', torch.zeros(self.emb_dim))
            self.register_buffer("top_k", torch.tensor(0.0))
            self.checked = False
        else:
            self.linear_router = nn.Linear(self.model_dim, self.experts, bias=False)
            self.register_buffer('experts_for_eval', torch.zeros(self.experts, mlp_dim).to(torch.uint8))

        self.attn_flag = attn_flag
        self.num_kv_heads = num_kv_heads
        self.experts_list = None

    def check_head_dim(self, head_dim):
        if self.checked:
            return self.actual_moe
        self.checked = True
        if self.experts_for_eval[-1, :].sum() < (head_dim / 8) * 2:
            self.actual_moe = True
        else:
            self.actual_moe = False
        return self.actual_moe

    def forward(self, x, return_binary=False):
        if self.attn_flag:
            batch_size, sequence_length, hidden_dim = x.shape
            out = self.linear_router(x)
            full_emb = self.rnn_state + out
            out_before_binary = self.linear_decoder(F.gelu(self.ln(full_emb)))

            binary_approx = gumbel_sigmoid_function(
                logits=out_before_binary, tau=self.T, offset=self.base, sample=True, hard=False
            )

            if self.top_k == 0:
                binary = hard_sample(binary_approx).view(batch_size, sequence_length, -1, self.head_dim)
                self.dynamic_width = binary.sum(-1).max()
            else:
                binary = hard_sample(binary_approx)
                width_max = int(self.top_k)
                scores = binary_approx.view(batch_size, sequence_length, self.head_dim)
                topk_idx = scores.topk(width_max, dim=-1).indices
                topk_mask = torch.zeros_like(binary_approx)
                topk_mask.scatter_(-1, topk_idx, 1.0)
                binary = torch.where(
                    (binary.sum(dim=-1, keepdim=True) > width_max), topk_mask, binary
                )
                zero_mask = (binary.sum(dim=-1, keepdim=True) == 0)
                if zero_mask.any():
                    max_idx = binary_approx.view(batch_size, sequence_length, self.head_dim).argmax(dim=-1, keepdim=True)
                    fallback = torch.zeros_like(binary)
                    fallback.scatter_(-1, max_idx, 1.0)
                    binary = torch.where(zero_mask, fallback, binary)
                binary = binary.view(batch_size, sequence_length, -1, self.head_dim)

            if return_binary:
                return binary, binary_approx
            else:
                raise NotImplementedError("return_binary must be True")
        else:
            if self.experts_list is None:
                self.experts_list = [
                    torch.nonzero(self.experts_for_eval[i, :])
                    for i in range(self.experts_for_eval.size(0))
                ]
            batch_size, sequence_length, hidden_dim = x.shape
            out = self.linear_router(x.view(-1, hidden_dim))
            hard_max, router_logits = gumbel_softmax(out, T=self.T, hard_sample=True, return_soft=True, sample=True)
            _, indices = hard_max.max(dim=-1)
            return indices, router_logits


# ---------------------------------------------------------------------------
# Utility layers
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

    def __init__(self, config, device=None):
        super().__init__()
        self.config = config
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        dim = int(head_dim * partial_rotary_factor)
        self.rotary_dim = dim
        rope_theta = getattr(config, "rope_theta", 10000.0)
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        device = x.device
        if position_ids.dim() == 2:
            position_ids = position_ids.unsqueeze(0).expand(4, -1, -1)
        mrope_sections = getattr(self.config, "mrope_section", None) or getattr(self.config, "mrope_sections", [self.rotary_dim // 2])
        inv_freq = self.inv_freq.to(device)
        cos_parts, sin_parts = [], []
        freq_offset = 0
        for i, section_size in enumerate(mrope_sections):
            section_inv_freq = inv_freq[freq_offset:freq_offset + section_size]
            pos = position_ids[i].float()
            freqs = torch.einsum("bs,d->bsd", pos, section_inv_freq)
            cos_parts.append(freqs.cos())
            sin_parts.append(freqs.sin())
            freq_offset += section_size
        cos = torch.cat(cos_parts, dim=-1)
        sin = torch.cat(sin_parts, dim=-1)
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)
        return cos.to(x.dtype), sin.to(x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, qk_index=None, unsqueeze_dim=1):
    """Apply partial RoPE with optional index selection for pruned heads."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    rotary_dim = cos.shape[-1]

    if qk_index is not None:
        # Pruned heads: select relevant RoPE dimensions
        cos = cos[..., qk_index]
        sin = sin[..., qk_index]
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
    else:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q_embed = torch.cat([(q_rot * cos) + (rotate_half(q_rot) * sin), q_pass], dim=-1)
        k_embed = torch.cat([(k_rot * cos) + (rotate_half(k_rot) * sin), k_pass], dim=-1)

    return q_embed, k_embed


def repeat_kv(hidden_states, n_rep):
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# ---------------------------------------------------------------------------
# MoE MLP (all 32 layers)
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

        self.experts_module = single_experts_module(config.intermediate_size, config.hidden_size)
        self.actual_moe = self.experts_module.actual_moe

    def moe_forward(self, x, indices):
        indices = indices.to(x.device).squeeze()
        gate_weight = self.gate_proj.weight[indices, :]
        up_weight = self.up_proj.weight[indices, :]
        down_weight = self.down_proj.weight[:, indices]
        gate_proj = F.linear(x, gate_weight)
        up_proj = F.linear(x, up_weight)
        before_down_proj = self.act_fn(gate_proj) * up_proj
        down_proj = F.linear(before_down_proj, down_weight)
        return down_proj

    def forward(self, x):
        binary, router_logits = self.experts_module(x)
        if self.actual_moe:
            if self.experts_module.experts_list is None:
                self.experts_module.experts_list = [
                    torch.nonzero(self.experts_module.experts_for_eval[i, :])
                    for i in range(self.experts_module.experts_for_eval.size(0))
                ]
            batch_size, sequence_length, hidden_dim = x.shape
            x_flat = x.view(-1, hidden_dim)
            binary_flat = binary.view(batch_size * sequence_length, -1)
            cnts = binary_flat.new_zeros((binary_flat.shape[0], self.experts_module.experts_for_eval.size(0)))
            cnts.scatter_(1, binary_flat, 1)
            tokens_per_expert = cnts.sum(dim=0)
            idxs = binary_flat.view(-1).argsort()
            sorted_tokens = x_flat[idxs]
            tokens_per_expert_np = tokens_per_expert.cpu().numpy()

            outputs = []
            start_idx = 0
            for i, num_tokens in enumerate(tokens_per_expert_np):
                end_idx = start_idx + num_tokens
                if num_tokens == 0:
                    continue
                expert = self.experts_module.experts_list[i]
                tokens_for_this_expert = sorted_tokens[start_idx:end_idx, :]
                expert_out = self.moe_forward(tokens_for_this_expert, expert)
                outputs.append(expert_out)
                start_idx = end_idx
            outs = torch.cat(outputs, dim=0)
            new_x = torch.empty_like(outs)
            new_x[idxs] = outs
            down_proj = new_x.view(batch_size, sequence_length, hidden_dim)
            return down_proj, router_logits
        else:
            down_proj = self.down_proj(
                (self.act_fn(self.gate_proj(x)) * binary) * (self.up_proj(x) * binary)
            )
            return down_proj, router_logits


# ---------------------------------------------------------------------------
# MoE Attention (8 full-attention layers only)
# ---------------------------------------------------------------------------

class Qwen3_5Attention(nn.Module):
    """Qwen 3.5 attention with MoE binary masking on head dimensions.

    Preserves Qwen 3.5 specifics: doubled q_proj (query + gate), q_norm/k_norm,
    partial RoPE, output gate.
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

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim * 2, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Expert module for binary head-dimension masking
        self.experts_module = single_experts_module(
            self.head_dim, self.hidden_size, head_dim=self.head_dim, attn_flag=True
        )

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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache], torch.Tensor]:
        bsz, q_len, _ = hidden_states.size()
        self.actual_moe = self.experts_module.check_head_dim(self.head_dim)

        # Q projection (doubled: query + gate)
        qkv = self.q_proj(hidden_states)
        qkv = qkv.view(bsz, q_len, self.num_heads, self.head_dim * 2)
        query_states, gate = qkv.split(self.head_dim, dim=-1)
        gate = gate.reshape(bsz, q_len, -1)

        # K projection
        key_states = self.k_proj(hidden_states)

        # Expert binary mask on V
        binary, router_logits = self.experts_module(hidden_states, return_binary=True)
        binary_vector = binary.view(bsz, q_len, -1, self.head_dim)

        value_states = self.v_proj(hidden_states)
        value_states = binary_vector.expand(-1, -1, self.num_key_value_heads, -1).reshape(bsz, q_len, -1) * value_states

        # QK norms
        query_states = self.q_norm(query_states)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        key_states = self.k_norm(key_states)

        # Transpose to [bsz, heads, q_len, head_dim]
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Partial RoPE with qk_index for pruned dimensions
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin,
            qk_index=self.experts_module.qk_index
        )

        # KV cache
        if past_key_value is not None:
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # GQA expansion
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = (attn_weights + causal_mask).contiguous()

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        # Apply binary mask to output
        attn_output = binary_vector.expand(-1, -1, self.num_key_value_heads, -1).reshape(bsz, q_len, -1) * attn_output

        # Qwen 3.5 output gate
        attn_output = attn_output * torch.sigmoid(gate)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value, router_logits


# ---------------------------------------------------------------------------
# Decoder Layer (hybrid: GDN untouched, full attention with MoE)
# ---------------------------------------------------------------------------

class Qwen3_5DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3_5Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        layer_types = getattr(config, "layer_types", None)
        if layer_types is not None:
            self.layer_type = layer_types[layer_idx]
        else:
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, ...]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        router_logits_attn = None

        if self.layer_type == "linear_attention":
            # GDN uses cache_params (not past_key_value)
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_value,
                attention_mask=attention_mask,
            )
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
            self_attn_weights = None
            present_key_value = None
        else:
            hidden_states, self_attn_weights, present_key_value, router_logits_attn = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = residual + hidden_states

        # MLP with MoE routing
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_logits = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            combined_logits = router_logits
            if router_logits_attn is not None:
                combined_logits = torch.cat([router_logits, router_logits_attn], dim=0)
            outputs += (combined_logits,)

        return outputs


# ---------------------------------------------------------------------------
# Pre-trained model base
# ---------------------------------------------------------------------------

class Qwen3_5PreTrainedModel(PreTrainedModel):
    config_class = Qwen3_5Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3_5DecoderLayer"]

    def _init_weights(self, module):
        std = getattr(self.config, "initializer_range", 0.02)
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
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, MoeModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        output_router_logits = output_router_logits if output_router_logits is not None else False
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = 0
            if past_key_values is not None and hasattr(past_key_values, "get_seq_length"):
                past_seen_tokens = past_key_values.get_seq_length()
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position)

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits:
                all_router_logits += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(
                v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_router_logits]
                if v is not None
            )
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
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
# Causal LM wrapper with MoE
# ---------------------------------------------------------------------------

class Qwen3_5ForCausalLM(Qwen3_5PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3_5TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()
        print('Qwen3_5 MoE model init completed')

        # Apply width configurations from pruning
        self.cfgs = Qwen3_5ForCausalLM.cfgs
        model_replace(self, self.cfgs)

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
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, MoeCausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        output_router_logits = output_router_logits if output_router_logits is not None else False
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
            output_router_logits=output_router_logits,
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
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return MoeCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


# ---------------------------------------------------------------------------
# model_replace: set up pruned dimensions from cfgs vector
# ---------------------------------------------------------------------------

def model_replace(model, vectors):
    """Replace MLP and attention projections with pruned dimensions.

    vectors format: [mlp_0_dim, mlp_1_dim, ..., attn_k_dim, ..., num_experts]
    For Qwen 3.5: 32 MLP entries + 8 attention entries + 1 num_experts = 41 entries
    Order follows depth-first module walk: GDN layers contribute MLP only,
    full-attention layers contribute attention then MLP.
    """
    modules = list(model.modules())
    i = 0
    for m in modules:
        if type(m).__name__ == 'Qwen3_5MLP':
            mid_dim = vectors[i]
            experts_module = single_experts_module(mid_dim, m.config.hidden_size, experts=vectors[-1])

            gate_proj = nn.Linear(m.config.hidden_size, mid_dim, bias=False)
            up_proj = nn.Linear(m.config.hidden_size, mid_dim, bias=False)
            down_proj = nn.Linear(mid_dim, m.config.hidden_size, bias=False)

            m.gate_proj = gate_proj
            m.up_proj = up_proj
            m.down_proj = down_proj
            m.experts_module = experts_module
            i += 1

        if type(m).__name__ == 'Qwen3_5Attention':
            pruned_head_dim = vectors[i]
            experts_module = single_experts_module(
                m.head_dim, m.hidden_size, head_dim=m.head_dim, attn_flag=True, experts=vectors[-1]
            )
            experts_module.register_buffer('qk_index', torch.zeros(pruned_head_dim).to(torch.int64))
            m.experts_module = experts_module

            # Pruned Q projection (doubled for gate)
            pruned_q_proj = nn.Linear(m.hidden_size, m.num_heads * pruned_head_dim * 2, bias=False)
            pruned_k_proj = nn.Linear(m.hidden_size, m.num_key_value_heads * pruned_head_dim, bias=False)

            m.q_proj = pruned_q_proj
            m.k_proj = pruned_k_proj

            # Update head_dim for pruned model
            m.head_dim = pruned_head_dim

            # Update Q/K norms for new head_dim
            m.q_norm = Qwen3_5RMSNorm(pruned_head_dim, eps=m.config.rms_norm_eps)
            m.k_norm = Qwen3_5RMSNorm(pruned_head_dim, eps=m.config.rms_norm_eps)

            i += 1


# ---------------------------------------------------------------------------
# Loss utilities
# ---------------------------------------------------------------------------

def router_load_balance_loss(router_logits, top_k=1):
    num_experts = router_logits.shape[-1]
    _, selected_experts = torch.topk(router_logits, k=1, dim=-1)
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)
    tokens_per_expert = torch.mean(expert_mask.float(), dim=0)
    router_prob_per_expert = torch.mean(router_logits, dim=0)
    loss = torch.mean(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return loss * num_experts
