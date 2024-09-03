""" PyTorch Cheems model."""
import inspect
import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    AttentionMaskConverter,
)
from transformers.modeling_outputs import (
    ModelOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import is_torch_greater_or_equal_than_1_13
from transformers.utils import (
    is_flash_attn_greater_or_equal_2_10,
    logging,
)
from transformers.utils.import_utils import (
    is_flash_attn_2_available,
    is_mamba_ssm_available,
    is_causal_conv1d_available
)
from transformers.utils.import_utils import is_torch_fx_available
from .configuration_cheems import CheemsConfig

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)

# 这使得`_prepare_4d_causal_attention_mask`成为FX图中的叶函数.
# 这意味着该函数将不会被跟踪, 只是作为图中的一个节点出现.
if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)

if is_mamba_ssm_available():
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
else:
    mamba_chunk_scan_combined = None

logger = logging.get_logger(__name__)


def repeat_qk(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    这是torch.repeat_interleave(x, dim=1, repeats=n_rep)的等效版本. 隐藏状态从(batch, num_query_key_heads, seqlen, head_dim)变为(batch, num_attention_heads, seqlen, head_dim)

    This is an equivalent version of torch.repeat_interleave(x, dim=1, repeats=n_rep). Hidden states go from (batch, num_query_key_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """

    batch, seqlen, num_query_key_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, seqlen, n_rep, num_query_key_heads, head_dim)
    return hidden_states.reshape(batch, seqlen, num_query_key_heads * n_rep, head_dim)


def rotate_half(x):
    """
    旋转输入的一半隐藏维度.
    Rotates half the hidden dims of the input.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_QK_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_BC_rotary_pos_emb(b, c, cos, sin, unsqueeze_dim=2):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    b_embed = (b * cos) + (rotate_half(b) * sin)
    c_embed = (c * cos) + (rotate_half(c) * sin)
    return b_embed, c_embed


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        super().__init__()

        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(x.device) / self.dim)
            )
        else:
            inv_freq = self.inv_freq
        
        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6, elementwise_affine: bool = True, bias: bool = True):
        """
        RMSNorm 是T5LayerNorm的等效
        RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        if isinstance(hidden_size, int):
            hidden_size = (hidden_size,)
        self.hidden_size = tuple(hidden_size)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.empty(self.hidden_size))
            if bias:
                self.bias = nn.Parameter(torch.empty(self.hidden_size))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()
    

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)


    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        # 权重和偏置
        # weight and bias
        if self.elementwise_affine:
            hidden_states = (hidden_states * self.weight).to(input_dtype)
            if self.bias is not None:
                hidden_states = (hidden_states + self.bias).to(input_dtype)

        return hidden_states


class CheemsSSM(nn.Module):

    def __init__(self, config: CheemsConfig):
        super().__init__()

        self.config = config
        self.dtype = config.torch_dtype

        self.hidden_size = config.hidden_size

        self.num_ssm_heads = config.num_ssm_heads
        self.num_ssm_groups = config.num_ssm_groups
        self.ssm_head_dim = self.hidden_size // self.num_ssm_heads
        self.ssm_state_size = config.ssm_d_state
        self.chunk_size = config.ssm_chunk_size

        self.x_proj = nn.Linear(
            self.hidden_size, 
            self.hidden_size, 
            bias=config.hidden_bias,
        )
        self.B_proj = nn.Linear(
            self.hidden_size,
            self.ssm_state_size * self.num_ssm_groups,
            bias=config.hidden_bias,
        )
        self.C_proj = nn.Linear(
            self.hidden_size,
            self.ssm_state_size * self.num_ssm_groups,
            bias=config.hidden_bias,
        )
        self.dt_proj = nn.Linear(
            self.hidden_size,
            self.num_ssm_heads,
            bias=config.hidden_bias,
        )
        dt = torch.exp(
            torch.rand(self.num_ssm_heads) * (math.log(1e-1) - math.log(1e-3))
            + math.log(1e-3)
        )
        dt = torch.clamp(dt, min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)

        A = torch.empty(self.num_ssm_heads, dtype=torch.float32).uniform_(1, self.num_ssm_heads)
        A_log = torch.log(A).to(dtype=self.dtype)
        self.A_log = nn.Parameter(A_log)

        self.out_proj = nn.Linear(
            self.hidden_size, 
            self.hidden_size, 
            bias=config.hidden_bias,
        )

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.scaling_factor = config.scaling_factor
        self.BC_rotary_emb = RotaryEmbedding(
            self.ssm_state_size,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
            scaling_factor=self.scaling_factor
        )


    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.size()
        
        x = self.x_proj(hidden_states)
        A = -torch.exp(self.A_log)
        B = self.B_proj(hidden_states)
        C = self.C_proj(hidden_states)
        dt = self.dt_proj(hidden_states)
        
        x = x.view(bsz, seq_len, self.num_ssm_heads, self.ssm_head_dim)
        B = B.view(bsz, seq_len, self.num_ssm_groups, self.ssm_state_size)
        C = C.view(bsz, seq_len, self.num_ssm_groups, self.ssm_state_size)

        cos, sin = self.BC_rotary_emb(hidden_states, position_ids=position_ids)
        B, C = apply_BC_rotary_pos_emb(B, C, cos, sin)
        
        y = mamba_chunk_scan_combined(
            x,
            dt,
            A,
            B,
            C,
            chunk_size=self.chunk_size,
            dt_bias=self.dt_bias,
            dt_softplus=True
        )

        y = y.view(bsz, seq_len, self.hidden_size).contiguous()
        out = self.out_proj(y)

        return out


class CheemsNonWoAttn(nn.Module):

    def __init__(self, config: CheemsConfig):
        super().__init__()

        self.config = config
        self.dtype = config.torch_dtype

        self.hidden_size = config.hidden_size

        self.num_attn_heads = config.num_attn_heads
        self.num_attn_groups = config.num_attn_groups
        self.attn_head_dim = self.hidden_size // self.num_attn_heads
        self.num_query_key_heads = self.num_attn_heads // self.num_attn_groups
        self.attn_is_causal = True

        self.Q_proj = nn.Linear(
            self.hidden_size,
            self.attn_head_dim * self.num_query_key_heads,
            bias=config.hidden_bias,
        )
        self.K_proj = nn.Linear(
            self.hidden_size,
            self.attn_head_dim * self.num_query_key_heads,
            bias=config.hidden_bias,
        )
        self.V_proj = nn.Linear(
            self.hidden_size,
            self.attn_head_dim * self.num_attn_heads,
            bias=config.hidden_bias,
        )

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.scaling_factor = config.scaling_factor

        self.QK_rotary_emb = RotaryEmbedding(
            self.attn_head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
            scaling_factor=self.scaling_factor
        )


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.size()

        Q = self.Q_proj(hidden_states)
        K = self.K_proj(hidden_states)
        V = self.V_proj(hidden_states)

        Q = Q.view(bsz, seq_len, self.num_query_key_heads, self.attn_head_dim)
        K = K.view(bsz, seq_len, self.num_query_key_heads, self.attn_head_dim)
        V = V.view(bsz, seq_len, self.num_attn_heads, self.attn_head_dim)

        cos, sin = self.QK_rotary_emb(V, position_ids=position_ids)
        Q, K = apply_QK_rotary_pos_emb(Q, K, cos, sin)
    
        Q = repeat_qk(Q, self.num_attn_groups)
        K = repeat_qk(K, self.num_attn_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : seq_len]

        attn = torch.nn.functional.scaled_dot_product_attention(
            Q.transpose(1, 2),
            K.transpose(1, 2),
            V.transpose(1, 2),
            attn_mask=causal_mask,
            is_causal=self.attn_is_causal and attention_mask is None and seq_len > 1,
        ).transpose(1, 2)

        attn = attn.view(bsz, seq_len, self.hidden_size).contiguous()
        
        return attn


class CheemsAttn(nn.Module):

    def __init__(self, config: CheemsConfig):
        super().__init__()

        self.config = config
        self.dtype = config.torch_dtype

        self.hidden_size = config.hidden_size

        self.num_attn_heads = config.num_attn_heads
        self.num_attn_groups = config.num_attn_groups
        self.attn_head_dim = self.hidden_size // self.num_attn_heads
        self.num_query_key_heads = self.num_attn_heads // self.num_attn_groups
        self.attn_is_causal = True

        self.Q_proj = nn.Linear(
            self.hidden_size,
            self.attn_head_dim * self.num_query_key_heads,
            bias=config.hidden_bias,
        )
        self.K_proj = nn.Linear(
            self.hidden_size,
            self.attn_head_dim * self.num_query_key_heads,
            bias=config.hidden_bias,
        )
        self.V_proj = nn.Linear(
            self.hidden_size,
            self.attn_head_dim * self.num_attn_heads,
            bias=config.hidden_bias,
        )
        self.out_proj = nn.Linear(
            self.hidden_size, 
            self.hidden_size, 
            bias=config.hidden_bias,
        )

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.scaling_factor = config.scaling_factor

        self.QK_rotary_emb = RotaryEmbedding(
            self.attn_head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
            scaling_factor=self.scaling_factor
        )


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.size()

        Q = self.Q_proj(hidden_states)
        K = self.K_proj(hidden_states)
        V = self.V_proj(hidden_states)

        Q = Q.view(bsz, seq_len, self.num_query_key_heads, self.attn_head_dim)
        K = K.view(bsz, seq_len, self.num_query_key_heads, self.attn_head_dim)
        V = V.view(bsz, seq_len, self.num_attn_heads, self.attn_head_dim)

        cos, sin = self.QK_rotary_emb(V, position_ids=position_ids)
        Q, K = apply_QK_rotary_pos_emb(Q, K, cos, sin)

        Q = repeat_qk(Q, self.num_attn_groups)
        K = repeat_qk(K, self.num_attn_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : seq_len]

        attn = torch.nn.functional.scaled_dot_product_attention(
            Q.transpose(1, 2),
            K.transpose(1, 2),
            V.transpose(1, 2),
            attn_mask=causal_mask,
            is_causal=self.attn_is_causal and attention_mask is None and seq_len > 1,
        ).transpose(1, 2)

        attn = attn.view(bsz, seq_len, self.hidden_size).contiguous()
        out = self.out_proj(attn)
        
        return out


class CheemsMLP(nn.Module):
    def __init__(self, config: CheemsConfig):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size
        self.dtype = config.torch_dtype

        self.gate_proj = nn.Linear(
            self.hidden_dim, 
            self.ffn_dim, 
            bias=config.hidden_bias,
        )
        self.gate_act_fn = ACT2FN[config.hidden_act]

        self.up_proj = nn.Linear(
            self.hidden_dim, 
            self.ffn_dim, 
            bias=config.hidden_bias,
        )
        self.up_act_fn = nn.Sigmoid()

        self.down_proj = nn.Linear(
            self.ffn_dim, 
            self.hidden_dim, 
            bias=config.hidden_bias,
        )

    def forward(
        self, 
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return self.down_proj(self.up_act_fn(self.up_proj(hidden_states.to(self.dtype))) * self.gate_act_fn(self.gate_proj(hidden_states.to(self.dtype))))


class CheemsCrossAttention(nn.Module):

    def __init__(self, config: CheemsConfig):
        super().__init__()
        self.config = config

        # 输入
        self.text_in_proj = nn.Linear(
            config.hidden_size, 
            config.hidden_size,
            bias=config.hidden_bias,
        )
        self.feature_in_proj = nn.Linear(
            config.hidden_size, 
            config.hidden_size,
            bias=config.hidden_bias,
        )

        # 交叉注意力门控
        self.text_gate_proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=config.hidden_bias,
        )
        self.feature_gate_proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=config.hidden_bias,
        )
        self.gate_act_fn = ACT2FN[config.hidden_act]

        # Attention
        self.text_attention = CheemsNonWoAttn(config)
        self.feature_attention = CheemsNonWoAttn(config)

        # 可以学习的缩放因子 (2个)
        self.text_scale = nn.Parameter(torch.ones(config.hidden_size))
        self.feature_scale = nn.Parameter(torch.ones(config.hidden_size))

        # W_o
        self.out_proj = nn.Linear(
            config.hidden_size * 2,
            config.hidden_size, 
            bias=config.hidden_bias,
        )


    def forward(
        self,
        text_hidden_states: torch.Tensor,
        feature_hidden_states: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        text_position_ids: Optional[torch.LongTensor] = None,
        feature_position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:

        # in_proj
        text_gate_states, feature_gate_states = self.gate_act_fn(self.text_gate_proj(text_hidden_states)), self.gate_act_fn(self.feature_gate_proj(feature_hidden_states))
        text_hidden_states, feature_hidden_states = self.text_in_proj(text_hidden_states), self.feature_in_proj(feature_hidden_states)
        
        # Attention
        text_hidden_states = self.text_attention(text_hidden_states, text_attention_mask, text_position_ids)
        feature_hidden_states = self.feature_attention(feature_hidden_states, feature_attention_mask, feature_position_ids)

        # 交叉注意力门控
        text_state = text_gate_states * text_hidden_states * torch.exp(self.text_scale)
        feature_state = feature_gate_states * feature_hidden_states * torch.exp(self.feature_scale)
        cross_hidden_states = torch.cat([text_state, feature_state], dim=-1)

        # W_o
        hidden_state = self.out_proj(cross_hidden_states.to(self.out_proj.weight.dtype))
        return hidden_state


class CheemsTransformer(nn.Module):
    
    def __init__(self, config: CheemsConfig):
        super().__init__()
        self.config = config

        self.pre_attention_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = CheemsAttn(config)
        self.post_attention_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = CheemsMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:

        residual = hidden_states
        hidden_states = self.pre_attention_norm(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask, position_ids)
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.post_attention_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states


class CheemsSSMformer(nn.Module):
        
    def __init__(self, config: CheemsConfig):
        super().__init__()
        self.config = config

        self.pre_ssm_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ssm = CheemsSSM(config)
        self.post_ssm_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = CheemsMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:

        residual = hidden_states
        hidden_states = self.pre_ssm_norm(hidden_states)
        hidden_states = self.ssm(hidden_states, position_ids)
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.post_ssm_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states


 
class CheemsCrossTransformer(nn.Module):
    
    def __init__(self, config: CheemsConfig):
        super().__init__()
        self.config = config

        # 文本解码器
        self.pre_text_decoder_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.text_decoder = CheemsSSM(config)
        self.post_text_decoder_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.text_mlp = CheemsMLP(config)

        # 特征解码器
        self.pre_feature_decoder_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.feature_decoder = CheemsSSM(config)
        self.post_feature_decoder_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.feature_mlp = CheemsMLP(config)
        
        # 交叉注意力
        self.text_pre_cross_attention_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.feature_pre_cross_attention_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.cross_attention = CheemsCrossAttention(config)
        self.post_cross_attention_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.cross_mlp = CheemsMLP(config)

        # transformer
        self.attn = nn.ModuleList([CheemsTransformer(config) for _ in range(8)])

        # # ssmformer
        # self.ssm = nn.ModuleList([CheemsSSMformer(config) for _ in range(1)])

    def forward(
        self,
        text_hidden_states: torch.Tensor,
        feature_hidden_states: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        text_position_ids: Optional[torch.LongTensor] = None,
        feature_position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:

        # 文本解码器
        text_residual = text_hidden_states
        text_hidden_states = self.pre_text_decoder_norm(text_hidden_states)
        text_hidden_states = self.text_decoder(text_hidden_states, position_ids=text_position_ids)
        text_hidden_states = text_hidden_states + text_residual

        text_residual = text_hidden_states
        text_hidden_states = self.post_text_decoder_norm(text_hidden_states)
        text_hidden_states = self.text_mlp(text_hidden_states)
        text_hidden_states = text_hidden_states + text_residual

        # 特征解码器
        feature_residual = feature_hidden_states
        feature_hidden_states = self.pre_feature_decoder_norm(feature_hidden_states)
        feature_hidden_states = self.feature_decoder(feature_hidden_states, position_ids=feature_position_ids)
        feature_hidden_states = feature_hidden_states + feature_residual

        feature_residual = feature_hidden_states
        feature_hidden_states = self.post_feature_decoder_norm(feature_hidden_states)
        feature_hidden_states = self.feature_mlp(feature_hidden_states)
        feature_hidden_states = feature_hidden_states + feature_residual

        # 交叉注意力
        text_hidden_states = self.text_pre_cross_attention_norm(text_hidden_states)
        feature_hidden_states = self.feature_pre_cross_attention_norm(feature_hidden_states)
        cross_hidden_states = self.cross_attention(
            text_hidden_states,
            feature_hidden_states,
            text_attention_mask,
            feature_attention_mask,
            text_position_ids,
            feature_position_ids,
        )

        residual = cross_hidden_states
        cross_hidden_states = self.post_cross_attention_norm(cross_hidden_states)
        cross_hidden_states = self.cross_mlp(cross_hidden_states)
        cross_hidden_states = cross_hidden_states + residual

        # transformer
        for transformer in self.attn:
            cross_hidden_states = transformer(cross_hidden_states, text_attention_mask, text_position_ids)

        # # ssmformer
        # for ssm in self.ssm:
        #     cross_hidden_states = ssm(cross_hidden_states, text_position_ids)

        return cross_hidden_states



class CheemsPreTrainedModel(PreTrainedModel):
    config_class = CheemsConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["CheemsCrossTransformer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class Embedding(torch.nn.Module):
    def __init__(self, config: CheemsConfig):
        super(Embedding, self).__init__()
        
        self.hidden_size = config.hidden_size
        # 文本嵌入.
        self.text_embeddings = nn.Embedding(
            config.text_vocab_size,
            self.hidden_size,
            padding_idx=config.pad_token_id,
        )
        # 特征嵌入.
        self.feature_embeddings = nn.Embedding(
            config.feature_vocab_size,
            self.hidden_size,
            padding_idx=config.pad_token_id,
        )

    def forward(
        self, 
        input_ids: Optional[torch.Tensor] = None,
        feature_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 文本嵌入.
        text_embeddings = self.text_embeddings(input_ids)
        # 特征嵌入.
        feature_embeddings = self.feature_embeddings(feature_ids)
        
        return text_embeddings, feature_embeddings


@dataclass
class CheemsModelOutput(ModelOutput):
    hidden_states: torch.FloatTensor = None


class CheemsModel(CheemsPreTrainedModel):

    def __init__(self, config: CheemsConfig):
        super().__init__(config)
        self.config = config
        self.text_feature_embedding = Embedding(config)

        # 解码器
        self.decoder = CheemsCrossTransformer(config)
        
        self._attn_implementation = config._attn_implementation
        # 最终的LayerNorm
        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # 初始化权重并应用最终处理
        self.post_init()
    
    def get_input_embeddings(self):
        return self.text_feature_embedding, self.text_feature_embedding
    
    def set_input_embeddings(self, text_embeddings, feature_embeddings):
        self.text_feature_embedding.text_embeddings = text_embeddings
        self.text_feature_embedding.feature_embeddings = feature_embeddings

    def forward(
        self,
        text_ids: Optional[torch.Tensor] = None,
        feature_ids: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> torch.Tensor:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 文本嵌入和特征嵌入.
        text_embeddings, feature_embeddings = self.text_feature_embedding(text_ids, feature_ids)

        # position_ids
        text_position_ids = torch.arange(text_embeddings.shape[1], device=text_embeddings.device)
        feature_position_ids = torch.arange(feature_embeddings.shape[1], device=feature_embeddings.device)

        # causal_mask
        text_causal_mask = self._update_causal_mask(text_attention_mask, text_embeddings, text_position_ids)
        feature_causal_mask = self._update_causal_mask(feature_attention_mask, feature_embeddings, feature_position_ids)

        text_position_ids = text_position_ids.unsqueeze(0)
        feature_position_ids = feature_position_ids.unsqueeze(0)

        # 解码器
        cross_hidden_states = self.decoder(
            text_hidden_states=text_embeddings,
            feature_hidden_states=feature_embeddings,
            text_attention_mask=text_causal_mask,
            feature_attention_mask=feature_causal_mask,
            text_position_ids=text_position_ids,
            feature_position_ids=feature_position_ids,
        )

        # 最终的LayerNorm
        hidden_states = self.final_layernorm(cross_hidden_states)

        if not return_dict:
            return hidden_states
        
        return CheemsModelOutput(
            hidden_states=hidden_states
        )


    def _update_causal_mask(self, attention_mask, input_tensor, cache_position):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        target_length = cache_position[-1] + 1

        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit 复制到连续内存以进行原地编辑
            if attention_mask.dim() == 2:
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
                causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            # 注意 causal_mask 中完全掩码行中的所有令牌, 例如在使用左填充时相关的第一行. 这是由F.scaled_dot_product_attention节省内存的注意力路径所需的.
            # 详细信息: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask


# [CLS]分类器
class DefaultSequenceClassifier(nn.Module):
    def __init__(self, config: CheemsConfig):
        super().__init__()
        self.classifier = nn.Linear(
            config.hidden_size, 
            config.num_labels, 
            bias=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.classifier(hidden_states[:, -1])
        return logits


# 最大池化分类器
class MaxPoolSequenceClassifier(nn.Module):
    def __init__(self, config: CheemsConfig):
        super().__init__()
        self.classifier = nn.Linear(
            config.hidden_size, 
            config.num_labels, 
            bias=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.classifier(torch.max(hidden_states, dim=1).values)
        return logits


# 平均池化分类器
class MeanPoolSequenceClassifier(nn.Module):
    def __init__(self, config: CheemsConfig):
        super().__init__()
        self.classifier = nn.Linear(
            config.hidden_size, 
            config.num_labels, 
            bias=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.classifier(torch.mean(hidden_states, dim=1))
        return logits


@dataclass
class SequenceClassifierOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: torch.Tensor = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None


class CheemsForSequenceClassification(CheemsPreTrainedModel):
    def __init__(self, config: CheemsConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.cheems = CheemsModel(config)
        self.classifier = DefaultSequenceClassifier(config)
        self.post_init()
    
    def forward(
        self,
        text_ids: Optional[torch.Tensor] = None,
        feature_ids: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Cheems模型
        output = self.cheems(
            text_ids=text_ids,
            feature_ids=feature_ids,
            text_attention_mask=text_attention_mask,
            feature_attention_mask=feature_attention_mask,
            return_dict=return_dict,
        )

        # 分类器
        logits = self.classifier(output.hidden_states)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=output.hidden_states,
        )
