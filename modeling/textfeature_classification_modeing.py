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
    from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
else:
    selective_state_update, selective_scan_fn, mamba_inner_fn = None, None, None

if is_causal_conv1d_available():
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
else:
    causal_conv1d_fn, causal_conv1d_update = None, None

is_fast_path_available = all(
    (selective_state_update, selective_scan_fn, causal_conv1d_fn, causal_conv1d_update, mamba_inner_fn)
)

logger = logging.get_logger(__name__)


def _config_to_kwargs(args):
    common_kwargs = {
        "dtype": args.torch_dtype,
    }
    return common_kwargs


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6, elementwise_affine: bool = True, bias: bool = True):
        """
        RMSNorm 是T5LayerNorm的等效
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
    ) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        # weight and bias
        if self.elementwise_affine:
            hidden_states = (hidden_states * self.weight).to(input_dtype)
            if self.bias is not None:
                hidden_states = (hidden_states + self.bias).to(input_dtype)

        return hidden_states


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    这是torch.repeat_interleave(x, dim=1, repeats=n_rep)的等效版本. 隐藏状态从(batch, num_key_value_heads, seqlen, head_dim)变为(batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class CheemsNonWoAttention(nn.Module):
    """
    Multi-headed attention 来自 'Attention Is All You Need' 论文. 修改为使用滑动窗口注意力: Longformer 和 "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: CheemsConfig, hidden_size: Tuple[int]):
        super().__init__()
        self.config = config

        self.hidden_size = hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size 必须能被 num_heads 整除 (得到 `hidden_size`: {self.hidden_size}"
                f" 和 `num_heads`: {self.num_heads})."
            )
    
        
        self.q_proj = nn.Linear(
            self.hidden_size, 
            self.num_heads * self.head_dim, 
            bias=config.hidden_bias,
            **_config_to_kwargs(config)
        )
        self.k_proj = nn.Linear(
            self.hidden_size, 
            self.num_key_value_heads * self.head_dim, bias=config.hidden_bias,
            **_config_to_kwargs(config)
        )
        self.v_proj = nn.Linear(
            self.hidden_size, 
            self.num_key_value_heads * self.head_dim, 
            bias=config.hidden_bias,
            **_config_to_kwargs(config)
        )

        # W_o 移动到外部进行共享
        # self.o_proj = nn.Linear(
        #     self.num_heads * self.head_dim, 
        #     self.hidden_size, 
        #     bias=config.hidden_bias,
        #     **_config_to_kwargs(config)
        # )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "传递`padding_mask`是不推荐的, 并且将在v4.37中删除. 请确保使用`attention_mask`."
            )
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"缓存结构自版本v4.36以来发生了变化. 如果您正在使用 {self.__class__.__name__} 进行自回归解码并使用k/v缓存, 请确保使用层索引初始化注意力类."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

        # 重复k/v头部, 如果n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        offset = 64
        query_length = query_states.size(1)
        key_length = key_states.size(1)
        logn = torch.arange(offset+1, offset+key_length+1, dtype=torch.float32, device=query_states.device)[-query_length:] # [query_length]
        base = torch.tensor(256).to(query_states.device) # 训练数据的平均长度
        logn = torch.log(logn) / torch.log(base)
        logn[logn < 1.0] = 1.0
        logn = logn.to(query_states.dtype).view(1, query_length, 1, 1)
        query_states = query_states * logn

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"注意力权重应该是大小为{(bsz, self.num_heads, q_len, kv_seq_len)}, 但是是{attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"注意力掩码应该是大小为{(bsz, 1, q_len, kv_seq_len)}, 但是是{attention_mask.size()}"
                )

            attn_weights = attn_weights + attention_mask

        # 将注意力上升到fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output`应该是大小为{(bsz, self.num_heads, q_len, self.head_dim)}, 但是是{attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        # W_o 移动到外部进行共享
        # attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class CheemsNonWoFlashAttention2(CheemsNonWoAttention):
    """
    cheems flash attention 模块. 此模块继承自 `CheemsAttention`, 因为模块的权重保持不变. 唯一需要更改的是在前向传递中, 它需要正确调用flash attention的公共API, 并在输入包含任何填充标记的情况下处理它们.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: 一旦RoCm的Flash Attention升级到2.1, 就应该删除这个. flash_attn<2.1生成左上对齐的因果掩码, 而这里需要的是右下对齐, 这是flash_attn>=2.1的默认设置. 这个属性用于处理这种差异. 参考: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # 请注意, 对于flash_attn<2.1, 使用q_seqlen != k_seqlen(除了q_seqlen == 1的情况)会产生一个错误的掩码(左上).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            **kwargs,
    ):
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
                "传递`padding_mask`已被弃用, 将在v4.37中删除. 请确保使用`attention_mask`代替."
            )

            # 使用padding_mask覆盖attention_mask
            attention_mask = kwargs.pop("padding_mask")
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"自版本v4.36以来, 缓存结构发生了变化. 如果您正在使用 {self.__class__.__name__} 进行自回归解码并使用k/v缓存, 请确保使用层索引初始化注意力类."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        use_sliding_windows = (
                _flash_supports_window_size
                and getattr(self.config, "sliding_window", None) is not None
                and kv_seq_len > self.config.sliding_window
        )

        if not _flash_supports_window_size:
            logger.warning_once(
                "当前的flash attention版本不支持滑动窗口注意力, 为了更高效的内存实现, 请确保升级flash-attn库."
            )

        if past_key_value is not None:
            # 激活切片缓存, 只有在配置中有一个值`sliding_windows`属性时
            cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
            if (
                    getattr(self.config, "sliding_window", None) is not None
                    and kv_seq_len > self.config.sliding_window
                    and cache_has_contents
            ):
                slicing_tokens = 1 - self.config.sliding_window

                past_key = past_key_value[self.layer_idx][0]
                past_value = past_key_value[self.layer_idx][1]

                past_key = past_key[:, :, slicing_tokens:, :].contiguous()
                past_value = past_value[:, :, slicing_tokens:, :].contiguous()

                if past_key.shape[-2] != self.config.sliding_window - 1:
                    raise ValueError(
                        f"过去的键必须具有形状(`batch_size, num_heads, self.config.sliding_window-1, head_dim`), 得到{past_key.shape}"
                    )

                if attention_mask is not None:
                    attention_mask = attention_mask[:, slicing_tokens:]
                    attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1)

            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

        # 如果n_kv_heads < n_heads, 重复k/v头
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # 在PEFT中, 通常我们为了训练稳定性的原因将层规范转换为float32, 因此输入隐藏状态会被静默地转换为float32. 因此, 我们需要将它们转换回float16, 以确保一切都按预期工作.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # 处理模型被量化的情况
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"输入隐藏状态似乎被静默地转换为float32, 这可能与您已经将嵌入或层规范层转换为float32有关. 我们将把输入转换回{target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # 重新调整形状以符合Flash Attention的预期形状
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
 

        offset = 64
        query_length = query_states.size(1)
        key_length = key_states.size(1)
        logn = torch.arange(offset+1, offset+key_length+1, dtype=torch.float32, device=query_states.device)[-query_length:] # [query_length]
        base = torch.tensor(256).to(query_states.device)
        logn = torch.log(logn) / torch.log(base)
        logn[logn < 1.0] = 1.0
        logn = logn.to(query_states.dtype).view(1, query_length, 1, 1)
        query_states = query_states * logn

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            use_sliding_windows=use_sliding_windows,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()

        # W_o 移动到外部进行共享
        # attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            query_length,
            dropout=0.0,
            softmax_scale=None,
            use_sliding_windows=False,
    ):
        """
        告知Flash Attention的forward方法, 如果输入隐藏状态至少包含一个填充标记, 首先取消填充输入, 然后计算注意力分数并填充最终注意力分数.

        args:
            query_states (`torch.Tensor`):
                要传递给Flash Attention API的输入查询状态
            key_states (`torch.Tensor`):
                要传递给Flash Attention API的输入键状态
            value_states (`torch.Tensor`):
                要传递给Flash Attention API的输入值状态
            attention_mask (`torch.Tensor`):
                填充掩码 - 对应于大小为`(batch_size, seq_len)`的张量, 其中0表示填充标记的位置, 1表示非填充标记的位置.
            dropout (`int`, *optional*):
                注意力dropout
            softmax_scale (`float`, *optional*):
                在应用softmax之前对QK^T进行缩放. 默认为1 / sqrt(head_dim)
            use_sliding_windows (`bool`, *optional*):
                是否激活滑动窗口注意力.
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: 一旦RoCm的Flash Attention升级到2.1, 就应该删除`query_length != 1`检查. 有关详细信息, 请参见LlamaFlashAttention2 __init__中的注释.
            if is_flash_attn_greater_or_equal_2_10():
                causal = self.is_causal
            else:
                causal = self.is_causal and query_length != 1

        # 序列中至少包含一个填充标记
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            if not use_sliding_windows:
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
            else:
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=(self.config.sliding_window, self.config.sliding_window),
                )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            if not use_sliding_windows:
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
            else:
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=(self.config.sliding_window, self.config.sliding_window),
                )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape

        # 在第一次迭代中, 我们需要通过在正确的位置切片它来正确重新创建填充掩码
        if kv_seq_len != attention_mask.shape[-1]:
            attention_mask_num_tokens = attention_mask.shape[-1]
            attention_mask = attention_mask[:, attention_mask_num_tokens - kv_seq_len :]

        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)

        key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)
        value_layer = index_first_axis(value_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)

        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # 这里有一个memcpy, 这是非常糟糕的.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # -q_len:切片假设左填充.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class CheemsNonWoSdpaAttention(CheemsNonWoAttention):
    """
    cheems attention 模块使用torch.nn.functional.scaled_dot_product_attention. 该模块继承自`CheemsAttention`, 因为模块的权重保持不变. 唯一的更改是在前向传递中, 以适应SDPA API.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: 一旦实现了这一点, 通过例如`model.config.attn_implementation = "manual"`来改进这个警告.
            logger.warning_once(
                "CheemsModel正在使用CheemsSdpaAttention, 但`torch.nn.functional.scaled_dot_product_attention`不支持`output_attentions=True`. 回退到手动注意力实现, 但是从Transformers版本v5.0.0开始, 将需要指定手动实现. 可以在加载模型时使用参数`attn_implementation='eager'`来删除此警告."
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"注意力掩码应该是大小为{(bsz, 1, q_len, kv_seq_len)}, 但是是{attention_mask.size()}"
                )

        # 当具有自定义attn_mask的非连续输入时, 使用内存高效后端的SDPA目前(torch==2.1.2)存在错误, 参考: https://github.com/pytorch/pytorch/issues/112577.
        
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()
        

        offset = 64
        query_length = query_states.size(1)
        key_length = key_states.size(1)
        logn = torch.arange(offset+1, offset+key_length+1, dtype=torch.float32, device=query_states.device)[-query_length:]
        base = torch.tensor(256).to(query_states.device)
        logn = torch.log(logn) / torch.log(base)
        logn[logn < 1.0] = 1.0
        logn = logn.to(query_states.dtype).view(1, query_length, 1, 1)
        query_states = query_states * logn

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # q_len > 1是必要的, 以匹配AttentionMaskConverter.to_causal_4d, 如果q_len == 1, 它不会创建一个因果掩码.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        # W_o 移动到外部进行共享
        # attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


NON_WO_ATTENTION_CLASSES = {
    "eager": CheemsNonWoAttention,
    "flash_attention_2": CheemsNonWoFlashAttention2,
    "sdpa": CheemsNonWoSdpaAttention,
}


class HybridMambaAttentionDynamicCache(DynamicCache):
    """
    一个动态缓存, 可以处理注意力缓存(具有seq_len维度)和mamba缓存(无论seq_len如何都具有恒定的形状).

    它将Key和Value状态存储为张量列表, 每个层一个.
    每个注意力层的预期形状为`[batch_size, num_heads, seq_len, head_dim]`.
    对于mamba层, `key_cache`表示卷积状态, 具有形状`[batch_size, d_inner, 1, d_conv]`, `value_cache`表示ssm状态, 具有形状`[batch_size, d_inner, 1, d_state]`.
    Mamba缓存形状[2]是一个虚拟的"seqlen"维度, 以匹配注意力缓存维度的数量. 对于mamba, 缓存不随seqlen增长, 因此该维度始终为1.
    """

    def __init__(self) -> None:
        super().__init__()
        self.attention_layer_idx = None  # 用于知道哪一层在缓存形状中有关于seqlen的数据

    def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        更新缓存, 使用新的`key_states`和`value_states`更新`layer_idx`层的缓存.

        Parameters:
            key_states (`torch.Tensor`):
                要缓存的新键状态.
            value_states (`torch.Tensor`):
                要缓存的新值状态.
            layer_idx (`int`):
                要缓存状态的层的索引.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                缓存子类的其他参数. 在`HybridMambaAttentionDynamicCache`中不使用其他参数.
        """
        # 更新已看到的token数量
        if self.attention_layer_idx is None and self._is_attn_layer(key_states, value_states):
            self.attention_layer_idx = layer_idx
        if self.attention_layer_idx is not None and layer_idx == self.attention_layer_idx:
            if hasattr(self, "_seen_tokens"):
                self._seen_tokens += key_states.shape[-2]
            else:
                self.seen_tokens += key_states.shape[-2]

        # 更新缓存
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            if self._is_attn_layer(self.key_cache[layer_idx], self.value_cache[layer_idx]):
        
                # 注意力层 - 将新状态附加到现有缓存的seqlen维度上
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            else:
                
                # mamba层 - 用新状态替换缓存
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = None) -> int:
        """返回缓存状态的序列长度. 可以选择传递一个层索引."""
        if layer_idx is not None:
            if len(self.key_cache) <= layer_idx:
                return 0
            if self._is_attn_layer(self.key_cache[layer_idx], self.value_cache[layer_idx]):
                return self.key_cache[layer_idx].shape[-2]
            else:
                warnings.warn(
                    f"要求从不是注意力层的第{layer_idx}层的缓存中获取序列长度. 忽略这一点, 使用一个注意力层缓存"

                )
        if self.attention_layer_idx is None or len(self.key_cache) <= self.attention_layer_idx:
            return 0
        return self.key_cache[self.attention_layer_idx].shape[-2]

    @staticmethod
    def _is_attn_layer(key_states: torch.Tensor, value_states: torch.Tensor):
        return key_states.shape[-1] == value_states.shape[-1]


@dataclass
class MambaCacheParams:
    seqlen_offset: int = 0
    conv_states: Dict[int, torch.Tensor] = field(default_factory=dict)
    ssm_states: Dict[int, torch.Tensor] = field(default_factory=dict)


class LognConv1d(nn.Conv1d):
    """
    对数卷积核权重: 调整卷积核的权重, 使其随着位置的增加而逐渐变化, 模拟对数位置编码的效果.
    """
    def __init__(self, in_channels, out_channels, bias, kernel_size, groups, padding, log_scale_init=0.0, dtype=torch.float32):
        super().__init__(in_channels, out_channels, kernel_size, groups=groups, padding=padding, bias=bias, dtype=dtype)
        self.log_scale = nn.Parameter(torch.tensor(log_scale_init, dtype=torch.float32))

        # TODO: 很糟糕的是, Mamba 是调用卷积的权重进行计算的, 并不直接使用这里定义的卷积forward方法. 这意味着我们只能在初始化时调整权重, 而不能在forward方法中调整权重. 我们急需对 Mamba 进行重构, 以便我们可以在forward方法中调整权重.

        # 获取卷积核的权重
        weight = self.weight # [out_channels, in_channels, kernel_size]
        offset = 64
        # 获取out_channels
        out_channels = weight.size(0)
        # 计算logn
        logn = torch.arange(offset+1, offset+out_channels+1, dtype=torch.float32)[-out_channels:] # [out_channels]
        # base 是训练数据的平均序列长度
        base = torch.tensor(256)
        logn = torch.log(logn) / torch.log(base)
        logn[logn < 1.0] = 1.0
        logn[logn > 1.0] *= torch.exp(self.log_scale)
        logn = logn.to(weight.dtype).view(out_channels, 1, 1)
        # 对卷积核的权重进行调整
        self.weight = nn.Parameter(weight * logn)
    
    def forward(
        self, 
        input: torch.Tensor
    ) -> torch.Tensor:
        # 计算卷积
        return F.conv1d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class CheemsMambaMixer(nn.Module):
    """
    计算∆, A, B, C和D状态空间参数, 并计算`contextualized_states`.
    A, D是独立于输入的(参见Mamba论文[1]第3.5.2节"对A的解释", 了解为什么A不是选择性的)
    ∆, B, C是依赖于输入的(这是Mamba和线性时不变S4之间的一个关键区别, 这就是为什么Mamba被称为**选择性**状态空间)
    """

    def __init__(self, config: CheemsConfig, layer_idx):
        super().__init__()
        self.config = config
        self.dtype = config.torch_dtype
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.mamba_d_state
        self.conv_kernel_size = config.mamba_d_conv
        self.intermediate_size = config.mamba_expand * config.hidden_size
        self.time_step_rank = config.mamba_dt_rank
        self.use_conv_bias = config.mamba_conv_bias
        self.use_bias = config.mamba_proj_bias
        # self.conv1d = nn.Conv1d(
        #     in_channels=self.intermediate_size,
        #     out_channels=self.intermediate_size,
        #     bias=self.use_conv_bias,
        #     kernel_size=self.conv_kernel_size,
        #     groups=self.intermediate_size,
        #     padding=self.conv_kernel_size - 1,
        # )
        self.conv1d = LognConv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            bias=self.use_conv_bias,
            kernel_size=self.conv_kernel_size,
            groups=self.intermediate_size,
            padding=self.conv_kernel_size - 1,
        )

        self.activation = config.mamba_act
        self.act = ACT2FN[config.mamba_act]
        self.apply_inner_layernorms = config.mamba_inner_layernorms

        self.use_fast_kernels = config.use_mamba_kernels

        # 输入隐藏状态的投影
        self.in_proj = nn.Linear(
            self.hidden_size, 
            self.intermediate_size * 2, 
            bias=self.use_bias,
        )
 
        # 用于使dt, B和C依赖于输入的选择性投影
        self.x_proj = nn.Linear(
            self.intermediate_size, 
            self.time_step_rank + self.ssm_state_size * 2, 
            bias=False,
        )

        # 时间步投影(离散化)
        self.dt_proj = nn.Linear(
            self.time_step_rank, 
            self.intermediate_size, 
            bias=True,
        )

        # S4D真实初始化. 这些不是离散化的!
        # 核心是加载它们, 计算离散状态, 然后写入更新的状态. 保持内存有界
        A = torch.arange(1, self.ssm_state_size + 1, dtype=torch.float32)[None, :]
        A = A.expand(self.intermediate_size, -1).contiguous()

        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.intermediate_size))
        self.out_proj = nn.Linear(
            self.intermediate_size, 
            self.hidden_size, 
            bias=self.use_bias,
            **_config_to_kwargs(config)
        )

        if self.apply_inner_layernorms:
            self.dt_layernorm = RMSNorm(self.time_step_rank, eps=config.rms_norm_eps)
            self.B_layernorm = RMSNorm(self.ssm_state_size, eps=config.rms_norm_eps)
            self.C_layernorm = RMSNorm(self.ssm_state_size, eps=config.rms_norm_eps)
        else:
            self.dt_layernorm = None
            self.B_layernorm = None
            self.C_layernorm = None

        if not is_fast_path_available:
            logger.warning_once(
                "快速路径不可用, 因为`(selective_state_update, selective_scan_fn, causal_conv1d_fn, causal_conv1d_update, mamba_inner_fn)`中的一个是None. 要安装, 请访问 https://github.com/state-spaces/mamba/#installation 和 https://github.com/Dao-AILab/causal-conv1d. 如果要使用朴素实现, 请在模型配置中设置`use_mamba_kernels=False`"
            )

    def _apply_layernorms(self, dt, B, C):
        if self.dt_layernorm is not None:
            dt = self.dt_layernorm(dt)
        if self.B_layernorm is not None:
            B = self.B_layernorm(B)
        if self.C_layernorm is not None:
            C = self.C_layernorm(C)
        return dt, B, C

    def cuda_kernels_forward(self, hidden_states: torch.Tensor, cache_params: MambaCacheParams = None):
        # 1. 门控MLP的线性投影
        projected_states = self.in_proj(hidden_states.to(torch.float32)).transpose(1, 2)

        if (
            self.training and cache_params is None and not self.apply_inner_layernorms
        ):  # 不支持输出状态 -> 用于训练
            contextualized_states = mamba_inner_fn(
                projected_states,
                self.conv1d.weight,
                self.conv1d.bias if self.use_conv_bias else None,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias.float() if self.use_bias else None,
                -torch.exp(self.A_log.float()),
                None,  # 输入相关的B
                None,  # 输入相关的C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )

        else:
            hidden_states, gate = projected_states.chunk(2, dim=1)

            # 2. 卷积序列转换
            conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
            if cache_params is not None and cache_params.seqlen_offset > 0:
                hidden_states = causal_conv1d_update(
                    hidden_states.squeeze(-1),
                    cache_params.conv_states[self.layer_idx].to(hidden_states.dtype),
                    conv_weights,
                    self.conv1d.bias,
                    self.activation,
                )
                hidden_states = hidden_states.unsqueeze(-1)
            else:
                if cache_params is not None:
                    conv_states = nn.functional.pad(
                        hidden_states, (self.conv_kernel_size - hidden_states.shape[-1], 0)
                    )
                    cache_params.conv_states[self.layer_idx].copy_(conv_states)
                hidden_states = causal_conv1d_fn(
                    hidden_states, conv_weights, self.conv1d.bias, activation=self.activation
                )

            # 3. 状态空间模型序列转换
            # 3.a. 时间步, B和C的输入变化初始化
            ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
            time_step, B, C = torch.split(
                ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
            )
            time_step, B, C = self._apply_layernorms(time_step, B, C)

            # 这里我们需要应用没有偏差的dt_proj, 因为偏差是在选择性扫描内核中添加的.
            # 这是一个应用dt_proj的hack, 同时仍然使用`torch.nn.Linear`的前向传递, 这是为了使量化工作.
            # 量化代码将`torch.nn.Linear`层替换为量化的线性层, 并要求直接调用前向传递.
            # 这里的原始代码是: ```discrete_time_step = self.dt_proj.weight @ time_step.transpose(1, 2)```
            
            if hasattr(self.dt_proj, "base_layer"):
                # 如果是LoRA, 我们需要访问基础层以获取权重
                time_proj_bias = self.dt_proj.base_layer.bias
                self.dt_proj.base_layer.bias = None
            else:
                time_proj_bias = self.dt_proj.bias
                self.dt_proj.bias = None
            discrete_time_step = self.dt_proj(time_step).transpose(1, 2)
            if hasattr(self.dt_proj, "base_layer"):
                self.dt_proj.base_layer.bias = time_proj_bias
            else:
                self.dt_proj.bias = time_proj_bias

            A = -torch.exp(self.A_log.float())
            # 3.c 执行循环 y ← SSM(A, B, C)(x)
            time_proj_bias = time_proj_bias.float() if time_proj_bias is not None else None
            if cache_params is not None and cache_params.seqlen_offset > 0:
                scan_outputs = selective_state_update(
                    cache_params.ssm_states[self.layer_idx],
                    hidden_states[..., 0],
                    discrete_time_step[..., 0],
                    A,
                    B[:, 0],
                    C[:, 0],
                    self.D,
                    gate[..., 0],
                    time_proj_bias,
                    dt_softplus=True,
                ).unsqueeze(-1)
            else:
                scan_outputs, ssm_state = selective_scan_fn(
                    hidden_states,
                    discrete_time_step,
                    A,
                    B.transpose(1, 2).float(),
                    C.transpose(1, 2),
                    self.D.float(),
                    gate,
                    time_proj_bias,
                    delta_softplus=True,
                    return_last_state=True,
                )
                if ssm_state is not None and cache_params is not None:
                    cache_params.ssm_states[self.layer_idx].copy_(ssm_state)

            # 4. 最终线性投影
            contextualized_states = self.out_proj(scan_outputs.to(self.dtype).transpose(1, 2))
        return contextualized_states

    # fmt: off
    def slow_forward(self, input_states, cache_params: MambaCacheParams = None):
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype
        # 1. 门控MLP的线性投影
        projected_states = self.in_proj(input_states.to(torch.float32)).transpose(1, 2) # [batch, 2 * intermediate_size, seq_len]
        hidden_states, gate = projected_states.chunk(2, dim=1)

        # 2. 卷积序列转换
        if cache_params is not None:
            if self.training:
                # 在训练模式下, 我们不希望对ssm_state执行原地操作, 以便我们可以计算反向传递
                ssm_state = cache_params.ssm_states[self.layer_idx].clone()
            else:
                ssm_state = cache_params.ssm_states[self.layer_idx]

            ssm_state = ssm_state.to(hidden_states.device)

            if cache_params.seqlen_offset > 0:
                conv_state = cache_params.conv_states[self.layer_idx] # [batch, intermediate_size, conv_kernel_size]
                conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
                conv_state[:, :, -1] = hidden_states[:, :, 0]
                cache_params.conv_states[self.layer_idx].copy_(conv_state)
                hidden_states = torch.sum(conv_state * self.conv1d.weight[:, 0, :], dim=-1)
                if self.use_conv_bias:
                    hidden_states += self.conv1d.bias
                hidden_states = self.act(hidden_states).to(dtype).unsqueeze(-1) # [batch, intermediate_size, 1] : decoding
            else:
                conv_state = nn.functional.pad(
                    hidden_states,
                    (self.conv_kernel_size - hidden_states.shape[-1], 0)
                )
                cache_params.conv_states[self.layer_idx].copy_(conv_state)
                hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len]) # [batch, intermediate_size, seq_len]
        else:
            ssm_state = torch.zeros(
                (batch_size, self.intermediate_size, self.ssm_state_size),
                device=hidden_states.device, dtype=dtype
            )
            hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len]) # [batch, intermediate_size, seq_len]

        # 3. 状态空间模型序列转换
        # 3.a. 选择: [batch, seq_len, self.time_step_rank + self.ssm_state_size * 2]
        ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
        time_step, B, C = torch.split(
            ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
        )
        time_step, B, C = self._apply_layernorms(time_step, B, C)
        discrete_time_step = self.dt_proj(time_step) # [batch, seq_len, intermediate_size]
        discrete_time_step = nn.functional.softplus(discrete_time_step).transpose(1, 2) # [batch, intermediate_size, seq_len]

        # 3.b. 离散化: B和C到[batch, seq_len, intermediate_size, ssm_state_size] (SRAM)
        A = -torch.exp(self.A_log.float()) # [intermediate_size, ssm_state_size]
        discrete_A = torch.exp(A[None, :, None, :] * discrete_time_step[:, :, :, None]) # [batch, intermediate_size, seq_len, ssm_state_size]
        discrete_B = discrete_time_step[:, :, :, None] * B[:, None, :, :].float() # [batch, intermediade_size, seq_len, ssm_state_size]
        deltaB_u = discrete_B * hidden_states[:, :, :, None].float()

        # 3.c 执行循环 y ← SSM(A, B, C)(x)
        scan_outputs = []
        for i in range(seq_len):
            ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :] # [batch, intermediade_size, ssm_state]
            scan_output = torch.matmul(ssm_state.to(dtype), C[:, i, :].unsqueeze(-1)) # [batch, intermediade_size, 1]
            scan_outputs.append(scan_output[:, :, 0])
        scan_output = torch.stack(scan_outputs, dim=-1) # [batch, intermediade_size, seq_len]
        scan_output = scan_output + (hidden_states * self.D[None, :, None])
        scan_output = (scan_output * self.act(gate))

        if cache_params is not None:
            cache_params.ssm_states[self.layer_idx].copy_(ssm_state)

        # 4. 最终线性投影
        contextualized_states = self.out_proj(scan_output.transpose(1, 2)) # [batch, seq_len, hidden_size]
        return contextualized_states
    # fmt: on

    def mixer_forward(self, hidden_states, cache_params: MambaCacheParams = None):
        if self.use_fast_kernels:
            if not is_fast_path_available or "cuda" not in self.x_proj.weight.device.type:
                raise ValueError(
                    "快速Mamba内核不可用. 确保它们已安装, 并且mamba模块在CUDA设备上"
                )
            return self.cuda_kernels_forward(hidden_states, cache_params)
        return self.slow_forward(hidden_states, cache_params)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[HybridMambaAttentionDynamicCache] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        if past_key_value is not None:
            cache_params = MambaCacheParams(
                seqlen_offset=0 if hidden_states.shape[1] > 1 else past_key_value.seen_tokens,
            )
            if len(past_key_value.key_cache) > self.layer_idx:
                # 我们已经为这一层缓存了, 使用它删除虚拟的seqlen维度(dim=2)
                cache_params.conv_states[self.layer_idx] = past_key_value.key_cache[self.layer_idx].squeeze(2)
                cache_params.ssm_states[self.layer_idx] = past_key_value.value_cache[self.layer_idx].squeeze(2)
            else:
                # 我们没有为这一层缓存, 用零初始化它
                batch_size = hidden_states.shape[0]
                cache_params.conv_states[self.layer_idx] = torch.zeros(
                    batch_size,
                    self.intermediate_size,
                    self.conv_kernel_size,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
                cache_params.ssm_states[self.layer_idx] = torch.zeros(
                    batch_size,
                    self.intermediate_size,
                    self.ssm_state_size,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
        else:
            cache_params = None

        res = self.mixer_forward(hidden_states, cache_params)

        if past_key_value is not None:
            past_key_value.update(
                # 添加虚拟的seqlen维度(dim=2)以匹配注意力缓存的维度数量
                cache_params.conv_states[self.layer_idx].unsqueeze(2),
                cache_params.ssm_states[self.layer_idx].unsqueeze(2),
                self.layer_idx,
            )

        return res, past_key_value


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
            **_config_to_kwargs(config)
        )
        # 改进点
        self.gate_act_fn = nn.Tanh()

        self.up_proj = nn.Linear(
            self.hidden_dim, 
            self.ffn_dim, 
            bias=config.hidden_bias,
            **_config_to_kwargs(config)
        )
        self.up_act_fn = ACT2FN[config.hidden_act]

        self.down_proj = nn.Linear(
            self.ffn_dim, 
            self.hidden_dim, 
            bias=config.hidden_bias,
            **_config_to_kwargs(config)
        )

    def forward(
        self, 
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        return self.down_proj(self.up_act_fn(self.up_proj(hidden_states.to(self.dtype))) * self.gate_act_fn(self.gate_proj(hidden_states.to(self.dtype))))


class CheemsCrossAttention(nn.Module):

    def __init__(self, config: CheemsConfig):
        super().__init__()
        self.config = config
        self.intermediate_size = config.hidden_size * 2

        # 输入
        self.text_in_proj = nn.Linear(
            config.hidden_size, 
            self.intermediate_size,
            bias=config.hidden_bias,
            **_config_to_kwargs(config)
        )
        self.feature_in_proj = nn.Linear(
            config.hidden_size, 
            self.intermediate_size,
            bias=config.hidden_bias,
            **_config_to_kwargs(config)
        )

        # 交叉注意力门控
        self.text_gate_proj = nn.Linear(
            config.hidden_size,
            self.intermediate_size,
            bias=config.hidden_bias,
            **_config_to_kwargs(config)
        )
        self.feature_gate_proj = nn.Linear(
            config.hidden_size,
            self.intermediate_size,
            bias=config.hidden_bias,
            **_config_to_kwargs(config)
        )
        self.gate_act_fn = ACT2FN[config.hidden_act]

        # Non-Wo-Attention
        self.non_wo_attention = NON_WO_ATTENTION_CLASSES[config._attn_implementation](config, self.intermediate_size)

        # 可以学习的缩放因子 (2个)
        self.log = nn.Parameter(torch.log(torch.ones(self.intermediate_size)))
        self.scale = nn.Parameter(torch.ones(self.intermediate_size))

        # W_o
        self.out_proj = nn.Linear(
            self.intermediate_size * 2,
            config.hidden_size, 
            bias=config.hidden_bias,
            **_config_to_kwargs(config)
        )

        self._attn_implementation = config._attn_implementation


    def forward(
        self,
        text_hidden_states: torch.Tensor,
        feature_hidden_states: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:

        batch_size, text_seq_len, _ = text_hidden_states.shape
        batch_size, feature_seq_len, _ = feature_hidden_states.shape

        if self._attn_implementation == "flash_attention_2":
            # 2d mask通过层传递
            text_attention_mask = text_attention_mask if (text_attention_mask is not None and 0 in text_attention_mask) else None
            feature_attention_mask = feature_attention_mask if (feature_attention_mask is not None and 0 in feature_attention_mask) else None
        elif self._attn_implementation == "sdpa":
            # 当使用SDPA时, 无法支持output_attentions=True, 我们回退到在所有情况下都需要4D因果掩码的手动实现.
            text_attention_mask = _prepare_4d_causal_attention_mask(
                text_attention_mask, 
                (batch_size, text_seq_len),
                text_hidden_states, 
                0,
            )
            feature_attention_mask = _prepare_4d_causal_attention_mask(
                feature_attention_mask, 
                (batch_size, feature_seq_len),
                feature_hidden_states, 
                0,
            )

        # in_proj
        text_gate_states, feature_gate_states = self.gate_act_fn(self.text_gate_proj(text_hidden_states)), self.gate_act_fn(self.feature_gate_proj(feature_hidden_states))
        text_hidden_states, feature_hidden_states = self.text_in_proj(text_hidden_states), self.feature_in_proj(feature_hidden_states)
        
        # Non-Wo-Attention
        text_hidden_states, _, _ = self.non_wo_attention(text_hidden_states, attention_mask=text_attention_mask)
        feature_hidden_states, _, _ = self.non_wo_attention(feature_hidden_states, attention_mask=feature_attention_mask)

        # 交叉注意力门控
        text_state = text_gate_states * text_hidden_states * torch.exp(self.log.float()) / torch.sqrt(self.scale)
        feature_state = feature_gate_states * feature_hidden_states * torch.exp(self.log.float()) / torch.sqrt(self.scale)
        cross_hidden_states = torch.cat([text_state, feature_state], dim=-1)

        # W_o
        hidden_state = self.out_proj(cross_hidden_states.to(self.out_proj.weight.dtype))

        
        return hidden_state
 

class CheemsCrossTransformer(nn.Module):
    
    def __init__(self, config: CheemsConfig):
        super().__init__()
        self.config = config

        # 文本解码器
        self.pre_text_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.text_decoder = CheemsMambaMixer(config, layer_idx=0)
        self.text_decoder_dropout = nn.Dropout(config.hidden_dropout)

        # 特征解码器
        self.pre_feature_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.feature_decoder = CheemsMambaMixer(config, layer_idx=0)
        self.feature_decoder_dropout = nn.Dropout(config.hidden_dropout)
        
        # 交叉注意力
        self.pre_cross_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.cross_attention = CheemsCrossAttention(config)
        self.cross_attention_dropout = nn.Dropout(config.hidden_dropout)

        # MLP层
        self.pre_mlp_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = CheemsMLP(config)
        self.mlp_dropout = nn.Dropout(config.hidden_dropout)


    def forward(
        self,
        text_hidden_states: torch.Tensor,
        feature_hidden_states: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:

        # 文本解码器
        text_residual = text_hidden_states
        text_hidden_states = self.pre_text_layernorm(text_hidden_states)
        text_hidden_states, _ = self.text_decoder(text_hidden_states)
        text_hidden_states = self.text_decoder_dropout(text_hidden_states)
        text_hidden_states = text_hidden_states + text_residual

        # 特征解码器
        feature_residual = feature_hidden_states
        feature_hidden_states = self.pre_feature_layernorm(feature_hidden_states)
        feature_hidden_states, _ = self.feature_decoder(feature_hidden_states)
        feature_hidden_states = self.feature_decoder_dropout(feature_hidden_states)
        feature_hidden_states = feature_hidden_states + feature_residual

        # 交叉注意力
        text_residual = text_hidden_states
        feature_residual = feature_hidden_states
        text_hidden_states = self.pre_cross_attention_layernorm(text_hidden_states)
        feature_hidden_states = self.pre_cross_attention_layernorm(feature_hidden_states)
        cross_hidden_states = self.cross_attention(text_hidden_states, feature_hidden_states, text_attention_mask, feature_attention_mask)
        cross_hidden_states = self.cross_attention_dropout(cross_hidden_states)
        cross_hidden_states = cross_hidden_states + text_residual + feature_residual

        # MLP层
        residual = cross_hidden_states
        hidden_states = self.pre_mlp_layernorm(cross_hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.mlp_dropout(hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states



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

    @staticmethod
    def _convert_to_standard_cache(
            past_key_value: Tuple[Tuple[torch.Tensor, torch.Tensor]], batch_size: int
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
        """
        标准化缓存的格式, 以匹配大多数实现, 即将seqlen作为第三个维度, 也适用于mamba层
        """
        attn_layer_index = [k.shape == v.shape for k, v in past_key_value].index(True)
        seqlen = past_key_value[attn_layer_index][0].shape[2]
        standard_past_key_value = ()
        for k, v in past_key_value:
            if k.shape != v.shape:
                # mamba层
                # 扩展不会使用更多内存, 因此在这里执行它是可以的
                standard_past_key_value += ((k.expand(-1, -1, seqlen, -1), v.expand(-1, -1, seqlen, -1)),)
            else:
                standard_past_key_value += ((k, v),)
        return standard_past_key_value

    @staticmethod
    def _convert_to_cheems_cache(
            past_key_value: Tuple[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
        """
        将缓存转换为期望的格式, 即对于mamba层, 具有大小1的虚拟seqlen维度
        """
        mamba_past_key_value = ()
        for k, v in past_key_value:
            if k.shape != v.shape:
                # mamba layer
                mamba_past_key_value += ((k[:, :, :1, :], v[:, :, :1, :]),)
            else:
                mamba_past_key_value += ((k, v),)
        return mamba_past_key_value


class Embedding(torch.nn.Module):
    def __init__(self, config: CheemsConfig):
        super(Embedding, self).__init__()
        
        self.hidden_size = config.hidden_size
        # 文本嵌入.
        self.text_embeddings = nn.Embedding(
            config.text_vocab_size,
            self.hidden_size,
            padding_idx=config.pad_token_id,
            **_config_to_kwargs(config),
        )
        # 特征嵌入.
        self.feature_embeddings = nn.Embedding(
            config.feature_vocab_size,
            self.hidden_size,
            padding_idx=config.pad_token_id,
            **_config_to_kwargs(config),
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

        # 解码器
        cross_hidden_states = self.decoder(
            text_hidden_states=text_embeddings,
            feature_hidden_states=feature_embeddings,
            text_attention_mask=text_attention_mask,
            feature_attention_mask=feature_attention_mask,
        )

        # 最终的LayerNorm
        hidden_states = self.final_layernorm(cross_hidden_states)

        if not return_dict:
            return hidden_states
        
        return CheemsModelOutput(
            hidden_states=hidden_states
        )


# [CLS]分类器
class DefaultSequenceClassifier(nn.Module):
    def __init__(self, config: CheemsConfig):
        super().__init__()
        self.classifier = nn.Linear(
            config.hidden_size, 
            config.num_labels, 
            bias=False,
            **_config_to_kwargs(config)
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
            **_config_to_kwargs(config)
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
            **_config_to_kwargs(config)
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
        self.classifier = MeanPoolSequenceClassifier(config)
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
