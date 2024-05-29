import math

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class CheemsConfig(PretrainedConfig):
    model_type = "cheems"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_vocab_size=65536,
        feature_vocab_size=2,
        hidden_size=2048,     
        intermediate_size=2048*4,
        hidden_bias=False,
        hidden_dropout=0.1,
        hidden_act="silu",

        num_attention_heads=64,
        num_key_value_heads=32,
        attn_implementation="flash_attention_2",
        sliding_window=None,
        n_ctx=262144,
        attention_dropout=0.1,
        
        initializer_range=0.02,
        rms_norm_eps=1e-06,
        use_cache=True,
        calc_logits_for_entire_prompt=True,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        torch_dtype="bfloat16",
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        
        use_mamba_kernels=True,
        mamba_act="silu",
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_dt_rank="auto",
        mamba_conv_bias=False,
        mamba_proj_bias=False,
        mamba_inner_layernorms=True,
        mamba_in_mlp=False,
        mamba_out_mlp=False,
        
        **kwargs
    ):

        # 基础配置
        self.text_vocab_size = text_vocab_size
        self.feature_vocab_size = feature_vocab_size
        self.tie_word_embeddings = tie_word_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_bias = hidden_bias
        self.hidden_dropout = hidden_dropout
        self.hidden_act = hidden_act

        # Attention配置
        self.num_attention_heads = num_attention_heads
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.attn_implementation = attn_implementation
        self.sliding_window = sliding_window
        self.n_ctx = n_ctx
        self.attention_dropout = attention_dropout

        # 初始化配置
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.calc_logits_for_entire_prompt = calc_logits_for_entire_prompt
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef

        # Mamba 配置
        self.use_mamba_kernels = use_mamba_kernels
        self.mamba_act = mamba_act
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand
        self.mamba_dt_rank = math.ceil(self.hidden_size / 16) if mamba_dt_rank == "auto" else mamba_dt_rank
        self.mamba_conv_bias = mamba_conv_bias
        self.mamba_proj_bias = mamba_proj_bias
        self.mamba_inner_layernorms = mamba_inner_layernorms
        self.mamba_in_mlp = mamba_in_mlp
        self.mamba_out_mlp = mamba_out_mlp
        
        super().__init__(
            torch_dtype = torch_dtype,
            attn_implementation = attn_implementation,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
