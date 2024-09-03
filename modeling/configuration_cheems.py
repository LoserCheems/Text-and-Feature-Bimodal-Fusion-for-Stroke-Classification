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
        hidden_size=1024,     
        intermediate_size=1024*4,
        hidden_bias=False,
        hidden_dropout=0.0,
        hidden_act="silu",
        max_position_embeddings=16384,
        rope_theta=10000.0,
        scaling_factor=1.0,
        
        initializer_range=0.02,
        rms_norm_eps=1e-06,
        use_cache=True,
        calc_logits_for_entire_prompt=True,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        torch_dtype="float32",
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        
        ssm_chunk_size=256,
        num_ssm_heads=16,
        num_ssm_groups=1,
        ssm_d_state=64,

        num_attn_heads=16,
        num_attn_groups=1,
        attn_implementation="sdpa",
        
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
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.scaling_factor = scaling_factor

        # 初始化配置
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.calc_logits_for_entire_prompt = calc_logits_for_entire_prompt
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef

        # SSM配置
        self.ssm_chunk_size = ssm_chunk_size
        self.num_ssm_heads = num_ssm_heads
        self.num_ssm_groups = num_ssm_groups
        self.ssm_d_state = ssm_d_state

        # Attention配置
        self.num_attn_heads = num_attn_heads
        self.num_attn_groups = num_attn_groups

        
        super().__init__(
            torch_dtype = torch_dtype,
            attn_implementation = attn_implementation,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
