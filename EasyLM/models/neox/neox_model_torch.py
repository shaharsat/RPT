# coding=utf-8
# Copyright 2022 EleutherAI The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch GPTNeoX model."""
import warnings
from distutils import dist
from typing import Optional, Tuple, Union, Dict, Any, List

import einops
import numpy as np
import torch
import torch.utils.checkpoint
from more_itertools import chunked
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from transformers import AutoTokenizer, LogitsProcessorList, StoppingCriteriaList
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.generation import GenerateEncoderDecoderOutput, GenerateDecoderOnlyOutput, validate_stopping_criteria
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from EasyLM.models.neox.attention_torch import my_dot_product_attention_weights
from EasyLM.models.neox.gate_torch import GriffinGate
from EasyLM.models.neox.rpt_torch_utils import EncodedNeighbors, new_lookup_neighbors, GPTNeoXRetrieverNeighborOutput, \
    GPTNeoXRetrieverEncodedOutput, GPTNeoXLMOutput, GPTNeoXModelOutput, add_batch_index, create_prepare_inputs
from EasyLM.torch_utils import make_causal_mask, assign_slice, combine_masks

class GPTNeoXConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GPTNeoXModel`]. It is used to instantiate an
    GPTNeoX model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the GPTNeoX
    [EleutherAI/gpt-neox-20b](https://huggingface.co/EleutherAI/gpt-neox-20b) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 50432):
            Vocabulary size of the GPTNeoX model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`GPTNeoXModel`].
        hidden_size (`int`, *optional*, defaults to 6144):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 44):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 64):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 24576):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        rotary_pct (`float`, *optional*, defaults to 0.25):
            percentage of hidden dimensions to allocate to rotary embeddings
        rotary_emb_base (`int`, *optional*, defaults to 10000)
            base for computing rotary embeddings frequency
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio probability of the attention score.
        hidden_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio of (1) the word embeddings, (2) the post-attention hidden states, and (3) the post-mlp
            hidden states.
        classifier_dropout (`float`, *optional*, defaults to 0.1):
            Argument used when doing token classification, used in the model [`GPTNeoXForTokenClassification`].

            The dropout ratio for the hidden layer.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        initializer_range (`float`, *optional*, defaults to 1e-5):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        use_dec_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        use_parallel_residual (`bool`, *optional*, defaults to `True`):
            Whether to use a "parallel" formulation in each Transformer layer, which can provide a slight training
            speedup at large scales (e.g. 20B).
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
            `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
            these scaling strategies behave:
            https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This is an
            experimental feature, subject to breaking API changes in future versions.
        attention_bias (`bool`, *optional*, defaults to `True`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.

        Example:

    ```python
    >>> from transformers import GPTNeoXConfig, GPTNeoXModel

    >>> # Initializing a GPTNeoX gpt-neox-20b style configuration
    >>> configuration = GPTNeoXConfig()

    >>> # Initializing a model (with random weights) from the gpt-neox-20b style configuration
    >>> model = GPTNeoXModel(configuration)  # doctest: +SKIP

    >>> # Accessing the model configuration
    >>> configuration = model.config  # doctest: +SKIP
    ```"""

    model_type = "gpt_neox"

    def __init__(
            self,
            vocab_size=50432,
            hidden_size=6144,
            num_hidden_layers=44,
            num_attention_heads=64,
            intermediate_size=24576,
            hidden_act="gelu",
            rotary_pct=0.25,
            rotary_emb_base=10000,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            classifier_dropout=0.1,
            max_position_embeddings=2048,
            initializer_range=0.02,
            layer_norm_eps=1e-5,
            use_dec_cache=True,
            bos_token_id=0,
            eos_token_id=0,
            tie_word_embeddings=False,
            use_parallel_residual=True,
            rope_scaling=None,
            attention_bias=True,
            cca_freq: int = 0,
            retriever_fill_value: float = -10000.0,
            threshold_nei_scores: float = 0.0,
            null_attn_init: float = 0.0001,
            num_neighbors: int = 2,
            chunk_size: int = 64,
            ss_schedule_steps: int = 450000,
            scheduled_sampling_max_prob: float = 1.0,
            scheduled_sampling_min_prob: float = 0.01,
            atsc_margin_min: float = 1.0,
            num_scored_neighbors: int = 20,
            score_temp: float = 0.05,
            aux_scale: float = 1.0,
            document_length: int = 16384,
            cca_layernorm_init_scale: float = 0.0001,
            set_pt_params: bool = False,
            qk_layernorm: bool = False,
            null_k_init_mult: float = 1.0,
            mult_by_ndcg: bool = True,
            ret_score_in_cca: bool = True,
            remat_attention: str = "",
            apply_refactor: bool = False,
            n_query_aug_layers: Optional[int] = None,
            query_aug_dim: int = 128,
            gate_num_blocks: int = 8,
            lowres_ss: bool = False,
            apply_gating: bool = True,
            gate_refactor: bool = False,
            do_last_gate: bool = False,
            a_init_query: Optional[float] = None,
            a_init_nei: Optional[float] = None,
            tanh_xatt: bool = False,
            tanh_causal_att: bool = False,
            apply_query_to_nei_att: bool = False,
            append_next_chunk: bool = True,
            pooling_size: Optional[int] = -1,
            log_debug_metrics: bool = False,
            stop_grad_trick: bool = False,
            apply_tanh_in_cca: bool = False,
            use_allowed_tar_mask: bool = False,
            **kwargs,
    ):
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.use_allowed_tar_mask = use_allowed_tar_mask
        self.apply_tanh_in_cca = apply_tanh_in_cca
        self.stop_grad_trick = stop_grad_trick
        self.log_debug_metrics = log_debug_metrics
        self.append_next_chunk = append_next_chunk
        self.pooling_size = pooling_size
        self.apply_query_to_nei_att = apply_query_to_nei_att
        self.tanh_causal_att = tanh_causal_att
        self.tanh_xatt = tanh_xatt
        self.a_init_query = a_init_query
        self.a_init_nei = a_init_nei
        self.do_last_gate = do_last_gate
        self.gate_refactor = gate_refactor
        self.lowres_ss = lowres_ss
        self.apply_gating = apply_gating
        self.n_query_aug_layers = n_query_aug_layers
        self.gate_num_blocks = gate_num_blocks
        self.query_aug_dim = query_aug_dim
        self.apply_refactor = apply_refactor
        self.remat_attention = remat_attention
        self.ret_score_in_cca = ret_score_in_cca
        self.mult_by_ndcg = mult_by_ndcg
        self.qk_layernorm = qk_layernorm
        self.null_k_init_mult = null_k_init_mult
        self.set_pt_params = set_pt_params
        self.cca_layernorm_init_scale = cca_layernorm_init_scale
        self.document_length = document_length
        self.aux_scale = aux_scale
        self.atsc_margin_min = atsc_margin_min
        self.num_scored_neighbors = num_scored_neighbors
        self.score_temp = score_temp
        self.ss_schedule_steps = ss_schedule_steps
        self.scheduled_sampling_max_prob = scheduled_sampling_max_prob
        self.scheduled_sampling_min_prob = scheduled_sampling_min_prob
        self.num_neighbors = num_neighbors
        self.null_attn_init = null_attn_init
        self.chunk_size = chunk_size
        self.retriever_fill_value = retriever_fill_value
        self.threshold_nei_scores = threshold_nei_scores
        self.cca_freq = cca_freq
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.rotary_pct = rotary_pct
        self.rotary_emb_base = rotary_emb_base
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout
        self.classifier_dropout = classifier_dropout
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_dec_cache = use_dec_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.use_parallel_residual = use_parallel_residual
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self._rope_scaling_validation()

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "The hidden size is not divisble by the number of attention heads! Make sure to update them!"
            )

    # Copied from transformers.models.llama.configuration_llama.LlamaConfig._rope_scaling_validation
    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `type` and `factor`, "
                f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be a float > 1, got {rope_scaling_factor}")

    @classmethod
    def get_tokenizer(cls, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b",
                                                  pad_token='<|endoftext|>',
                                                  mask_token='<|endoftext|>',
                                                  **kwargs)
        return tokenizer


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "trl-internal-testing/tiny-random-GPTNeoXForCausalLM"
_REAL_CHECKPOINT_FOR_DOC = "EleutherAI/gpt-neox-20b"
_CONFIG_FOR_DOC = "GPTNeoXConfig"


# Copied from transformers.models.llama.modeling_llama._get_unpad_data
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


class GPTNeoXAttention(nn.Module):
    def __init__(self, config, dtype):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "The hidden size is not divisble by the number of attention heads! Make sure to update them"
            )
        self.head_size = self.hidden_size // self.num_attention_heads
        self.rotary_ndims = int(self.head_size * config.rotary_pct)

        self._init_rope()

        self.norm_factor = self.head_size ** -0.5
        self.query_key_value = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=config.attention_bias)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.is_causal = True

        if config.rope_scaling is None:
            max_seq_length = config.max_position_embeddings
        else:
            max_seq_length = int(config.max_position_embeddings * config.rope_scaling["factor"])

        self.causal_mask = make_causal_mask(torch.ones((1, max_seq_length), dtype=torch.bool), dtype=torch.bool)

        # TODO: Verify size
        self.register_buffer('cached_key', torch.zeros((1, 1, self.config.hidden_size, self.config.hidden_size)))
        self.register_buffer('cached_value', torch.zeros((1, 1, self.config.hidden_size, self.config.hidden_size)))
        self.register_buffer('cache_index', torch.Tensor([0]))

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = GPTNeoXRotaryEmbedding(
                self.rotary_ndims, self.config.max_position_embeddings, base=self.config.rotary_emb_base
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = GPTNeoXLinearScalingRotaryEmbedding(
                    self.rotary_ndims,
                    self.config.max_position_embeddings,
                    base=self.config.rotary_emb_base,
                    scaling_factor=scaling_factor,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = GPTNeoXDynamicNTKScalingRotaryEmbedding(
                    self.rotary_ndims,
                    self.config.max_position_embeddings,
                    base=self.config.rotary_emb_base,
                    scaling_factor=scaling_factor,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _concatenate_to_cache(self, key, value, query, attention_mask):
        cached_key = self.get_buffer("cached_key")
        cached_value = self.get_buffer("cached_value")
        cache_index = self.get_buffer("cache_index")

        *batch_dims, max_length, num_heads, depth_per_head = cached_key.shape
        cur_index = cache_index
        indices = (0,) * len(batch_dims) + (cur_index, 0, 0)

        key = assign_slice(cached_key, key, indices)
        value = assign_slice(cached_value, value, indices)

        self.register_buffer('cached_key', key)
        self.register_buffer('cached_value', value)
        num_updated_cache_vectors = query.shape[1]
        self.register_buffer('cache_index', cache_index + num_updated_cache_vectors)

        pad_mask = torch.broadcast_to(
            torch.arange(max_length) < cur_index + num_updated_cache_vectors,
            tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
        ).to(key.device)

        attention_mask = combine_masks(pad_mask, attention_mask)

        return key, value, attention_mask

    def forward(
            self,
            hidden_states: torch.FloatTensor,
            attention_mask: torch.FloatTensor,
            position_ids: torch.LongTensor,
            deterministic,
            init_cache: bool = False,
            head_mask: Optional[torch.FloatTensor] = None,
            layer_past: Optional[Tuple[torch.Tensor]] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False,
            padding_mask: Optional[torch.Tensor] = None,
    ):
        has_layer_past = layer_past is not None

        # Compute QKV
        # Attention heads [batch, seq_len, hidden_size]
        #   --> [batch, seq_len, (np * 3 * head_size)]
        # qkv = self.query_key_value(hidden_states)

        fused_qkv = self.query_key_value(hidden_states)
        batch, seq_len, _ = fused_qkv.shape
        fused_qkv = self._split_heads(fused_qkv)
        query, key, value = torch.split(fused_qkv, fused_qkv.shape[-1]//3, dim=-1) # TODO: Verify seq_len

        cos, sin = self.rotary_emb(seq_len)
        cos, sin = cos.to(hidden_states.device), sin.to(hidden_states.device)

        if self.rotary_ndims is not None:
            k_rot = key[..., : self.rotary_ndims]
            k_pass = key[..., self.rotary_ndims:]

            q_rot = query[..., : self.rotary_ndims]
            q_pass = query[..., self.rotary_ndims:]

            q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin, position_ids)

            key = torch.cat([k_rot, k_pass], dim=-1)
            query = torch.cat([q_rot, q_pass], dim=-1)
        else:
            query, key = apply_rotary_pos_emb(query, key, cos, sin, position_ids)
        query = query.type(self.dtype)
        key = key.type(self.dtype)

        query_length, key_length = query.shape[1], key.shape[1]

        if False: # TODO: Handle Falsify
            mask_shift = self.get_buffer("cache_index")
            max_decoder_length = self.get_buffer("cached_key").shape[1]

            causal_mask = assign_slice(
                self.causal_mask, (1, 1, query_length, max_decoder_length), (0, 0, mask_shift, 0)
            )
        else:
            causal_mask = self.causal_mask[:, :, :query_length, :key_length]

        causal_mask = torch.broadcast_to(causal_mask, (batch,) + causal_mask.shape[1:]).to(hidden_states.device)
        # TODO: Using views instead?

        attention_mask = torch.broadcast_to(attention_mask.unsqueeze(1).unsqueeze(1), causal_mask.shape).to(hidden_states.device)
        attention_mask = combine_masks(attention_mask, causal_mask)

        if False: # TODO: Unfalse
            key, value, attention_mask = self._concatenate_to_cache(key, value, query, attention_mask)

        # Cache QKV values
        if has_layer_past:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        present = (key, value) if use_cache else None

        # transform boolean mask into float mask
        attention_bias = torch.full(attention_mask.shape, torch.finfo(self.dtype).min).type(self.dtype).to(hidden_states.device)
        attention_bias[attention_mask > 0] = 0.0

        # TODO: Add deterministic
        # Compute attention
        attn_weights = my_dot_product_attention_weights(
            query,
            key,
            bias=attention_bias,
            dropout_rate=self.config.attention_dropout,
            dtype=torch.float32,
            apply_tanh=self.config.tanh_causal_att,
        )

        attn_weights = attn_weights.type(self.dtype)
        attn_output = torch.einsum("bhqk,bkhd->bqhd", attn_weights, value.type(self.dtype))

        # Reshape outputs
        attn_output = self._merge_heads(attn_output)
        attn_output = self.dense(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs

    def _split_heads(self, hidden_states):
        coeff = hidden_states.shape[-1] // self.hidden_size

        return hidden_states.reshape(hidden_states.shape[:-1] + (self.num_attention_heads, self.head_size * coeff))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.hidden_size,))

def attention_mask_func(attention_scores, ltor_mask):
    attention_scores.masked_fill_(~ltor_mask, torch.finfo(attention_scores.dtype).min)
    return attention_scores


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, dtype: torch.dtype = torch.float32,
                         device=None) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].type(dtype) / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs).type(dtype)
    sin, cos = torch.sin(freqs), torch.cos(freqs)
    freqs_cis = cos + 1j * sin
    return torch.tensor(freqs_cis, device=device)


def apply_rotary_emb_(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
        dtype: torch.dtype = torch.float32,
        freqs_cis_k: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    reshape_xq = xq.type(torch.float32).reshape(*xq.shape[:-1], -1, 2)
    reshape_xk = xk.type(torch.float32).reshape(*xk.shape[:-1], -1, 2)

    xq_ = torch.complex(reshape_xq[..., 0], reshape_xq[..., 1]).to(xq.device)
    xk_ = torch.complex(reshape_xk[..., 0], reshape_xk[..., 1]).to(xk.device)

    freqs_cis = torch.reshape(freqs_cis, (*freqs_cis.shape[:2], 1, *freqs_cis.shape[2:]))

    xq_out = xq_ * freqs_cis
    xq_out = torch.stack((torch.real(xq_out), torch.imag(xq_out)), dim=-1).reshape(*xq_out.shape[:-1], -1)
    if freqs_cis_k is None:
        xk_out = xk_ * freqs_cis
        xk_out = torch.stack((torch.real(xk_out), torch.imag(xk_out)), dim=-1).reshape(*xk_out.shape[:-1], -1)
    else:
        freqs_cis_k = torch.reshape(freqs_cis_k, (*freqs_cis_k.shape[:2], 1, *freqs_cis_k.shape[2:]))
        xk_out = xk_ * freqs_cis_k
        xk_out = torch.stack((torch.real(xk_out), torch.imag(xk_out)), dim=-1).reshape(*xk_out.shape[:-1], -1)

    return xq_out.type(dtype), xk_out.type(dtype)


def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
        dtype: torch.dtype = torch.float32,
        freqs_cis_k: torch.Tensor = None,
        rot_dim: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if rot_dim is not None and rot_dim > 0:

        # Separate the tensors based on the rotation dimensions
        xq_rot, xq_pass = xq[..., :rot_dim], xq[..., rot_dim:]
        xk_rot, xk_pass = xk[..., :rot_dim], xk[..., rot_dim:]

        # freqs_q_rot = freqs_q[..., :rot_dim]
        # freqs_k_rot = freqs_k[..., :rot_dim] if freqs_k is not None else None

        # Apply the function on the parts that need rotation
        ##print(freqs_cis.shape, xq_rot.shape, xk_rot.shape)
        xq_rot, xk_rot = apply_rotary_emb_(xq_rot, xk_rot, freqs_cis, dtype=dtype, freqs_cis_k=freqs_cis_k)

        # Concatenate the rotated and non-rotated parts
        xq_out = torch.cat((xq_rot, xq_pass), dim=-1)
        xk_out = torch.cat((xk_rot, xk_pass), dim=-1)
    else:
        xq_out, xk_out = apply_rotary_emb_(xq, xk, freqs_cis, dtype=dtype, freqs_cis_k=freqs_cis_k)

    return xq_out, xk_out


class GPTNeoXCrossAttention(nn.Module):

    def __init__(self, config, dtype):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.qk_layernorm = config.qk_layernorm
        self.head_dim = self.embed_dim // self.num_heads

        self.wq = nn.Linear(
            self.embed_dim,
            config.num_attention_heads * self.head_dim,
            bias=self.config.attention_bias,
            dtype=dtype
        )

        self.wk = nn.Linear(
            self.embed_dim,
            config.num_attention_heads * self.head_dim,
            bias=self.config.attention_bias,
            dtype=dtype
        )

        self.wv = nn.Linear(
            self.embed_dim,
            config.num_attention_heads * self.head_dim,
            bias=self.config.attention_bias,
            dtype=dtype
        )

        self.wo = nn.Linear(
            self.embed_dim,
            config.num_attention_heads * self.head_dim,
            bias=self.config.attention_bias,
            dtype=dtype
        )

        self.freqs_cis = precompute_freqs_cis(
            self.head_dim,
            config.max_position_embeddings * 2,
        )
        self.null_k = nn.Parameter(torch.normal(mean=0, std=0.0001, size=(1, 1, self.num_heads, self.head_dim)))
        self.null_v = nn.Parameter(torch.normal(mean=0, std=0.0001, size=(1, 1, self.num_heads, self.head_dim)))
        if self.qk_layernorm:
            self.q_layernorm = nn.LayerNorm(normalized_shape=self.head_dim, eps=self.config.layer_norm_eps, dtype=self.dtype)
            self.k_layernorm = nn.LayerNorm(normalized_shape=self.head_dim, eps=self.config.layer_norm_eps, dtype=self.dtype)

    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    def forward(
            self,
            hidden_states: torch.Tensor,
            key_value_states: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            kv_position_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            retriever_scores: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
            deterministic: bool = True,
    ) -> Tuple[torch.Tensor]:
        is_cross_attention = key_value_states is not None

        xq = self.wq(hidden_states)
        xq = self._split_heads(xq).type(self.dtype)
        xq = self.q_layernorm(xq) if self.qk_layernorm else xq

        if not is_cross_attention:
            key_value_states = hidden_states

        xk = self.wk(key_value_states)
        xk = self._split_heads(xk).type(self.dtype)
        xk = self.k_layernorm(xk) if self.qk_layernorm else xk

        xv = self.wv(key_value_states)
        xv = self._split_heads(xv).type(self.dtype)

        null_k = self.k_layernorm(self.null_k) if self.qk_layernorm else self.null_k

        query_length, key_length = xq.shape[1], xk.shape[1]
        batch_size = hidden_states.shape[0]
        ##print(f"{xq.shape=}, {xk.shape=}, {self.has_variable('cache', 'cached_key')=}")
        #print(f"{xq.shape=}, {xk.shape=}")

        if position_ids is None:
            position_ids = torch.arange(query_length, dtype=torch.int32)
            position_ids = torch.broadcast_to(position_ids[None, :], (batch_size, query_length)).type(torch.long)

        freqs_cis = self.freqs_cis[position_ids]

        if not is_cross_attention:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis, dtype=self.dtype, rot_dim=self.head_dim)
        else:
            if kv_position_ids is None:
                kv_position_ids = torch.arange(key_length, dtype=torch.int32)
                kv_position_ids = torch.broadcast_to(kv_position_ids[None, :], (batch_size, key_length))
            freqs_cis_k = self.freqs_cis[kv_position_ids]
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis, freqs_cis_k=freqs_cis_k, dtype=self.dtype, rot_dim=self.head_dim)

        null_k = torch.broadcast_to(null_k, (batch_size, 1, self.num_heads, self.head_dim)).type(self.dtype)
        xk = torch.cat((xk, null_k), dim=-3)
        null_v = torch.broadcast_to(self.null_v, (batch_size, 1, self.num_heads, self.head_dim)).type(self.dtype)
        xv = torch.cat((xv, null_v), dim=-3)

        if attention_mask is not None:
            null_mask = torch.ones((attention_mask.shape[0], 1), dtype=torch.float32)
            attention_mask = torch.cat((attention_mask, null_mask), dim=-1)
            attention_mask = attention_mask.unsqueeze(-2).unsqueeze(-2)

            if retriever_scores is None:
                attention_bias = torch.full(attention_mask.shape, 0.0).type(self.dtype)
            else:
                null_ret_score = torch.zeros((retriever_scores.shape[0], 1), dtype=torch.float32)
                attention_bias = torch.cat((retriever_scores, null_ret_score), dim=-1)
                attention_bias = attention_bias.unsqueeze(-2).unsqueeze(-2)

            attention_bias[attention_mask <= 0] = torch.finfo(self.dtype).min

            if xq.shape[0] != attention_bias.shape[0]:
                attention_bias = attention_bias[:batch_size, ...]
        else:
            attention_bias = None

        attn_weights = my_dot_product_attention_weights(
            xq,
            xk,
            bias=attention_bias,
            dropout_rate=self.config.attention_dropout,
            deterministic=deterministic,
            dtype=self.dtype,
            apply_tanh=self.config.tanh_xatt,
        )

        attn_output = torch.einsum("...hqk,...khd->...qhd", attn_weights, xv.type(self.dtype))

        attn_output = self._merge_heads(attn_output)
        attn_output = self.wo(attn_output).type(self.dtype)
        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)

        return outputs


class GPTNeoXChunkedCrossAttention(nn.Module):
    def __init__(self, config, dtype):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.chunk_size = self.config.chunk_size
        self.num_neighbors = self.config.num_neighbors
        self.cross_attention = GPTNeoXCrossAttention(self.config, dtype=self.dtype)

    def forward(
            self,
            hidden_states: torch.Tensor,
            neighbor_hidden_states,
            neighbor_mask,
            nei_position_ids,
            att_scores,
            position_ids: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
            deterministic: bool = True,
            chunk_size: Optional[int] = None,
            n_chunks_per_window: Optional[int] = None,
    ):
        if not self.config.ret_score_in_cca:
            att_scores = None

        func = self.new_call
        func = torch.vmap(func,
                          in_dims=(0, 0, 0, 0, 0 if att_scores is not None else None,
                                   0 if position_ids is not None else None, None, None, None, None),
                          out_dims=0,
                          )
        return func(hidden_states, neighbor_hidden_states, neighbor_mask, nei_position_ids, att_scores,
                    position_ids, output_attentions, deterministic, chunk_size, n_chunks_per_window)

    def new_call(
            self,
            hidden_states: torch.Tensor,
            neighbor_hidden_states,
            neighbor_mask,
            nei_position_ids,
            att_scores,
            position_ids: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
            deterministic: bool = True,
            chunk_size: Optional[int] = None,
            n_chunks_per_window: Optional[int] = None,
    ):

        batch_size, seq_len, hidden_dim = hidden_states.shape
        if chunk_size is None:
            chunk_size = self.chunk_size
        # chunk_size = self.chunk_size
        causal_padding = chunk_size - 1
        is_generating = position_ids is not None
        num_document_chunks, num_neighbors, ret_size = neighbor_mask.shape
        neighbor_mask = einops.rearrange(neighbor_mask, 'b k r -> b (k r)')
        nei_position_ids = einops.rearrange(nei_position_ids, 'b k r -> b (k r)')
        neighbor_hidden_states = einops.rearrange(neighbor_hidden_states, 'b k r d-> b (k r) d')
        if att_scores is not None:
            att_scores = einops.rearrange(att_scores, 'b k r -> b (k r)')

        # TODO: remove this
        nei_position_ids = torch.clip(nei_position_ids, min=0, max=2 * self.config.chunk_size - 1)

        # -> (-1, n_chunks_per_window, num_neighbors, 2*chunk_size, hidden_dim)
        if not is_generating:
            if n_chunks_per_window is None:
                # n_chunks_per_window = seq_len//chunk_size
                n_chunks_per_window = num_document_chunks // hidden_states.shape[0]
            # ->  (-1 ,chunk_size, hidden_dim)
            hidden_states = hidden_states.reshape([-1, n_chunks_per_window * chunk_size, hidden_dim])
            hidden_states = torch.pad(hidden_states[:, causal_padding:, :], ((0, 0), (0, causal_padding), (0, 0)),
                                      'constant')
            hidden_states = hidden_states.reshape([-1, chunk_size, hidden_dim])

            position_ids = torch.arange(chunk_size) + chunk_size - 1
            position_ids = torch.broadcast_to(position_ids, hidden_states.shape[:2])
        else:
            hidden_states = hidden_states.reshape([1, 1, hidden_dim])

        # cross attention
        output = self.cross_attention(
            hidden_states=hidden_states,
            key_value_states=neighbor_hidden_states,
            position_ids=position_ids,
            kv_position_ids=nei_position_ids,
            attention_mask=neighbor_mask,
            retriever_scores=att_scores,
            output_attentions=output_attentions,
            deterministic=deterministic,
        )

        # reshape back to original sequence
        cross_attention_out = output[0]
        if not is_generating:
            cross_attention_out = cross_attention_out.reshape([-1, n_chunks_per_window * chunk_size, hidden_dim])
            # # pad back to original, with 0s at the beginning (which will be added to the residual and be fine)
            cross_attention_out = torch.pad(cross_attention_out, ((0, 0), (causal_padding, 0), (0, 0)), 'constant')[:,
                                  :-causal_padding]
        cross_attention_out = cross_attention_out.reshape([batch_size, seq_len, hidden_dim])
        return (cross_attention_out,) + output[1:]


class GPTNeoXRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int, base: int = 10000, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.dtype = dtype

        fraction = torch.arange(0, self.dim, 2).type(torch.float32) / self.dim

        self.inv_freq = (1.0 / (self.base ** (fraction))).type(torch.float32)
        self.cos_cached, self.sin_cached = self._compute_cos_sin(self.max_position_embeddings)

    def _get_cos_sin_cache(self, seq_len):
        if seq_len > self.max_position_embeddings:
            return self._compute_cos_sin(seq_len)
        else:
            return self.cos_cached, self.sin_cached

    def _compute_cos_sin(self, seq_len):
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.einsum(
            "i,j->ij",
            t,
            self.inv_freq,
        )
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = torch.cos(emb).unsqueeze(0).unsqueeze(0) # TODO: Verify
        sin = torch.sin(emb).unsqueeze(0).unsqueeze(0)
        return cos, sin

    def forward(self, seq_len=None):
        cos_cached, sin_cached = self._get_cos_sin_cache(seq_len)
        return cos_cached[:seq_len, ...].type(torch.float32), sin_cached[:seq_len, ...].type(torch.float32)

# copied from transformers.models.llama.modeling_llama.LlamaLinearScalingRotaryEmbedding.__init__
# TODO @gante bring compatibility back
class GPTNeoXLinearScalingRotaryEmbedding(GPTNeoXRotaryEmbedding):
    """GPTNeoXRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        t = t / self.scaling_factor

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)


class GPTNeoXDynamicNTKScalingRotaryEmbedding(GPTNeoXRotaryEmbedding):
    """GPTNeoXRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    # copied from transformers.models.llama.modeling_llama.LlamaDynamicNTKScalingRotaryEmbedding.__init__
    # TODO @gante no longer copied from
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                    (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.mistral.modeling_mistral.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    gather_indices = position_ids[:, :, None, None].type(torch.int)  # [bs, seq_len, 1, 1]
    gather_indices = torch.repeat_interleave(gather_indices, cos.shape[1], dim=1)
    gather_indices = torch.repeat_interleave(gather_indices, cos.shape[3], dim=3).type(torch.long)
    cos = torch.take_along_dim(torch.repeat_interleave(cos, gather_indices.shape[0], dim=0), gather_indices, dim=2)
    sin = torch.take_along_dim(torch.repeat_interleave(sin, gather_indices.shape[0], dim=0), gather_indices, dim=2)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed



class GPTNeoXMLP(nn.Module):
    def __init__(self, config, dtype=torch.float32, intermediate_size=None):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.intermediate_size = intermediate_size

        embed_dim = self.config.hidden_size
        if self.intermediate_size is None:
            intermediate_size = self.config.intermediate_size
        else:
            intermediate_size = self.intermediate_size

        self.dense_h_to_4h = nn.Linear(embed_dim, intermediate_size)
        self.dense_4h_to_h = nn.Linear(intermediate_size, embed_dim)
        self.act = ACT2FN[self.config.hidden_act]

    def forward(self, hidden_states):
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dense_4h_to_h(hidden_states)
        return hidden_states


class GPTNeoXQueryAugmentorLayer(nn.Module):

    def __init__(self, config, dtype):
        super().__init__()
        self.config = config
        self.dtype = dtype

        kwargs = dict(width=self.config.hidden_size, dtype=self.dtype, param_dtype=self.param_dtype,
                      num_blocks=self.config.gate_num_blocks, after_refactor=self.config.gate_refactor)
        self.nei_gate = GriffinGate(**kwargs, a_init=self.config.a_init_nei) if self.config.apply_gating else None
        self.nei_to_query_att = GPTNeoXCrossAttention(self.config, dtype=self.dtype)
        self.nei_att_ln = nn.LayerNorm(eps=self.config.layer_norm_eps, dtype=self.dtype)
        self.nei_lowrank_mlp = GPTNeoXMLP(self.config, dtype=self.dtype, intermediate_size=self.config.query_aug_dim)
        self.nei_lowrank_mlp_ln = nn.LayerNorm(eps=self.config.layer_norm_eps, dtype=self.dtype)

        self.query_att_ln = nn.LayerNorm(eps=self.config.layer_norm_eps, dtype=self.dtype)
        if self.config.apply_query_to_nei_att:
            self.query_to_nei_att = GPTNeoXCrossAttention(self.config, dtype=self.dtype)
            self.query_lowrank_mlp = GPTNeoXMLP(self.config, dtype=self.dtype,
                                                intermediate_size=self.config.query_aug_dim)
            self.query_lowrank_mlp_ln = nn.LayerNorm(eps=self.config.layer_norm_eps, dtype=self.dtype)
            self.query_gate = GriffinGate(**kwargs,
                                          a_init=self.config.a_init_query) if self.config.apply_gating else None
        else:
            self.query_to_nei_att = None
            self.query_lowrank_mlp = None
            self.query_lowrank_mlp_ln = None

    def forward(self,
                query,
                query_attention_mask,
                nei,
                nei_attention_mask,
                nei_position_ids=None,
                chunk_size=None,
                is_last=False,
                ):
        if chunk_size is None:
            chunk_size = self.config.chunk_size

        original_query, original_nei = query, nei
        query, nei = self.query_att_ln(query), self.nei_att_ln(nei)
        num_document_chunks, num_neighbors, ret_size, hidden_dim = nei.shape

        if nei_position_ids is None:
            nei_position_ids = torch.arange(ret_size)
            nei_position_ids = torch.broadcast_to(nei_position_ids[None, None, :],
                                                  (num_document_chunks, num_neighbors, ret_size))
        else:
            nei_position_ids = nei_position_ids.reshape([num_document_chunks, num_neighbors, ret_size])
        query = query.reshape([-1, chunk_size, hidden_dim])
        query_attention_mask = query_attention_mask.reshape([-1, chunk_size])
        # ###
        if self.config.apply_query_to_nei_att:
            query_w_nei = self.query_to_nei_att(query,
                                                key_value_states=einops.rearrange(nei, 'b k r d-> b (k r) d'),
                                                attention_mask=einops.rearrange(nei_attention_mask, 'b k r -> b (k r)'),
                                                kv_position_ids=einops.rearrange(nei_position_ids, 'b k r -> b (k r)'))
            aug_query = query + query_w_nei[0]
            aug_query = self.query_lowrank_mlp(self.query_lowrank_mlp_ln(aug_query)) + aug_query
            aug_query = aug_query.reshape(original_query.shape)
            if self.query_gate is not None:
                # if using griffin gate, beta is between [0 ,  U[0.1, 0.7] ] at init.
                # alpha is in [U[0.3, 0.9] , 1] at init.
                ######
                # at init query_alpha should be at least ~0.99
                query_alpha, query_beta = self.query_gate(aug_query)
            else:
                query_alpha, query_beta = 0, 1
            query_gate = query_alpha * original_query + query_beta * aug_query

        # TODO: remove with_sharding_constraint
        copied_query = einops.repeat(query, "b c d -> (b k) c d", k=num_neighbors)
        # copied_query = with_sharding_constraint(copied_query, PS(("dp", "fsdp"), None, "mp"))
        copied_q_mask = einops.repeat(query_attention_mask, "b c -> (b k) c", k=num_neighbors)
        # copied_q_mask = with_sharding_constraint(copied_q_mask, PS(("dp", "fsdp"), None))
        nei = einops.rearrange(nei, 'b k r d -> (b k) r d')
        nei_w_query = self.nei_to_query_att(nei,
                                            key_value_states=copied_query,
                                            attention_mask=copied_q_mask
                                            )
        #print(f"{nei_w_query=}", flush=True)
        aug_nei = nei + nei_w_query[0]
        aug_nei = self.nei_lowrank_mlp(self.nei_lowrank_mlp_ln(aug_nei)) + aug_nei
        aug_nei = aug_nei.reshape(original_nei.shape)
        if self.nei_gate is not None:
            # at init nei_alpha should be at ~0.9 in first layer, 0.95 in second and 0.99 in third.
            nei_alpha, nei_beta = self.nei_gate(aug_nei)
        else:
            nei_alpha, nei_beta = 0, 1
        nei_gate = nei_alpha * original_nei + nei_beta * aug_nei
        if self.config.apply_query_to_nei_att:
            return query_gate, nei_gate
        else:
            return original_query, nei_gate


class GPTNeoXQueryAugmentor(nn.Module):

    def __init__(self, config, dtype):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.layers = [GPTNeoXQueryAugmentorLayer(self.config, dtype=self.dtype) for _ in
                       range(self.config.n_query_aug_layers)]

    def forward(self, query, query_attention_mask, nei, nei_attention_mask):
        original_nei = nei

        for i, layer in enumerate(self.layers):
            query, nei = layer(query,
                               query_attention_mask,
                               nei,
                               nei_attention_mask,
                               None,
                               None,
                               is_last=i == self.config.n_query_aug_layers - 1
                               )

        return nei


class GPTNeoXBlock(nn.Module):
    def __init__(self, config, dtype, has_cca):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.has_cca = has_cca

        self.use_parallel_residual = self.config.use_parallel_residual
        # TODO: input_layernorm input length and such
        self.input_layernorm = nn.LayerNorm(normalized_shape=self.config.hidden_size, eps=self.config.layer_norm_eps, dtype=self.dtype)
        self.attention = GPTNeoXAttention(self.config, dtype=self.dtype)
        self.post_attention_dropout = nn.Dropout(p=self.config.hidden_dropout)
        self.post_attention_layernorm = nn.LayerNorm(normalized_shape=self.config.hidden_size,
                                                     eps=self.config.layer_norm_eps, dtype=self.dtype)

        self.mlp = GPTNeoXMLP(self.config, dtype=self.dtype)
        self.post_mlp_dropout = nn.Dropout(p=self.config.hidden_dropout)
        if self.has_cca:
            self.cca = GPTNeoXChunkedCrossAttention(
                self.config,
                dtype=self.dtype,
            )
            self.cca_norm = nn.LayerNorm(
                normalized_shape=self.config.hidden_size,
                eps=self.config.layer_norm_eps,
                dtype=self.dtype,
            )
        else:
            self.cca = None
            self.cca_norm = None

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            position_ids=None,
            neighbor_hidden_states=None,
            neighbor_mask=None,
            nei_position_ids=None,
            att_scores=None,
            chunk_index=None,
            deterministic: bool = True,
            init_cache: bool = False,
            output_attentions: bool = False,
            cca_kwargs: Optional[Dict] = None,
            # use_cca_cache:bool=False,
    ):
        # hidden_states = with_sharding_constraint(hidden_states, PS(("dp", "fsdp"), None, "mp"))
        #print(f"{hidden_states=}", flush=True)
        if self.cca is not None and (neighbor_hidden_states is not None):
            cca_output = self.cca(hidden_states=self.cca_norm(hidden_states),
                                  neighbor_hidden_states=neighbor_hidden_states,
                                  neighbor_mask=neighbor_mask,
                                  nei_position_ids=nei_position_ids,
                                  att_scores=att_scores,
                                  position_ids=chunk_index,
                                  output_attentions=output_attentions,
                                  deterministic=deterministic,
                                  **(cca_kwargs if cca_kwargs is not None else dict())

                                  )
            cca_hidden_states = cca_output[0]
            if self.config.apply_tanh_in_cca:
                cca_hidden_states = (30 * torch.tanh(cca_hidden_states / 30)).type(cca_hidden_states.dtype)
            if not self.use_parallel_residual:
                hidden_states = cca_hidden_states + hidden_states
        else:
            cca_hidden_states = None

        att_input = self.input_layernorm(hidden_states).type(self.dtype)
        #print(f"{att_input=}", flush=True)
        # att_input = with_sharding_constraint(att_input, PS(("dp", "fsdp"), None, "mp"))
        # attention_mask = with_sharding_constraint(attention_mask, PS(("dp", "fsdp"), None))

        attn_outputs = self.attention(
            att_input,
            attention_mask,
            position_ids,
            deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]
        attn_output = self.post_attention_dropout(attn_output)  # TODO: Handle deterministic
        # attn_output = with_sharding_constraint(attn_output, PS(("dp", "fsdp"), None, "mp"))

        if self.use_parallel_residual:
            # pseudocode:
            # x = x + attn(ln1(x)) + mlp(ln2(x))
            mlp_output = self.mlp(self.post_attention_layernorm(hidden_states).type(self.dtype))
            # mlp_output = with_sharding_constraint(mlp_output, PS(("dp", "fsdp"), None, "mp"))
            mlp_output = self.post_mlp_dropout(mlp_output)  # TODO: Handle deterministic
            if cca_hidden_states is None:
                hidden_states = mlp_output + attn_output + hidden_states
            else:
                if self.config.log_debug_metrics and not self.is_initializing():
                    # TODO: Handle
                    cca_hidden_states_norm = torch.norm(cca_hidden_states, dim=-1).mean(where=attention_mask > 0)
                    att_hidden_states_norm = torch.norm(attn_output, dim=-1).mean(where=attention_mask > 0)
                    mlp_hidden_states_norm = torch.norm(mlp_output, dim=-1).mean(where=attention_mask > 0)
                    residual_hidden_states_norm = torch.norm(hidden_states, dim=-1).mean(where=attention_mask > 0)
                    self.sow('intermediates', f'cca_hidden_states_norm', cca_hidden_states_norm)
                    self.sow('intermediates', f'att_hidden_states_norm', att_hidden_states_norm)
                    self.sow('intermediates', f'mlp_hidden_states_norm', mlp_hidden_states_norm)
                    self.sow('intermediates', f'residual_hidden_states_norm', residual_hidden_states_norm)
                hidden_states = mlp_output + attn_output + cca_hidden_states + hidden_states
                if self.config.log_debug_metrics and not self.is_initializing():
                    added_hidden_states_norm = torch.norm(hidden_states, dim=-1).mean(where=attention_mask > 0)
                    self.sow('intermediates', f'added_hidden_states_norm', added_hidden_states_norm)
        else:
            # pseudocode:
            # x = x + attn(ln1(x))
            # x = x + mlp(ln2(x))

            attn_output = attn_output + hidden_states
            mlp_output = self.mlp(self.post_attention_layernorm(attn_output))
            mlp_output = self.post_mlp_dropout(mlp_output, deterministic=deterministic)
            hidden_states = mlp_output + attn_output
        # hidden_states = with_sharding_constraint(hidden_states, PS(("dp", "fsdp"), None, "mp"))
        return (hidden_states,) + attn_outputs[1:]


GPT_NEOX_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`~GPTNeoXConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

GPT_NEOX_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert *input_ids* indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
"""


def topk_chunks(retriever_scores, num_candidates, *, where=None):
    # TODO: @jax.vmap
    def _topk_chunks(retriever_scores):
        return (-retriever_scores).argsort()[:num_candidates]  # k = num_candidates

    if where is not None:
        retriever_scores[~where] = -torch.inf
    return _topk_chunks(retriever_scores)


def create_segment_mask(total_num_chunks, n_skip_chunks):
    # TODO: @jax.vmap
    def _create_segment_mask(chunk_index):
        max_chunk = n_skip_chunks * (chunk_index // n_skip_chunks)
        return torch.arange(total_num_chunks) < max_chunk - 2

    return _create_segment_mask(torch.arange(total_num_chunks))


# TODO: @jax.vmap
def _ranksigrelu(scores, mask, pair_mask, weights=None):
    score_i = scores.unsqueeze(-1)
    score_j = scores.unsqueeze(-1).unsqueeze(-1)
    score_pairs = torch.sigmoid((score_i - score_j))
    if weights is not None:
        score_pairs = weights.reshape(score_pairs.shape) * score_pairs

    pair_mask = pair_mask.reshape(score_pairs.shape)
    x = torch.sum(torch.maximum(score_pairs[pair_mask], torch.tensor(0)), dim=-1)
    x = x - torch.max(torch.maximum(x[mask], torch.tensor(0)), dim=-1)
    mask = mask.any(-1).unsqueeze(-1)
    x[~mask] = 0.0
    return x

def compute_pairs(a, op):
    """Computes pairs based on values of `a` and the given pairwise `op`.

  Args:
    a: The array used to form pairs. The last axis is used to form pairs.
    op: The binary op to map a pair of values to a single value.

  Returns:
    A :class:`jax.Array` with the same leading dimensions as `a`, but with the
    last dimension expanded so it includes all pairs `op(a[..., i], a[..., j])`
  """
    a_i = a.unsqueeze(-1)
    a_j = a.unsqueeze(-1).unsqueeze(-1)
    result_shape = torch.broadcast_shapes(a_i.shape, a_j.shape)
    result = torch.broadcast_to(op(a_i, a_j), result_shape)
    out_shape = tuple(result.shape[:-2]) + (result.shape[-2] * result.shape[-1],)
    return torch.reshape(result, out_shape)


def ranksigrelu(
        scores, mask, pair_mask=None, weights=None, margin=0, offset=1.0, substract_max=True,
):
    mask = mask.type(torch.bool)
    if pair_mask is None:
        pair_mask = compute_pairs(mask, torch.logical_and)
    if weights is None:
        weights = torch.ones_like(pair_mask)
    if substract_max:
        # TODO verify
        scores_max = torch.max(torch.maximum(scores[mask], torch.tensor(0)), dim=-1).unsqueeze(-1)
        scores = scores - scores_max
    scores = _ranksigrelu(scores, mask, pair_mask, weights)

    scores = scores + offset + margin
    scores = torch.nn.functional.softplus(scores)
    scores = torch.where(scores > 0, scores, 0)
    return scores


class GPTNeoXRetriever(nn.Module):

    def __init__(self, config, dtype):
        super().__init__()
        self.config = config
        self.dtype = dtype

        self.preret_bidir_attention = GPTNeoXCrossAttention(self.config, dtype=self.dtype)
        self.preret_bi_attention_norm = nn.LayerNorm(normalized_shape=self.config.hidden_size,
                                                     eps=self.config.layer_norm_eps, dtype=self.dtype)
        self.pre_key_norm = nn.LayerNorm(normalized_shape=self.config.hidden_size, eps=self.config.layer_norm_eps,
                                         dtype=self.dtype, )
        self.key_projection = nn.Linear(
            self.config.hidden_size,
            self.config.hidden_size,
            dtype=self.dtype,
            bias=True,
        )
        self.pre_query_norm = nn.LayerNorm(normalized_shape=self.config.hidden_size,
                                           eps=self.config.layer_norm_eps,
                                           dtype=self.dtype, )
        self.query_projection = nn.Linear(
            self.config.hidden_size,
            self.config.hidden_size,
            dtype=self.dtype,
            bias=True,
        )
        self.fill_value = self.config.retriever_fill_value
        self.n_skip_chunks = self.config.max_position_embeddings // self.config.chunk_size
        self.num_neighbors = self.config.num_neighbors
        self.threshold_nei_scores = self.config.threshold_nei_scores
        self.num_sequence_chunks = self.config.max_position_embeddings // self.config.chunk_size
        self.learned_margin = nn.Parameter(torch.Tensor([0]))

        self.scheduled_sampling_schedule_fn = None

    def compute_query_scores(self, encoded_output, n_skip_chunks, num_neighbors):
        query_based_scores = torch.einsum('qd,kd->qk', encoded_output.query_chunks,
                                          encoded_output.key_chunks)

        if n_skip_chunks > 0:
            chunk_mask = encoded_output.chunk_mask.to(encoded_output.device)
            segment_mask = create_segment_mask(query_based_scores.shape[0], n_skip_chunks)
            chunk_mask &= segment_mask
        else:
            chunk_mask = torch.ones_like(query_based_scores).type(torch.bool).to(encoded_output.device)
        query_score_based_idx = topk_chunks(query_based_scores, num_candidates=num_neighbors, where=chunk_mask)
        return query_based_scores, query_score_based_idx, chunk_mask

    def apply_scaling(self, scores, chunk_mask):
        scaled_scores = ranksigrelu(scores / self.config.score_temp,
                                    chunk_mask,
                                    margin=torch.nn.functional.softplus(self.learned_margin),
                                    offset=self.config.atsc_margin_min)

        return scaled_scores

    def batch_take_along_axis(self, scaled_scores, chunk_mask, idxs):
        idx_scores = torch.take_along_axis(scaled_scores, idxs, axis=-1)
        idx_mask = torch.take_along_axis(chunk_mask, idxs, axis=-1)
        return idx_scores, idx_mask

    def forward(
            self,
            encoded_output,
            append_next_chunk: Optional[bool] = None,
            n_skip_chunks: Optional[int] = None,
            num_neighbors: Optional[int] = None,
    ):
        if append_next_chunk is None:
            append_next_chunk = self.config.append_next_chunk
        if n_skip_chunks is None:
            n_skip_chunks = self.n_skip_chunks
        if num_neighbors is None:
            num_neighbors = self.num_neighbors

        query_based_scores, query_score_based_idx, chunk_mask = self.compute_query_scores(encoded_output, n_skip_chunks,
                                                                                          num_neighbors)
        scaled_scores = self.apply_scaling(query_based_scores, chunk_mask)
        query_att_scores, query_neighbor_mask = self.batch_take_along_axis(scaled_scores, chunk_mask,
                                                                           query_score_based_idx)

        self.sow('intermediates', f'query_att_scores', query_att_scores)
        top_nei_idx, nei_mask, att_scores = query_score_based_idx, query_neighbor_mask, query_att_scores
        aux_loss = None
        ret_metrics = {}

        att_scores[att_scores <= 0] = 0
        if self.config.stop_grad_trick:
            # TODO: 0 with att_scores dim
            att_scores = att_scores - att_scores

            # if self.config.log_debug_metrics and not self.is_initializing():
            #     self.sow('intermediates', f'nei_mask', nei_mask)
            #     self.sow('intermediates', f'pre_nei_mask', pre_nei_mask)
            #     self.sow('intermediates', f'neighbor_attention_mask', neighbor_attention_mask)
        neighbor_hidden_states, neighbor_mask, nei_position_ids, att_scores = self.select_nei(encoded_output,
                                                                                              top_nei_idx, nei_mask,
                                                                                              att_scores, num_neighbors,
                                                                                              append_next_chunk)

        loss_scale = None
        return GPTNeoXRetrieverNeighborOutput(aux_loss=aux_loss,
                                              neighbor_hidden_states=neighbor_hidden_states,
                                              loss_scale=loss_scale,
                                              neighbor_mask=neighbor_mask,
                                              retrieval_metrics=ret_metrics,
                                              att_scores=att_scores,
                                              encoded_output=encoded_output,
                                              nei_position_ids=nei_position_ids,
                                              )

    def select_nei(self, encoded_output, top_nei_idx, nei_mask, att_scores, num_neighbors, append_next_chunk):
        cand_hidden_states = encoded_output.encoded_hidden_states
        cand_attention_mask = encoded_output.attention_mask
        if not self.config.apply_refactor:
            chunk_size = self.config.chunk_size
            num_document_chunks = top_nei_idx.shape[0]
            shifted_hidden_states = torch.pad(cand_hidden_states[1:, ...], ((0, 1), (0, 0), (0, 0)))
            curr_neighbor_hidden_states = cand_hidden_states[top_nei_idx.reshape(-1)]
            next_neighbor_hidden_states = shifted_hidden_states[top_nei_idx.reshape(-1)]
            neighbor_hidden_states = torch.cat((curr_neighbor_hidden_states, next_neighbor_hidden_states),
                                                       dim=-2)
            neighbor_hidden_states = einops.rearrange(neighbor_hidden_states, '(b k) r d -> b k r d',
                                                      b=num_document_chunks)
            neighbor_mask = torch.broadcast_to(nei_mask.unsqueeze(-1), neighbor_hidden_states.shape[:-1])
            nei_position_ids = torch.arange(2 * chunk_size)
            nei_position_ids = torch.broadcast_to(nei_position_ids[None, :],
                                                  (num_document_chunks * num_neighbors, 2 * chunk_size))

        else:
            neighbor_hidden_states, neighbor_mask, nei_position_ids = new_lookup_neighbors(top_nei_idx,
                                                                                           cand_hidden_states,
                                                                                           cand_attention_mask,
                                                                                           append_next_chunk,
                                                                                           nei_mask=nei_mask)

            if self.config.ret_score_in_cca:
                att_scores = torch.broadcast_to(att_scores.unsqueeze(-1), neighbor_hidden_states.shape[:-1])
        return neighbor_hidden_states, neighbor_mask, nei_position_ids, att_scores

    def preret_encode(self,
                      hidden_states,
                      attention_mask,
                      deterministic,
                      pooling_size: int,
                      output_attentions: bool = False, ):
        original_hidden_states_shape = hidden_states.shape
        original_attention_mask_shape = attention_mask.shape
        # #print(hidden_states.shape)

        original_hidden_states = einops.rearrange(hidden_states, 'b (l c) ... -> (b l) c ... ', c=pooling_size)
        attention_mask = einops.rearrange(attention_mask, 'b (l c) ... -> (b l) c ... ', c=pooling_size)

        # add a chunk dimension
        # 1. apply bi-dir attention
        preret_bi_output = self.preret_bidir_attention(
            self.preret_bi_attention_norm(original_hidden_states),
            attention_mask=attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions)

        encoded_hidden_states = preret_bi_output[0] + original_hidden_states

        # 2. pool
        pooled_hidden_states = encoded_hidden_states.mean(dim=-2)

        # 3. project to query chunks and key chunks
        key_chunks = self.key_projection(self.pre_key_norm(pooled_hidden_states))
        query_chunks = self.query_projection(self.pre_query_norm(pooled_hidden_states))
        chunk_mask = attention_mask.type(torch.bool).any(-1)[..., None]
        if chunk_mask.shape[0] != pooled_hidden_states.shape[0]:
            chunk_mask = chunk_mask[:pooled_hidden_states.shape[0], ...]

        # nei_pos = jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0, a_max=2*self.config.chunk_size-1)
        original_hidden_states = original_hidden_states.reshape(original_hidden_states_shape)
        attention_mask = attention_mask.reshape(original_attention_mask_shape)

        key_chunks = key_chunks / torch.linalg.norm(key_chunks, dim=-1).unsqueeze(-1)
        query_chunks = query_chunks / torch.linalg.norm(query_chunks, dim=-1).unsqueeze(-1)

        return GPTNeoXRetrieverEncodedOutput(
            original_hidden_states=original_hidden_states,
            encoded_hidden_states=encoded_hidden_states,
            attention_mask=attention_mask,
            key_chunks=key_chunks,
            query_chunks=query_chunks,
            chunk_mask=chunk_mask,
            preret_attention=preret_bi_output[1:],
            # nei_position_ids=nei_pos,
        )


from transformers.modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithCrossAttentions


class GPTNeoXBlockCollection(nn.Module):
    def __init__(self, config, dtype):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.first_upcoder_layer = self.config.num_hidden_layers // 2
        self.lowcoder_layer_idxs = torch.arange(self.first_upcoder_layer)
        self.upcoder_layer_idxs = torch.arange(self.first_upcoder_layer, self.config.num_hidden_layers)
        if self.config.cca_freq > 0:
            self.cca_layer_idxs = torch.arange(self.first_upcoder_layer,
                                               self.config.num_hidden_layers,
                                               self.config.cca_freq)
        else:
            self.cca_layer_idxs = set()

        self.blocks = nn.ModuleList([
            GPTNeoXBlock(self.config, dtype=self.dtype, has_cca=i in list(self.cca_layer_idxs))
            for i in range(self.config.num_hidden_layers)
        ])

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            position_ids=None,
            neighbor_hidden_states=None,
            neighbor_mask=None,
            nei_position_ids=None,
            att_scores=None,
            chunk_index=None,
            deterministic: bool = True,
            init_cache: bool = False,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = True,
            mode: str = "all",
            cca_kwargs: Optional[Dict] = None,
            # use_cca_cache:bool=False
    ):
        if mode == "all":
            blocks = self.blocks
        elif mode == "lowcoder":
            blocks = [self.blocks[i] for i in self.lowcoder_layer_idxs]
        elif mode == "upcoder":
            blocks = [self.blocks[i] for i in self.upcoder_layer_idxs]
        else:
            raise ValueError(f"mode {mode} not recognized")

        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        all_cross_attentions = () if output_attentions else None

        for block in blocks:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = block(
                hidden_states,
                attention_mask,
                position_ids=position_ids,
                neighbor_hidden_states=neighbor_hidden_states,
                neighbor_mask=neighbor_mask,
                nei_position_ids=nei_position_ids,
                att_scores=att_scores,
                chunk_index=chunk_index,
                deterministic=deterministic,
                init_cache=init_cache,
                output_attentions=output_attentions,
                cca_kwargs=cca_kwargs,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1],)
                if block.has_cca:
                    all_cross_attentions += (layer_outputs[2],)

        if not return_dict:
            return (hidden_states,) + (all_hidden_states, all_attentions, all_cross_attentions)

        return BaseModelOutputWithCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


@add_start_docstrings(
    "The bare GPTNeoX Model transformer outputting raw hidden-states without any specific head on top.",
    GPT_NEOX_START_DOCSTRING,
)


class GPTNeoXPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GPTNeoXConfig
    base_model_prefix = "gpt_neox"

    supports_gradient_checkpointing = True
    _no_split_modules = ["GPTNeoXLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # Copied from transformers.models.gpt_neo.modeling_flax_gpt_neo.FlaxGPTNeoPreTrainedModel.__init__ with GPTNeo->GPTNeoX
    def __init__(
            self,
            config: GPTNeoXConfig,
            input_shape: Tuple = (1, 128),
            seed: int = 0,
            dtype: torch.dtype = torch.float32,
            _do_init: bool = True,
            **kwargs,
    ):
        super().__init__(config, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def sample(
            self,
            input_ids: torch.LongTensor,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            logits_warper: Optional[LogitsProcessorList] = None,
            max_length: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[Union[int, List[int]]] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_scores: Optional[bool] = None,
            output_logits: Optional[bool] = None,
            return_dict_in_generate: Optional[bool] = None,
            synced_gpus: bool = False,
            streamer: Optional["BaseStreamer"] = None,
            **model_kwargs,
    ) -> Union[Tuple[GenerateEncoderDecoderOutput, Any], Tuple[GenerateDecoderOnlyOutput, Any], Tuple[
        Union[torch.Tensor, torch.LongTensor], Any]]:
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_logits = output_logits if output_logits is not None else self.generation_config.output_logits
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        this_peer_finished = False  # used by synced_gpus only
        # auto-regressive generation
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                init_cache=True,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # sample
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

            # stop if we exceed the maximum length
            if stopping_criteria(input_ids, scores):
                this_peer_finished = True

            if this_peer_finished and not synced_gpus:
                break

        if streamer is not None:
            streamer.end()

        last_lowcoder_states = self.transformer.cached_array

        encoded_lowcoder_states = self.preret_forward(
            hidden_states=last_lowcoder_states,
            attention_mask=torch.ones(last_lowcoder_states.shape[:-1]))

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                ), encoded_lowcoder_states
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                ), encoded_lowcoder_states
        else:
            return input_ids, encoded_lowcoder_states

    def greedy_search(
            self,
            input_ids: torch.LongTensor,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            logits_warper: Optional[LogitsProcessorList] = None,
            max_length: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[Union[int, List[int]]] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_scores: Optional[bool] = None,
            output_logits: Optional[bool] = None,
            return_dict_in_generate: Optional[bool] = None,
            synced_gpus: bool = False,
            streamer: Optional["BaseStreamer"] = None,
            **model_kwargs,
    ) -> Union[Tuple[GenerateEncoderDecoderOutput, Any], Tuple[GenerateDecoderOnlyOutput, Any], Tuple[
        Union[torch.Tensor, torch.LongTensor], Any]]:
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_logits = output_logits if output_logits is not None else self.generation_config.output_logits
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        this_peer_finished = False  # used by synced_gpus only
        # auto-regressive generation
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                init_cache=True,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # sample
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.argmax(probs).squeeze(1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

            # stop if we exceed the maximum length
            if stopping_criteria(input_ids, scores):
                this_peer_finished = True

            if this_peer_finished and not synced_gpus:
                break

        if streamer is not None:
            streamer.end()

        last_lowcoder_states = self.transformer.cached_array

        encoded_lowcoder_states = self.preret_forward(
            hidden_states=last_lowcoder_states,
            attention_mask=torch.ones(last_lowcoder_states.shape[:-1]))

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                ), encoded_lowcoder_states
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                ), encoded_lowcoder_states
        else:
            return input_ids, encoded_lowcoder_states

    def single_lowcoder_forward(
            self,
            input_ids,
            attention_mask=None,
            position_ids=None,
            chunk_index=None,
            params: dict = None,
            train: bool = False,
            past_key_values: dict = None,
            output_attentions: bool = False,
            dropout_rng=None,
    ):

        apply_kwargs = self.create_apply_kwargs(params, dropout_rng, past_key_values)

        outputs = self.module.apply(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            deterministic=not train,
            output_attentions=output_attentions,
            method=self.module._lowcoder_forward,
            **apply_kwargs
        )
        return outputs

    # def batch_lowcoder_forward(self, input_ids, attention_mask, params):
    def batch_lowcoder_forward(self,
                               input_ids,
                               attention_mask=None,
                               position_ids=None,
                               chunk_index=None,
                               train: bool = False,
                               past_key_values: dict = None,
                               output_attentions: bool = False,
                               ):

        return self._lowcoder_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            deterministic=not train,
            output_attentions=output_attentions,
        )

    def preret_forward(
            self,
            hidden_states,
            attention_mask=None,
            params: dict = None,
            train: bool = False,
            output_attentions: bool = False,
            dropout_rng=None,
    ):

        apply_kwargs = self.create_apply_kwargs(params, dropout_rng)

        outputs = self.module.apply(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            deterministic=not train,
            output_attentions=output_attentions,
            method=self.module._encode_forward,
            **apply_kwargs
        )

        return outputs

    def create_apply_kwargs(self, params, dropout_rng, past_key_values=None):
        return {}



class GPTNeoXModel(GPTNeoXPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embed_in = nn.Embedding(config.vocab_size, config.hidden_size)
        self.emb_dropout = nn.Dropout(config.hidden_dropout)
        # self.layers = nn.ModuleList([GPTNeoXLayer(config) for _ in range(config.num_hidden_layers)])
        self.layers = GPTNeoXBlockCollection(self.config, dtype=self.dtype)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        if self.config.cca_freq > 0:
            self.retriever = GPTNeoXRetriever(self.config, dtype=self.dtype)
        else:
            self.retriever = None
        if self.config.n_query_aug_layers is not None and self.config.n_query_aug_layers > 0:
            self.query_augmentor = GPTNeoXQueryAugmentor(self.config, dtype=self.dtype)
        else:
            self.query_augmentor = None


    """
                input_ids,
            attention_mask,
            position_ids,
            deterministic=deterministic,
            output_attentions=output_attentions,"""
    def lowcoder(self,
                 input_ids: torch.Tensor,
                 attention_mask: torch.Tensor,
                 position_ids,
                 deterministic=True,
                 output_attentions: bool = False,
                 output_hidden_states: bool = False,
                 return_dict: bool = True,
                 # use_cca_cache:bool=False,
                 ):
        input_embeds = self.embed_in(input_ids.type(torch.int))

        hidden_states = self.emb_dropout(input_embeds)


        lowcoder_outputs = self.layers(
            hidden_states,
            attention_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            mode="lowcoder",
            # use_cca_cache=use_cca_cache,
        )
        return lowcoder_outputs

    @add_start_docstrings_to_model_forward(GPT_NEOX_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        real_checkpoint=_REAL_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
            self,
            input_ids,
            attention_mask,
            position_ids=None,
            deterministic=True,
            lowonly_input_ids=None,
            lowonly_attention_mask=None,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = True,
            upcoder_input=None,
            encoded_neighbors: Optional[EncodedNeighbors] = None,
            # use_cca_cache:bool=False,
            chunk_index: Optional[torch.Tensor] = None,
            mode: str = "all",
            retrieval_kwargs: Optional[dict] = None,
    ) -> Union[Tuple, GPTNeoXModelOutput]:
        lowcoder_outputs = None
        retriever_output = None
        neighbor_hidden_states = None
        neighbor_mask = None
        nei_position_ids = None
        retriever_input = None
        att_scores = None
        cca_kwargs = None

        retrieval_kwargs = dict()
        if position_ids is None:
            position_ids = torch.broadcast_to(
                torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0),
                input_ids.shape
            )

        if lowonly_input_ids is not None:
            # if we have lowonly_input_ids this means we are in SLED/FID mode
            # and we want to encode those inputs first so we could retrieve and attend to them.
            lo_encoded_output = self(lowonly_input_ids,
                                     lowonly_attention_mask,
                                     deterministic=deterministic,
                                     retrieval_kwargs=dict(pooling_size=lowonly_input_ids.shape[-1]),
                                     mode="encoded_output",
                                     )
            assert self.config.pooling_size > 0
            retrieval_kwargs = dict(pooling_size=self.config.pooling_size,
                                    append_next_chunk=False,
                                    n_skip_chunks=0,
                                    num_neighbors=retrieval_kwargs.get("num_neighbors", None),
                                    )
            cca_kwargs = dict(chunk_size=self.config.pooling_size, n_chunks_per_window=1)
        else:
            lo_encoded_output = None

        input_embeds = self.embed_in(input_ids.type(torch.int))
        hidden_states = self.emb_dropout(input_embeds, deterministic=deterministic)

        lowcoder_outputs = self.layers(
                hidden_states,
                attention_mask,
                position_ids,
                deterministic=deterministic,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mode="lowcoder",
            )

        hidden_states = lowcoder_outputs.last_hidden_state if  return_dict else lowcoder_outputs[0]

        retriever_input = hidden_states
        if self.retriever is not None and torch.prod(hidden_states.shape[:-1])>1:
            if encoded_neighbors is not None:
                neighbor_hidden_states = encoded_neighbors.neighbor_hidden_states
                neighbor_mask = encoded_neighbors.neighbor_mask
                chunk_index = encoded_neighbors.chunk_index
                att_scores = encoded_neighbors.att_scores
                nei_position_ids = encoded_neighbors.nei_position_ids
            else:
                encoded_output = self.retriever.preret_encode(
                        hidden_states,
                        attention_mask,
                        deterministic,
                        retrieval_kwargs.get("pooling_size", self.config.chunk_size),
                        output_attentions,
                        )
                if mode=="encoded_output":
                    return encoded_output
                if lo_encoded_output is not None:
                    encoded_output = GPTNeoXRetrieverEncodedOutput(
                                            original_hidden_states=None,
                                            encoded_hidden_states=lo_encoded_output.encoded_hidden_states,
                                            attention_mask=lo_encoded_output.attention_mask,
                                            key_chunks=lo_encoded_output.key_chunks,
                                            query_chunks=encoded_output.query_chunks,
                                            chunk_mask=None,
                                            preret_attention=None)

                retriever_output = self.retriever(encoded_output,
                                                n_skip_chunks=retrieval_kwargs.get("n_skip_chunks", None),
                                                append_next_chunk=retrieval_kwargs.get("append_next_chunk", None),
                                                num_neighbors=retrieval_kwargs.get("num_neighbors", None),
                                                )
                neighbor_hidden_states = retriever_output.neighbor_hidden_states
                neighbor_mask = retriever_output.neighbor_mask
                att_scores = retriever_output.att_scores
                nei_position_ids = retriever_output.nei_position_ids

        if self.query_augmentor is not None and encoded_output is not None:
            # TODO: add nei_position_ids to this.
            neighbor_hidden_states = self.query_augmentor(encoded_output.encoded_hidden_states,
                                encoded_output.attention_mask,
                                neighbor_hidden_states,
                                neighbor_mask,
                                 )
        upcoder_outputs = self.layers(hidden_states, attention_mask, position_ids, neighbor_hidden_states,
                                      neighbor_mask, nei_position_ids, att_scores,
                                    deterministic=deterministic,
                                    output_attentions=output_attentions,
                                    output_hidden_states=output_hidden_states,
                                    return_dict=return_dict,
                                    chunk_index=chunk_index,
                                    cca_kwargs=cca_kwargs,
                                    mode="upcoder")

        hidden_states = upcoder_outputs.last_hidden_state if return_dict else upcoder_outputs[0]
        hidden_states = self.final_layer_norm(hidden_states)

        if not return_dict:
            return (hidden_states,) + upcoder_outputs + lowcoder_outputs

        return GPTNeoXModelOutput(
            last_hidden_state=upcoder_outputs.last_hidden_state,
            upcoder_hidden_states=upcoder_outputs.hidden_states,
            upcoder_attentions=upcoder_outputs.attentions,
            cross_attentions=None,
            lowcoder_last_hidden_state=lowcoder_outputs.last_hidden_state if lowcoder_outputs is not None else None,
            lowcoder_hidden_states=lowcoder_outputs.hidden_states if lowcoder_outputs is not None else None,
            lowcoder_attentions=lowcoder_outputs.attentions if lowcoder_outputs is not None else None,
            retriever_output=retriever_output,
            retriever_input=retriever_input,
        )

@add_start_docstrings(
    """GPTNeoX Model with a `language modeling` head on top for CLM fine-tuning.""", GPT_NEOX_START_DOCSTRING
)

class GPTNeoXForCausalLMModule(GPTNeoXPreTrainedModel):
    _tied_weights_keys = ["embed_out.weight"]

    def __init__(self, config):
        super().__init__(config)

        self.gpt_neox = GPTNeoXModel(config)
        self.embed_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        prefix_tokenizer = config.get_tokenizer(truncation_side='left', padding_side='left')
        self.prepare_inputs = create_prepare_inputs(prefix_tokenizer, input_length=2048) # TODO: 2048 ?

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(GPT_NEOX_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional tensors are
            only required when the model is used as a decoder in a Sequence to Sequence model.

            Contains pre-computed hidden-states (key and values in the self-attention blocks that can be used (see
            `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
            ignored (masked), the loss is only computed for the tokens with labels n `[0, ..., config.vocab_size]`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, GPTNeoXForCausalLM, GPTNeoXConfig
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        >>> config = GPTNeoXConfig.from_pretrained("EleutherAI/gpt-neox-20b")
        >>> config.is_decoder = True
        >>> model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b", config=config)

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> prediction_logits = outputs.logits
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.gpt_neox(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        lm_logits = self.embed_out(hidden_states)

        lm_loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shift_logits = lm_logits[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithPast(
            loss=lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        input_shape = input_ids.shape
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
                "use_cache": kwargs.get("use_cache"),
            }
        )

        return model_inputs

    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
                + layer_past[2:],
            )
        return reordered_past


    def _lowcoder_forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, position_ids, deterministic, output_attentions):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if position_ids is None:
            position_ids = torch.broadcast_to(
                torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0),
                input_ids.shape
            )
        lowcoder_outputs = self.gpt_neox.lowcoder(
            input_ids,
            attention_mask,
            position_ids,
            deterministic=deterministic,
            output_attentions=output_attentions,
        )

        outputs = self.gpt_neox.retriever.preret_encode(
            lowcoder_outputs.last_hidden_state,
            attention_mask,
            deterministic,
            input_ids.shape[-1],  # note this assumes we are chunked
            output_attentions,
        )
        return outputs

    def forward(
            self,
            input_ids,
            attention_mask=None,
            position_ids=None,
            deterministic: bool = True,
            lowonly_input_ids=None,
            lowonly_attention_mask=None,
            init_cache: bool = False,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = True,
            upcoder_input=None,
            encoded_neighbors: Optional[EncodedNeighbors] = None,
            chunk_index: Optional[torch.Tensor] = None,
            retrieval_kwargs: Optional[dict] = None,
    ):

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            assert len(input_ids.shape) == len(
                attention_mask.shape), "input_ids and attention_mask must have the same number of dimensions"

        if position_ids is None:
            position_ids = torch.broadcast_to(
                torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0),
                input_ids.shape
            )
        else:
            assert len(input_ids.shape) == len(
                position_ids.shape), "input_ids and position_ids must have the same number of dimensions"
        should_squeeze = False
        if len(input_ids.shape) == 2:
            input_ids = input_ids[None, ...]
            attention_mask = attention_mask[None, ...]
            position_ids = position_ids[None, ...]
            should_squeeze = True

        transformer_input = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            upcoder_input=upcoder_input,
            encoded_neighbors=encoded_neighbors,
            chunk_index=chunk_index,
            lowonly_input_ids=lowonly_input_ids,
            lowonly_attention_mask=lowonly_attention_mask,
            retrieval_kwargs=retrieval_kwargs,
        )

        def transformer(**kwargs):
            return self.gpt_neox(
                deterministic=deterministic,
                init_cache=init_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        # if jax.process_count() > 1 and not self.is_initializing():
        #     transformer = jax.vmap(transformer)
        #     outputs = transformer(**transformer_input)
        #
        # else:
        # transformer_input = jax.tree_map(add_process_dim, transformer_input)
        outputs = transformer(**transformer_input)
        # outputs = jax.tree_map(remove_process_dim, outputs)

        hidden_states = outputs.last_hidden_state if return_dict else outputs[0]

        lm_logits = self.embed_out(hidden_states)

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
        else:
            output = GPTNeoXLMOutput(
                logits=lm_logits,
                upcoder_hidden_states=outputs.upcoder_hidden_states,
                upcoder_attentions=outputs.upcoder_attentions,
                cross_attentions=outputs.cross_attentions,
                lowcoder_last_hidden_state=outputs.lowcoder_last_hidden_state,
                lowcoder_hidden_states=outputs.lowcoder_hidden_states,
                lowcoder_attentions=outputs.lowcoder_attentions,
                retriever_output=outputs.retriever_output,
                retriever_input=outputs.retriever_input,
            )
        if should_squeeze:
            output = output.squeeze(0)
        return output

    def encode(self, text, batch_size=64, split_by_newline=False):
        inputs = [self.prepare_inputs(prefix, split_by_newline) for prefix in text]
        inputs = [add_batch_index(x,j) for j,x in enumerate(inputs)]
        inputs = sum(inputs,[])

        def collate_fn(batch):
            with torch.no_grad():
                input_ids = torch.tensor(np.array([x["input_ids"].squeeze() for x in batch]), device=self.device)
                attention_mask = torch.tensor(np.array([x["attention_mask"].squeeze() for x in batch]), device=self.device)
                return input_ids, attention_mask

        with torch.no_grad():
            outputs = []
            for i, batch in enumerate(chunked(inputs, batch_size)):
                print(f'{batch[-1]["batch_index"]+1}/{len(inputs)}')
                input_ids, attention_mask = collate_fn(batch)
                encoded_output = self.batch_lowcoder_forward(input_ids, attention_mask)
                for key in encoded_output['key_chunks']:
                    outputs.append(key)

        return torch.stack(outputs)


class GPTNeoXForCausalLM(GPTNeoXForCausalLMModule):

    def prepare_inputs_for_generation(self,
                                      input_ids,
                                      max_length,
                                      attention_mask: Optional[torch.Tensor] = None,
                                      past_key_values=None,
                                      **kwargs,
                                      ):
        # initializing the cache
        batch_size, seq_length = input_ids.shape
        if past_key_values is None:
            past_key_values = self.init_cache(batch_size, max_length)

        # Note that usually one would have to put 0's in the attention_mask for x > input_ids.shape[-1] and x < cache_length.
        # But since GPTNeoX uses a causal mask, those positions are masked anyways.
        # Thus we can create a single static attention_mask here, which is more efficient for compilation
        extended_attention_mask = torch.ones((batch_size, max_length), dtype=torch.int32)
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(dim=-1) - 1
            extended_attention_mask = assign_slice(extended_attention_mask, attention_mask, (0, 0))
        else:
            position_ids = torch.broadcast_to(torch.arange(seq_length, dtype=torch.int32)[None, :],
                                              (batch_size, seq_length))
        chunk_index = torch.full([batch_size, 1], fill_value=63, dtype=torch.int32)
        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
            "chunk_index": chunk_index,
            **kwargs
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        if "chunk_index" in model_kwargs:
            model_kwargs["chunk_index"] = torch.clip(model_kwargs["chunk_index"] + 1,
                                                     max=2 * (self.config.chunk_size - 1))  # will need to modify later
        if "encoded_neighbors" in model_kwargs:
            encoded_neighbors = model_kwargs["encoded_neighbors"]
            if encoded_neighbors is not None:
                model_kwargs["encoded_neighbors"] = EncodedNeighbors(
                    neighbor_hidden_states=encoded_neighbors.neighbor_hidden_states[-1:, ...],  # assumes bs=1
                    neighbor_mask=encoded_neighbors.neighbor_mask[-1:, ...],  # assumes bs=1
                    nei_position_ids=encoded_neighbors.nei_position_ids[-1:, ...],  # assumes bs=1
                )

        return model_kwargs
