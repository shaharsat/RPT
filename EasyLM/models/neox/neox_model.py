# coding=utf-8
# Copyright 2023 The EleutherAI and The HuggingFace Inc. team.
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
# coding=utf-8
# Copyright 2022 EleutherAI and The HuggingFace Inc. team. All rights reserved.
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
""" GPTNeoX model configuration"""
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

import einops
logger = logging.get_logger(__name__)

GPT_NEOX_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "EleutherAI/gpt-neox-20b": "https://huggingface.co/EleutherAI/gpt-neox-20b/resolve/main/config.json",
    # See all GPTNeoX models at https://huggingface.co/models?filter=gpt_neox
}
import jax.numpy as jnp
import gin


""" Flax GPT NeoX model."""

from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax

from transformers.modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput
from transformers.modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring
# from transformers.models.gpt_neox.configuration_gpt_neox import GPTNeoXConfig
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
import gin

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "EleutherAI/gpt-neox-20b"
_CONFIG_FOR_DOC = "GPTNeoXConfig"

from jax.sharding import PartitionSpec as PS
from jax.sharding import PartitionSpec as P
from collections import namedtuple
import optax
import rax
import flax
import operator
from typing import Any, Callable, Optional, Sequence, TypeVar, Dict
from transformers.utils import ModelOutput
from transformers import AutoTokenizer
import copy
from EasyLM.jax_utils import (
    with_sharding_constraint, get_jax_mesh, get_gradient_checkpoint_policy,put_along_zeroth_axis, create_target_scores,add_process_dim,remove_process_dim
)
from functools import partial
from transformers import GenerationConfig, FlaxLogitsProcessorList
@flax.struct.dataclass
class GreedyState:
	cur_len: jnp.ndarray
	sequences: jnp.ndarray
	running_token: jnp.ndarray
	is_sent_finished: jnp.ndarray
	model_kwargs: Dict[str, jnp.ndarray]

@flax.struct.dataclass
class FlaxGreedySearchOutput(ModelOutput):
	sequences: jnp.ndarray = None
 
from flax.linen import partitioning as nn_partitioning
from mlxu import function_args_to_config, load_pickle, open_file
RetrieverSupervision = namedtuple('RetrieverSupervision', ['nei_scores', 'nei_idx'])    
from EasyLM.models.neox.rpt_utils import EncodedNeighbors, new_lookup_neighbors
remat = nn_partitioning.remat
from ml_collections import ConfigDict
from EasyLM.models.neox.attention import my_dot_product_attention_weights
from EasyLM.models.neox.gate import GriffinGate

@gin.configurable
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
        cca_freq:int = 0,
        retriever_fill_value: float = -10000.0,
        threshold_nei_scores: float = 0.0,
        null_attn_init:float = 0.0001,
        num_neighbors:int = 2,
        chunk_size:int = 64,
        ss_schedule_steps:int = 450000,
        scheduled_sampling_max_prob:float = 1.0,
        scheduled_sampling_min_prob:float = 0.01,
        atsc_margin_min:float = 1.0,
        num_scored_neighbors:int = 20,
        score_temp:float = 0.05,
        aux_scale:float = 1.0,
        document_length:int = 16384,
        cca_layernorm_init_scale:float =  0.0001,
        set_pt_params:bool = False,
        qk_layernorm:bool = False,
        null_k_init_mult:float = 1.0,
        mult_by_ndcg: bool = True,
        ret_score_in_cca:bool = True,
        remat_attention:str = "",
        apply_refactor:bool = False,
        n_query_aug_layers:Optional[int] = None,
        query_aug_dim: int = 128,
        gate_num_blocks: int = 8,
        lowres_ss: bool = False,
        apply_gating:bool = True,
        gate_refactor:bool = False,
        do_last_gate:bool = False,
        a_init_query:Optional[float] = None,
        a_init_nei:Optional[float] = None,
        tanh_xatt:bool = False,
        tanh_causal_att:bool = False,
        apply_query_to_nei_att:bool = False,
        append_next_chunk:bool = True,
        pooling_size:Optional[int]=-1,
        log_debug_metrics:bool = False,
        stop_grad_trick:bool = False,
        apply_tanh_in_cca:bool = False,
        use_allowed_tar_mask:bool = False,
        **kwargs,
    ):
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.use_allowed_tar_mask = use_allowed_tar_mask
        self.apply_tanh_in_cca = apply_tanh_in_cca
        self.stop_grad_trick = stop_grad_trick
        self.log_debug_metrics=log_debug_metrics
        self.append_next_chunk = append_next_chunk
        self.pooling_size=pooling_size
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
    def get_default_config(cls, updates=None):
        config = function_args_to_config(cls.__init__)

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        

        return config
    
    @classmethod
    def get_jax_mesh(self, mesh_dim):
        return get_jax_mesh(mesh_dim, ('dp', 'fsdp', 'mp'))
    @classmethod
    def get_tokenizer(cls,**kwargs):
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b",
                            pad_token='<|endoftext|>',
                            mask_token='<|endoftext|>',
                            **kwargs)
        return tokenizer

    @staticmethod
    def get_partition_rules():
        """ Parition rules for GPTJ. Note that these rules are orderd, so that
            the beginning rules match first. It is important to use
            PartitionSpec() instead of None here because JAX does not treat
            None as a pytree leaf.
        """
        return (
            # embeddings
            ("embed_in/embedding", PS("mp", "fsdp")),
            # atention
            ("(wq|wk|wv)/kernel", PS("fsdp", "mp")),
            ("wo/kernel", PS("mp", "fsdp")),
            ("attention/query_key_value/kernel", PS("fsdp","mp")),
            ("attention/dense/kernel", PS("mp", "fsdp")),
            # mlp
            ("dense_h_to_4h/kernel", PS("fsdp", "mp")),
            ("down_proj/kernel", PS("fsdp", "mp")),
            ("up_proj/kernel", PS("mp", "fsdp")),
            ("dense_4h_to_h/kernel", PS("mp", "fsdp")),
            ("query_projection/kernel",PS("fsdp", "mp")),
            ("key_projection/kernel",PS("fsdp", "mp")),
            ("proj/w",PS(None,("fsdp", "mp"), None)),
            ("a_param_proj/w",PS(None,("fsdp", "mp"), None)),
            # output head
            ("embed_out/kernel", PS("fsdp", "mp")),
            ('.*', PS(None)),
        )
        # return (
        #     # embeddings
        #     ("embed_in/embedding", PS("fsdp", "mp")),
        #     # atention
        #     ("(wq|wk|wv)/kernel", PS("mp","fsdp")),
        #     ("wo/kernel", PS("fsdp", "mp")),
        #     ("attention/query_key_value/kernel", PS("mp","fsdp")),
        #     ("attention/dense/kernel", PS("fsdp", "mp")),
        #     # mlp
        #     ("mlp/dense_h_to_4h/kernel", PS("mp","fsdp")),
        #     ("mlp/dense_4h_to_h/kernel", PS("fsdp", "mp")),
        #     ("query_projection/kernel",PS("mp","fsdp")),
        #     ("key_projection/kernel",PS("mp","fsdp")),
        #     # output head
        #     ("embed_out/kernel", PS("mp","fsdp")),
        #     ('.*', PS(None)),
        # )

    @staticmethod
    def get_weight_decay_exclusions():
        return tuple()

    @staticmethod
    def rng_keys():
        return ('params', 'dropout', 'fcm')


    @classmethod
    def load_config(cls, path):
        load_type, load_path = path.split('::', 1)
        if load_type == 'pickle':
            return cls.from_dict(load_pickle(load_path)['neox_config'])
        elif load_type == 'json':
            with open_file(load_path, 'r') as fin:
                raw_config = fin.read()
            return cls.from_dict(json.loads(raw_config))
        elif load_type=="hf":
            return cls.from_pretrained(load_path)
        else:
            raise ValueError(f'Unsupported load config type: {load_type}')


import json


@flax.struct.dataclass
class FlaxBaseModelOutputCrossAttentions(ModelOutput):
    last_hidden_state: jnp.ndarray = None
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    attentions: Optional[Tuple[jnp.ndarray]] = None
    cross_attentions: Optional[Tuple[jnp.ndarray]] = None
    

    
@flax.struct.dataclass
class FlaxGPTNeoXRetrieverEncodedOutput(ModelOutput):
    original_hidden_states: jnp.ndarray = None
    encoded_hidden_states: jnp.ndarray = None
    attention_mask: jnp.ndarray = None
    key_chunks: jnp.ndarray = None
    query_chunks: jnp.ndarray = None
    chunk_mask: jnp.ndarray = None
    preret_attention: Optional[jnp.ndarray] = None
    
@flax.struct.dataclass
class FlaxGPTNeoXRetrieverNeighborOutput(ModelOutput):
    aux_loss: jnp.ndarray = None
    loss_scale: jnp.ndarray = None
    neighbor_hidden_states: jnp.ndarray = None
    neighbor_mask: jnp.ndarray = None
    retrieval_metrics: Optional[Dict[str, jnp.ndarray]] = None
    att_scores: jnp.ndarray = None
    encoded_output: Optional[FlaxGPTNeoXRetrieverEncodedOutput] = None
    nei_position_ids: Optional[jnp.ndarray] = None
    
@flax.struct.dataclass
class FlaxGPTNeoXRetrieverLossOutput(ModelOutput):
    aux_loss: jnp.ndarray = None
    target_neighbor_mask: jnp.ndarray = None
    target_score_based_idx: jnp.ndarray = None
    ret_metrics: Optional[Dict[str, jnp.ndarray]] = None
    
@flax.struct.dataclass
class FlaxGPTNeoXLowcoderRetrieverEncodedOutput(ModelOutput):
    hidden_states: jnp.ndarray = None
    attention_mask: jnp.ndarray = None
    neighbor_hidden_states: jnp.ndarray = None
    neighbor_mask: jnp.ndarray = None

@flax.struct.dataclass
class FlaxGPTNeoXModelOutput(ModelOutput):
    last_hidden_state: jnp.ndarray = None
    past_key_values: Optional[Tuple[Tuple[jnp.ndarray]]] = None
    upcoder_hidden_states: Optional[Tuple[jnp.ndarray]] = None
    upcoder_attentions: Optional[Tuple[jnp.ndarray]] = None
    cross_attentions: Optional[Tuple[jnp.ndarray]] = None
    lowcoder_last_hidden_state: Optional[jnp.ndarray] = None
    lowcoder_hidden_states: Optional[Tuple[jnp.ndarray]] = None
    lowcoder_attentions: Optional[Tuple[jnp.ndarray]] = None
    retriever_output: FlaxGPTNeoXRetrieverNeighborOutput = None
    retriever_input: Optional[jnp.ndarray] = None
    
@flax.struct.dataclass
class FlaxGPTNeoXLMOutput(ModelOutput):
    logits: jnp.ndarray = None
    past_key_values: Optional[Tuple[Tuple[jnp.ndarray]]] = None
    upcoder_hidden_states: Optional[Tuple[jnp.ndarray]] = None
    upcoder_attentions: Optional[Tuple[jnp.ndarray]] = None
    cross_attentions: Optional[Tuple[jnp.ndarray]] = None
    lowcoder_last_hidden_state: Optional[jnp.ndarray] = None
    lowcoder_hidden_states: Optional[Tuple[jnp.ndarray]] = None
    lowcoder_attentions: Optional[Tuple[jnp.ndarray]] = None
    retriever_output: FlaxGPTNeoXRetrieverNeighborOutput = None
    retriever_input: Optional[jnp.ndarray] = None

GPT_NEOX_START_DOCSTRING = r"""
    This model inherits from [`FlaxPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a Flax nn
    [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a
    regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.

    Finally, this model supports inherent JAX features such as:

    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

    Parameters:
        config ([`GPTNeoXConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~FlaxPreTrainedModel.from_pretrained`] method to load the model weights.
        dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):
            The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and
            `jax.numpy.bfloat16` (on TPUs).

            This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
            specified all the computation will be performed with the given `dtype`.

            **Note that this only specifies the dtype of the computation and does not influence the dtype of model
            parameters.**

            If you wish to change the dtype of the model parameters, see [`~FlaxPreTrainedModel.to_fp16`] and
            [`~FlaxPreTrainedModel.to_bf16`].
"""

GPT_NEOX_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`numpy.ndarray` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length`. Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.
        past_key_values (`Dict[str, np.ndarray]`, *optional*, returned by `init_cache` or when passing previous `past_key_values`):
            Dictionary of pre-computed hidden-states (key and values in the attention blocks) that can be used for fast
            auto-regressive decoding. Pre-computed key and value hidden-states are of shape *[batch_size, max_length]*.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


def rotate_half(hidden_states):
    first_half = hidden_states[..., : hidden_states.shape[-1] // 2]
    second_half = hidden_states[..., hidden_states.shape[-1] // 2 :]
    return jnp.concatenate((-second_half, first_half), axis=-1)


class FlaxGPTNeoXRotaryEmbedding(nn.Module):
    dim: int
    max_position_embeddings: int
    base: int = 10000
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        fraction = jnp.arange(0, self.dim, 2).astype(jnp.float32) / self.dim
        
        self.inv_freq =( 1.0 / (self.base ** (fraction))).astype(jnp.float32)
        self.cos_cached, self.sin_cached = self._compute_cos_sin(self.max_position_embeddings)

    def _get_cos_sin_cache(self, seq_len):
        if seq_len > self.max_position_embeddings:
            return self._compute_cos_sin(seq_len)
        else:
            return self.cos_cached, self.sin_cached

    def _compute_cos_sin(self, seq_len):
        t = jnp.arange(seq_len, dtype=jnp.float32)
        freqs = jnp.einsum(
            "i,j->ij",
            t,
            self.inv_freq,
            precision=jax.lax.Precision.HIGHEST,
        )
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        cos = jnp.expand_dims(jnp.expand_dims(jnp.cos(emb), 0), 0)
        sin = jnp.expand_dims(jnp.expand_dims(jnp.sin(emb), 0), 0)
        return cos, sin

    def __call__(self, seq_len=None):
        cos_cached, sin_cached = self._get_cos_sin_cache(seq_len)
        return cos_cached[:seq_len, ...].astype(jnp.float32), sin_cached[:seq_len, ...].astype(jnp.float32)


class FlaxGPTNeoXLinearScalingRotaryEmbedding(FlaxGPTNeoXRotaryEmbedding):
    """FlaxGPTNeoXRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    scaling_factor: float = 1.0

    def _compute_cos_sin(self, seq_len):
        t = jnp.arange(seq_len, dtype=jnp.float32)
        t = t / self.scaling_factor
        freqs = jnp.einsum(
            "i,j->ij",
            t,
            self.inv_freq,
            precision=jax.lax.Precision.HIGHEST,
        )
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        cos = jnp.expand_dims(jnp.expand_dims(jnp.cos(emb), 0), 0)
        sin = jnp.expand_dims(jnp.expand_dims(jnp.sin(emb), 0), 0)
        return cos.astype(jnp.float32), sin.astype(jnp.float32)


class FlaxGPTNeoXDynamicNTKScalingRotaryEmbedding(FlaxGPTNeoXRotaryEmbedding):
    """FlaxGPTNeoXRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    scaling_factor: float = 1.0

    def _compute_cos_sin(self, seq_len):
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim))
        else:
            inv_freq = self.inv_freq.astype(jnp.float32)

        t = jnp.arange(seq_len, dtype=jnp.float32)

        freqs = jnp.einsum(
            "i,j->ij",
            t,
            inv_freq,
            precision=jax.lax.Precision.HIGHEST,
        )
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        cos = jnp.expand_dims(jnp.expand_dims(jnp.cos(emb), 0), 0)
        sin = jnp.expand_dims(jnp.expand_dims(jnp.sin(emb), 0), 0)
        return cos.astype(jnp.float32), sin.astype(jnp.float32)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    gather_indices = position_ids[:, :, None, None].astype(int)  # [bs, seq_len, 1, 1]
    gather_indices = jnp.repeat(gather_indices, cos.shape[1], axis=1)
    gather_indices = jnp.repeat(gather_indices, cos.shape[3], axis=3)
    cos = jnp.take_along_axis(cos.repeat(gather_indices.shape[0], axis=0), gather_indices, axis=2)
    sin = jnp.take_along_axis(sin.repeat(gather_indices.shape[0], axis=0), gather_indices, axis=2)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed



from EasyLM.memory_efficient_attention import dot_product_attention_multihead
class FlaxGPTNeoXAttention(nn.Module):
    config: GPTNeoXConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    def setup(self):
        config = self.config
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_attention_heads
        self.rotary_ndims = int(self.head_size * config.rotary_pct)
        self.norm_factor = jnp.sqrt(self.head_size)
        self.query_key_value = nn.Dense(
            3 * config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        self.dense = nn.Dense(
            config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

        if config.rope_scaling is None:
            max_seq_length = config.max_position_embeddings
        else:
            max_seq_length = int(config.max_position_embeddings * config.rope_scaling["factor"])

        self.causal_mask = make_causal_mask(jnp.ones((1, max_seq_length), dtype="bool"), dtype="bool")
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = FlaxGPTNeoXRotaryEmbedding(
                self.rotary_ndims, self.config.max_position_embeddings, base=self.config.rotary_emb_base,dtype=jnp.float32
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = FlaxGPTNeoXLinearScalingRotaryEmbedding(
                    self.rotary_ndims,
                    self.config.max_position_embeddings,
                    base=self.config.rotary_emb_base,
                    scaling_factor=scaling_factor,
                    dtype=jnp.float32,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = FlaxGPTNeoXDynamicNTKScalingRotaryEmbedding(
                    self.rotary_ndims,
                    self.config.max_position_embeddings,
                    base=self.config.rotary_emb_base,
                    scaling_factor=scaling_factor,
                    dtype=jnp.float32,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")
    # @partial(
    #     nn.vmap,
    #     in_axes=(0, 0, 0, 0),
    #     out_axes=0,   
    #     variable_axes={'params': None, 'intermediates': 0},
    #     split_rngs={'dropout': True, "params": False},
    # )
    @nn.compact
    # Copied from transformers.models.gpt_neo.modeling_flax_gpt_neo.FlaxGPTNeoSelfAttention._concatenate_to_cache
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slighly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # detect if we're initializing by absence of existing cache data.
        is_initialized = self.has_variable("cache", "cached_key")
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # update key, value caches with our new 1d spatial slices
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            cached_key.value = key
            cached_value.value = value
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # causal mask for cached decoder self-attention: our single query position should only attend to those key positions that have already been generated and cached, not the remaining zero elements.
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            attention_mask = combine_masks(pad_mask, attention_mask)
        return key, value, attention_mask

    def _split_heads(self, hidden_states):
        coeff = hidden_states.shape[-1]//self.hidden_size
        
        return hidden_states.reshape(hidden_states.shape[:-1] + (self.num_attention_heads, self.head_size * coeff))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.hidden_size,))
    @partial(
        nn.vmap,
        in_axes=(0, 0, 0, None,None,None),
        out_axes=0,   
        variable_axes={'params': None, 'intermediates': 0, "cache" : 0},
        split_rngs={'dropout': True, "params": False},
    )
    def __call__(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
    ):
        # qkv = self.query_key_value(hidden_states)

        # proj q, k, v
        jax.debug.print('1: hidden_states={x}', x=hidden_states)
        fused_qkv = self.query_key_value(hidden_states)
        jax.debug.print('2: fused_qkv={x}', x=fused_qkv)
        batch, seq_len, _ = fused_qkv.shape
        fused_qkv = self._split_heads(fused_qkv)
        jax.debug.print('3: fused_qkv={x}', x=fused_qkv)
        query, key, value = jnp.split(fused_qkv, 3, axis=-1)
        jax.debug.print('4: query={x}', x=query)
        jax.debug.print('4: key={x}', x=key)
        jax.debug.print('4: value={x}', x=value)
        # key = with_sharding_constraint(key, PS(("dp", "fsdp"), None, "mp"))
        # value = with_sharding_constraint(value, PS(("dp", "fsdp"), None, "mp"))

        cos, sin = self.rotary_emb(seq_len)

        jax.debug.print('5: cos={x}', x=cos)
        jax.debug.print('5: sin={x}', x=sin)
        if self.rotary_ndims is not None:
            k_rot = key[..., : self.rotary_ndims]
            k_pass = key[..., self.rotary_ndims :]

            q_rot = query[..., : self.rotary_ndims]
            q_pass = query[..., self.rotary_ndims :]

            jax.debug.print('6: k_rot={x}', x=k_rot)
            jax.debug.print('6: k_pass={x}', x=k_pass)

            jax.debug.print('6: q_rot={x}', x=q_rot)
            jax.debug.print('6: q_pass={x}', x=q_pass)

            q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin, position_ids)

            jax.debug.print('7: q_rot={x}', x=q_rot)
            jax.debug.print('7: q_pass={x}', x=q_pass)

            key = jnp.concatenate([k_rot, k_pass], axis=-1)
            query = jnp.concatenate([q_rot, q_pass], axis=-1)

            jax.debug.print('8: key={x}', x=key)
            jax.debug.print('8: query={x}', x=query)
        else:
            query, key = apply_rotary_pos_emb(query, key, cos, sin, position_ids)
        query = query.astype(self.dtype)
        key = key.astype(self.dtype)

        query_length, key_length = query.shape[1], key.shape[1]

        if self.has_variable("cache", "cached_key"):
            mask_shift = self.variables["cache"]["cache_index"]
            max_decoder_length = self.variables["cache"]["cached_key"].shape[1]

            causal_mask = lax.dynamic_slice(
                self.causal_mask, (0, 0, mask_shift, 0), (1, 1, query_length, max_decoder_length)
            )
        else:
            causal_mask = self.causal_mask[:, :, :query_length, :key_length]

        causal_mask = jnp.broadcast_to(causal_mask, (batch,) + causal_mask.shape[1:])
        attention_mask = jnp.broadcast_to(jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape)
        attention_mask = combine_masks(attention_mask, causal_mask)

        dropout_rng = None
        if not deterministic and self.config.attention_dropout > 0.0:
            dropout_rng = self.make_rng("dropout")

        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.
        # if self.has_variable("cache", "cached_key") or init_cache:

        if self.has_variable("cache", "cached_key") or init_cache:
            key, value, attention_mask = self._concatenate_to_cache(key, value, query, attention_mask)

        # transform boolean mask into float mask
        attention_bias = lax.select(
            attention_mask > 0,
            jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
            jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
        )

        jax.debug.print('9: query={x}', x=query)
        jax.debug.print('9: key={x}', x=key)
        jax.debug.print('9: attention_bias={x}', x=attention_bias)

        # if False:
        attn_weights = my_dot_product_attention_weights(
            query,
            key,
            bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.config.attention_dropout,
            deterministic=deterministic,
            dtype=jnp.promote_types(self.dtype, jnp.float32),
            precision=None,
            apply_tanh=self.config.tanh_causal_att,
        )

        jax.debug.print('10: attn_weights={x}', x=attn_weights)

        # print()
        # attn_weights = with_sharding_constraint(attn_weights, PS(("dp", "fsdp"), "mp", None, None))
        attn_weights = attn_weights.astype(self.dtype)
        attn_output = jnp.einsum("bhqk,bkhd->bqhd", attn_weights, value.astype(self.dtype))
        # else:
        #     attn_output = dot_product_attention_multihead(
        #         query.astype(jnp.promote_types(self.dtype, jnp.float32)),
        #         key.astype(jnp.promote_types(self.dtype, jnp.float32)),
        #         value.astype(jnp.promote_types(self.dtype, jnp.float32)),
        #         bias=attention_bias.astype(jnp.promote_types(self.dtype, jnp.float32)),
        #         dropout_rng=dropout_rng,
        #         dropout_rate=self.config.attention_dropout,
        #         enable_dropout=not deterministic,
        #         dtype=jnp.promote_types(self.dtype, jnp.float32),
        #         precision=None,
        #         float32_logits=True
        #     ).astype(self.dtype)
        #     attn_weights = None

        attn_output = self._merge_heads(attn_output)
        attn_output = self.dense(attn_output).astype(self.dtype)

        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs
        # outputs = with_sharding_constraint(outputs, PS(("dp", "fsdp"), None, "mp"))

import numpy as np
def precompute_freqs_cis(dim: int, end: int, theta: float=10000.0) -> jnp.ndarray:
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[: (dim // 2)].astype(jnp.float32) / dim))
    t = np.arange(end)  # type: ignore
    freqs = np.outer(t, freqs).astype(jnp.float32)  # type: ignore
    sin, cos = np.sin(freqs), np.cos(freqs)
    freqs_cis = np.complex64(cos + 1j * sin)
    return jnp.asarray(freqs_cis)

def apply_rotary_emb_(
    xq: jnp.ndarray,
    xk: jnp.ndarray,
    freqs_cis: jnp.ndarray,
    dtype: jnp.dtype=jnp.float32,
    freqs_cis_k: jnp.ndarray = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:

    reshape_xq = xq.astype(jnp.float32).reshape(*xq.shape[:-1], -1, 2)
    reshape_xk = xk.astype(jnp.float32).reshape(*xk.shape[:-1], -1, 2)

    xq_ = jax.lax.complex(reshape_xq[..., 0], reshape_xq[..., 1])
    xk_ = jax.lax.complex(reshape_xk[..., 0], reshape_xk[..., 1])

    # add head dim
    freqs_cis = jnp.reshape(freqs_cis, (*freqs_cis.shape[:2], 1, *freqs_cis.shape[2:]))

    xq_out = xq_ * freqs_cis
    xq_out = jnp.stack((jnp.real(xq_out), jnp.imag(xq_out)), axis=-1).reshape(*xq_out.shape[:-1], -1)
    if freqs_cis_k is None:
        xk_out = xk_ * freqs_cis
        xk_out = jnp.stack((jnp.real(xk_out), jnp.imag(xk_out)), axis=-1).reshape(*xk_out.shape[:-1], -1)
    else:
        freqs_cis_k = jnp.reshape(freqs_cis_k, (*freqs_cis_k.shape[:2], 1, *freqs_cis_k.shape[2:]))
        xk_out = xk_ * freqs_cis_k
        xk_out = jnp.stack((jnp.real(xk_out), jnp.imag(xk_out)), axis=-1).reshape(*xk_out.shape[:-1], -1)

    return xq_out.astype(dtype), xk_out.astype(dtype)

def apply_rotary_emb(
    xq: jnp.ndarray,
    xk: jnp.ndarray,
    freqs_cis: jnp.ndarray,
    dtype: jnp.dtype=jnp.float32,
    freqs_cis_k: jnp.ndarray = None,
    rot_dim: Optional[int] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    if rot_dim is not None and rot_dim>0:

        # Separate the tensors based on the rotation dimensions
        xq_rot, xq_pass = xq[..., :rot_dim], xq[..., rot_dim:]
        xk_rot, xk_pass = xk[..., :rot_dim], xk[..., rot_dim:]


        # Apply the function on the parts that need rotation
        xq_rot, xk_rot = apply_rotary_emb_(xq_rot, xk_rot, freqs_cis, dtype=dtype, freqs_cis_k=freqs_cis_k)

        # Concatenate the rotated and non-rotated parts
        xq_out = jnp.concatenate((xq_rot, xq_pass), axis=-1)
        xk_out = jnp.concatenate((xk_rot, xk_pass), axis=-1)
    else:
        xq_out, xk_out = apply_rotary_emb_(xq, xk, freqs_cis, dtype=dtype, freqs_cis_k=freqs_cis_k)

    return xq_out, xk_out

from einops import repeat, rearrange
from typing import Optional, Union

class FlaxGPTNeoXCrossAttention(nn.Module):
    config: GPTNeoXConfig
    dtype: jnp.dtype=jnp.float32
    param_dtype: jnp.dtype=jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]]=None

    def setup(self):
        config = self.config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.qk_layernorm = config.qk_layernorm
        self.head_dim = self.embed_dim // self.num_heads
        kernel_init = jax.nn.initializers.normal(self.config.initializer_range)

        self.wq = nn.Dense(
            config.num_attention_heads*self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=self.config.attention_bias,
            kernel_init=kernel_init,
            precision=self.precision,
        )
        self.wk = nn.Dense(
            config.num_attention_heads*self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=self.config.attention_bias,
            kernel_init=kernel_init,
            precision=self.precision,
        )
        self.wv = nn.Dense(
            config.num_attention_heads*self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=self.config.attention_bias,
            kernel_init=kernel_init,
            precision=self.precision,
        )

        self.wo = nn.Dense(
            config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=self.config.attention_bias,
            kernel_init=kernel_init,
            precision=self.precision,
        )


        self.freqs_cis = precompute_freqs_cis(
            self.head_dim,
            config.max_position_embeddings * 2,
        )
        self.null_k =  self.param(f'null_k', jax.nn.initializers.normal(self.config.null_k_init_mult*self.config.null_attn_init), (1,1,self.num_heads,self.head_dim))
        self.null_v =  self.param(f'null_v', jax.nn.initializers.normal(self.config.null_attn_init), (1,1,self.num_heads,self.head_dim))
        if self.qk_layernorm:
            self.q_layernorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
            self.k_layernorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)




    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))


    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    @nn.compact
    def _concatenate_to_cache(self, key, value, attention_mask):
        """
        This function takes projected key and value states and concatenates them to the cached states.
        """
        # detect if we're initializing by absence of existing cache data.
        # is_initialized = self.has_variable("cache", "cached_key")
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        cached_attention_mask = self.variable("cache", "attention_mask", jnp.zeros, attention_mask.shape, attention_mask.dtype)
        # cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        # if is_initialized:
        # cached_key.value = key
        # cached_value.value = value
        # cached_attention_mask.value = attention_mask
        cached_key.value = key[:,-1:,...]
        cached_value.value = value[:,-1:,...]
        cached_attention_mask.value = attention_mask[:,-1:,...]
            # cache_index.value = cache_index.value + 1

        return key, value, attention_mask

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        key_value_states: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        kv_position_ids: Optional[jnp.ndarray] = None,
        attention_mask: Optional[jnp.ndarray] = None,
        retriever_scores: Optional[jnp.ndarray] = None,
        output_attentions: bool = False,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray]:
        #TODO: add init_cca_cache and a _concatenate_to_cache function and document this...

        is_cross_attention = key_value_states is not None



        xq = self.wq(hidden_states)
        xq = self._split_heads(xq).astype(self.dtype)
        xq = self.q_layernorm(xq) if self.qk_layernorm else xq

        if not is_cross_attention:
            key_value_states = hidden_states
        # if is_init and use_cca_cache:
        #     xk, xv = self.variables["cache"]["cached_key"], self.variables["cache"]["cached_value"]
        #     attention_mask =  self.variables["cache"]["attention_mask"]
        #     # attention_mask = lax.dynamic_slice(
        #     #     attention_mask, (0, 0, mask_shift, 0), (1, 1, query_length, max_decoder_length)
        #     # )
        # else:
        xk = self.wk(key_value_states)
        xk = self._split_heads(xk).astype(self.dtype)
        xk = self.k_layernorm(xk) if self.qk_layernorm else xk

        xv = self.wv(key_value_states)
        xv = self._split_heads(xv).astype(self.dtype)




        null_k = self.k_layernorm(self.null_k) if self.qk_layernorm else self.null_k



        query_length, key_length = xq.shape[1], xk.shape[1]
        batch_size = hidden_states.shape[0]
        print(f"{xq.shape=}, {xk.shape=}, {self.has_variable('cache', 'cached_key')=}")

        if position_ids is None:
            position_ids = jnp.arange(query_length, dtype=jnp.int32)
            position_ids = jnp.broadcast_to(position_ids[None, :], (batch_size, query_length))

        freqs_cis = jnp.take(self.freqs_cis, position_ids, axis=0)
        if not is_cross_attention:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis, dtype=self.dtype, rot_dim=self.head_dim)
        else:
            if kv_position_ids is None:
                kv_position_ids = jnp.arange(key_length, dtype=jnp.int32)
                kv_position_ids = jnp.broadcast_to(kv_position_ids[None, :], (batch_size, key_length))
            freqs_cis_k = jnp.take(self.freqs_cis, kv_position_ids, axis=0)
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis, freqs_cis_k=freqs_cis_k, dtype=self.dtype, rot_dim=self.head_dim)

        null_k = jnp.broadcast_to(null_k, (batch_size, 1, self.num_heads, self.head_dim)).astype(self.dtype)
        xk = jnp.concatenate((xk, null_k), axis = -3)
        null_v = jnp.broadcast_to(self.null_v, (batch_size, 1, self.num_heads, self.head_dim)).astype(self.dtype)
        xv = jnp.concatenate((xv, null_v), axis = -3)


        if attention_mask is not None:

            null_mask = jnp.ones((attention_mask.shape[0], 1), dtype=jnp.float32)
            attention_mask = jnp.concatenate((attention_mask, null_mask), axis = -1)
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
            if retriever_scores is None:
                attention_bias = jnp.full(attention_mask.shape, 0.0).astype(self.dtype)
            else:
                null_ret_score = jnp.zeros((retriever_scores.shape[0], 1), dtype=jnp.float32)
                attention_bias = jnp.concatenate((retriever_scores, null_ret_score), axis = -1)
                attention_bias = jnp.expand_dims(attention_bias, axis=(-3, -2))


            attention_bias = lax.select(
                attention_mask > 0,
                attention_bias.astype(self.dtype),
                jnp.full(attention_bias.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
            )
            if xq.shape[0]!=attention_bias.shape[0]:
              attention_bias = attention_bias[:batch_size,...]

        else:
            attention_bias = None

        dropout_rng = None
        if not deterministic and self.config.attention_dropout > 0.0:
            dropout_rng = self.make_rng("dropout")


        attn_weights = my_dot_product_attention_weights(
            xq,
            xk,
            bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.config.attention_dropout,
            deterministic=deterministic,
            dtype=jnp.promote_types(self.dtype, jnp.float32),
            precision=self.precision,
            apply_tanh=self.config.tanh_xatt,
        ).astype(self.dtype)
        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, xv.astype(self.dtype), precision=self.precision)

        attn_output = self._merge_heads(attn_output)
        attn_output = self.wo(attn_output).astype(self.dtype)
        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs




class FlaxGPTNeoXChunkedCrossAttention(nn.Module):
    config: GPTNeoXConfig
    dtype: jnp.dtype=jnp.float32
    param_dtype: jnp.dtype=jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]]=None

    def setup(self):
        self.chunk_size = self.config.chunk_size
        self.num_neighbors = self.config.num_neighbors
        self.cross_attention  = FlaxGPTNeoXCrossAttention(self.config, dtype=self.dtype, param_dtype=self.param_dtype, precision=self.precision)

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        neighbor_hidden_states,
        neighbor_mask,
        nei_position_ids,
        att_scores,
        position_ids: Optional[jnp.array] = None,
        output_attentions: bool=False,
        deterministic:bool=True,
        chunk_size:Optional[int]=None,
        n_chunks_per_window:Optional[int]=None,
    ):
        if not self.config.ret_score_in_cca:
             att_scores=None

        if self.config.apply_refactor:
            func = self.new_call
        else:
            func = self.old_call
        func = jax.vmap(func,
                in_axes=(0, 0, 0, 0, 0 if att_scores is  not None else None,
                        0 if position_ids is  not None else None, None, None, None, None),
                out_axes=0,
                # variable_axes={'params': None, 'intermediates': 0},
                # split_rngs={'dropout': True, "params": False},
            )
        return func(hidden_states, neighbor_hidden_states, neighbor_mask, nei_position_ids, att_scores,
                     position_ids, output_attentions, deterministic, chunk_size, n_chunks_per_window)
    def old_call(
        self,
        hidden_states: jnp.ndarray,
        neighbor_hidden_states,
        neighbor_mask,
        nei_position_ids,
        att_scores,
        position_ids: Optional[jnp.array] = None,
        output_attentions: bool=False,
        deterministic:bool=True,
        chunk_size:Optional[int]=None,
        n_chunks_per_window:Optional[int]=None,
        ):
        num_neighbors = self.config.num_neighbors
        if chunk_size is None:
            chunk_size = self.config.chunk_size

        causal_padding = chunk_size - 1
        num_devices, seq_len, hidden_dim = hidden_states.shape
        print(f"{neighbor_hidden_states=}")
        num_neighbors = neighbor_hidden_states.shape[-3]
        neighbor_hidden_states = neighbor_hidden_states.reshape([-1, 2*chunk_size*num_neighbors, hidden_dim])
        num_document_chunks = neighbor_hidden_states.shape[0]
        nei_position_ids = nei_position_ids.reshape([-1, 2*chunk_size*num_neighbors])



        neighbor_mask = neighbor_mask.reshape([-1, 2*chunk_size*num_neighbors])
        # num_document_chunks, num_neighbors, _, _ = neighbor_hidden_states.shape

        # -> (-1, num_devices_chunks, num_neighbors, 2*chunk_size, hidden_dim)
        if att_scores is not None:
            att_scores = jnp.broadcast_to(att_scores[...,None], att_scores.shape+(2*chunk_size,)).reshape([-1, 2*chunk_size*num_neighbors])
        #TODO: smells really bad
        if num_document_chunks>1:
            num_devices_chunks = num_document_chunks//hidden_states.shape[0]
            # ->  (-1 ,chunk_size, hidden_dim)
            hidden_states = hidden_states.reshape([-1, num_devices_chunks*chunk_size, hidden_dim])
            hidden_states = jnp.pad(hidden_states[:,causal_padding:,:], ((0,0),(0, causal_padding),(0,0)), 'constant')
            hidden_states = hidden_states.reshape([-1,chunk_size, hidden_dim])

            position_ids = jnp.arange(chunk_size)+chunk_size-1
            position_ids = jnp.broadcast_to(position_ids[None, :], (hidden_states.shape[0], chunk_size))
        else:
            hidden_states = hidden_states.reshape([1,1, hidden_dim])
            assert position_ids is not None





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
        if num_document_chunks>1:
            cross_attention_out = cross_attention_out.reshape([-1, num_devices_chunks*chunk_size, hidden_dim])
            # # pad back to original, with 0s at the beginning (which will be added to the residual and be fine)
            cross_attention_out = jnp.pad(cross_attention_out, ((0,0),(causal_padding, 0),(0,0)), 'constant')[:,:-causal_padding]
        cross_attention_out = cross_attention_out.reshape([num_devices, seq_len, hidden_dim])
        return (cross_attention_out,)+output[1:]


    def new_call(
        self,
        hidden_states: jnp.ndarray,
        neighbor_hidden_states,
        neighbor_mask,
        nei_position_ids,
        att_scores,
        position_ids: Optional[jnp.array] = None,
        output_attentions: bool=False,
        deterministic:bool=True,
        chunk_size:Optional[int]=None,
        n_chunks_per_window:Optional[int]=None,
    ):

        batch_size, seq_len, hidden_dim = hidden_states.shape
        if chunk_size is None:
            chunk_size = self.chunk_size
        # chunk_size = self.chunk_size
        causal_padding = chunk_size - 1
        is_generating = position_ids is not None
        num_document_chunks, num_neighbors, ret_size = neighbor_mask.shape
        neighbor_mask = rearrange(neighbor_mask, 'b k r -> b (k r)')
        nei_position_ids = rearrange(nei_position_ids, 'b k r -> b (k r)')
        neighbor_hidden_states = rearrange(neighbor_hidden_states, 'b k r d-> b (k r) d')
        if att_scores is not None:
            att_scores = rearrange(att_scores, 'b k r -> b (k r)')


        # TODO: remove this
        nei_position_ids = jnp.clip(nei_position_ids, a_min=0, a_max=2*self.config.chunk_size-1)


        # -> (-1, n_chunks_per_window, num_neighbors, 2*chunk_size, hidden_dim)
        if not is_generating:
            if n_chunks_per_window is None:
                # n_chunks_per_window = seq_len//chunk_size
                n_chunks_per_window = num_document_chunks//hidden_states.shape[0]
            # ->  (-1 ,chunk_size, hidden_dim)
            hidden_states = hidden_states.reshape([-1, n_chunks_per_window*chunk_size, hidden_dim])
            hidden_states = jnp.pad(hidden_states[:,causal_padding:,:], ((0,0),(0, causal_padding),(0,0)), 'constant')
            hidden_states = hidden_states.reshape([-1, chunk_size, hidden_dim])

            position_ids = jnp.arange(chunk_size)+chunk_size-1
            position_ids = jnp.broadcast_to(position_ids, hidden_states.shape[:2])
        else:
            hidden_states = hidden_states.reshape([1,1, hidden_dim])


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
            cross_attention_out = cross_attention_out.reshape([-1, n_chunks_per_window*chunk_size, hidden_dim])
            # # pad back to original, with 0s at the beginning (which will be added to the residual and be fine)
            cross_attention_out = jnp.pad(cross_attention_out, ((0,0),(causal_padding, 0),(0,0)), 'constant')[:,:-causal_padding]
        cross_attention_out = cross_attention_out.reshape([batch_size, seq_len, hidden_dim])
        return (cross_attention_out,)+output[1:]

class FlaxGPTNeoXMLP(nn.Module):
    config: GPTNeoXConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    intermediate_size: Optional[int] = None

    def setup(self):
        embed_dim = self.config.hidden_size
        if self.intermediate_size is None:
            intermediate_size = self.config.intermediate_size
        else:
            intermediate_size = self.intermediate_size

        kernel_init = jax.nn.initializers.normal(self.config.initializer_range)

        self.dense_h_to_4h = nn.Dense(intermediate_size, dtype=self.dtype, kernel_init=kernel_init, param_dtype=self.param_dtype)
        self.dense_4h_to_h = nn.Dense(embed_dim, dtype=self.dtype, kernel_init=kernel_init, param_dtype=self.param_dtype)

        self.act = ACT2FN[self.config.hidden_act]

    def __call__(self, hidden_states):
        hidden_states = self.dense_h_to_4h(hidden_states).astype(self.dtype)
        hidden_states = self.act(hidden_states).astype(self.dtype)
        hidden_states = self.dense_4h_to_h(hidden_states).astype(self.dtype)
        return hidden_states


class FlaxGPTNeoXBlock(nn.Module):
    config: GPTNeoXConfig
    dtype: jnp.dtype = jnp.float32
    has_cca: bool = False
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self):
        self.use_parallel_residual = self.config.use_parallel_residual
        self.input_layernorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        attention_module = FlaxGPTNeoXAttention
        if self.config.remat_attention != '':
            attention_module = remat(
                attention_module, static_argnums=(3, 4, 5),
                policy=get_gradient_checkpoint_policy(self.config.remat_attention)
            )
        self.attention = attention_module(self.config, dtype=self.dtype, param_dtype=self.param_dtype)
        self.post_attention_dropout = nn.Dropout(rate=self.config.hidden_dropout)
        self.post_attention_layernorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

        self.mlp = FlaxGPTNeoXMLP(self.config, dtype=self.dtype, param_dtype=self.param_dtype)
        self.post_mlp_dropout = nn.Dropout(rate=self.config.hidden_dropout)
        if self.has_cca:
            self.cca = FlaxGPTNeoXChunkedCrossAttention(
                self.config,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
            )
            self.cca_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps,
                                         dtype=self.dtype,
                                         scale_init=jax.nn.initializers.constant(self.config.cca_layernorm_init_scale),)


        else:
            self.cca = None
            self.cca_norm = None



    def __call__(
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
        print(f"{hidden_states=}",flush=True)
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
                cca_hidden_states = (30*jax.nn.tanh(cca_hidden_states/30)).astype(cca_hidden_states.dtype)
            if not self.use_parallel_residual:
                hidden_states = cca_hidden_states + hidden_states
        else:
            cca_hidden_states = None


        jax.debug.print('sum={x}', x=jnp.sum(hidden_states))
        att_input = self.input_layernorm(hidden_states).astype(self.dtype)
        jax.debug.print('att_input={x}', x=att_input)
        #print(f"{att_input=}",flush=True)
        # att_input = with_sharding_constraint(att_input, PS(("dp", "fsdp"), None, "mp"))
        # attention_mask = with_sharding_constraint(attention_mask, PS(("dp", "fsdp"), None))

        attn_outputs = self.attention(
            att_input,
            attention_mask,
            position_ids,
            deterministic,
            init_cache,
            output_attentions,
        )
        attn_output = attn_outputs[0]
        attn_output = self.post_attention_dropout(attn_output, deterministic=deterministic)
        # attn_output = with_sharding_constraint(attn_output, PS(("dp", "fsdp"), None, "mp"))



        if self.use_parallel_residual:
            # pseudocode:
            # x = x + attn(ln1(x)) + mlp(ln2(x))
            mlp_output = self.mlp(self.post_attention_layernorm(hidden_states).astype(self.dtype))
            # mlp_output = with_sharding_constraint(mlp_output, PS(("dp", "fsdp"), None, "mp"))
            mlp_output = self.post_mlp_dropout(mlp_output, deterministic=deterministic)
            if cca_hidden_states is None:
                hidden_states = mlp_output + attn_output + hidden_states
            else:
                if self.config.log_debug_metrics and not self.is_initializing():
                    # TODO:
                    cca_hidden_states_norm = jax.numpy.linalg.norm(cca_hidden_states, axis=-1).mean(where=attention_mask>0)
                    att_hidden_states_norm = jax.numpy.linalg.norm(attn_output, axis=-1).mean(where=attention_mask>0)
                    mlp_hidden_states_norm = jax.numpy.linalg.norm(mlp_output, axis=-1).mean(where=attention_mask>0)
                    residual_hidden_states_norm = jax.numpy.linalg.norm(hidden_states, axis=-1).mean(where=attention_mask>0)
                    self.sow('intermediates', f'cca_hidden_states_norm', cca_hidden_states_norm)
                    self.sow('intermediates', f'att_hidden_states_norm', att_hidden_states_norm)
                    self.sow('intermediates', f'mlp_hidden_states_norm', mlp_hidden_states_norm)
                    self.sow('intermediates', f'residual_hidden_states_norm', residual_hidden_states_norm)
                hidden_states = mlp_output + attn_output + cca_hidden_states + hidden_states
                if self.config.log_debug_metrics and not self.is_initializing():
                    added_hidden_states_norm = jax.numpy.linalg.norm(hidden_states, axis=-1).mean(where=attention_mask>0)
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

from transformers.generation.flax_logits_process import FlaxLogitsProcessorList
from transformers.generation.flax_utils import SampleState

class FlaxGPTNeoXPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GPTNeoXConfig
    base_model_prefix = "gpt_neox"
    module_class: nn.Module = None

    # Copied from transformers.models.gpt_neo.modeling_flax_gpt_neo.FlaxGPTNeoPreTrainedModel.__init__ with GPTNeo->GPTNeoX
    def __init__(
        self,
        config: GPTNeoXConfig,
        input_shape: Tuple = (1, 128),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    # Copied from transformers.models.gpt_neo.modeling_flax_gpt_neo.FlaxGPTNeoPreTrainedModel.init_weights with GPTNeo->GPTNeoX
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # init input tensors
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        random_params = self.module.init(rngs, input_ids, attention_mask, position_ids, return_dict=False)["params"]

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    # Copied from transformers.models.gpt_neo.modeling_flax_gpt_neo.FlaxGPTNeoPreTrainedModel.init_cache
    def init_cache(self, batch_size, max_length):
        r"""
        Args:
            batch_size (`int`):
                batch_size used for fast auto-regressive decoding. Defines the batch size of the initialized cache.
            max_length (`int`):
                maximum possible length for auto-regressive decoding. Defines the sequence length of the initialized
                cache.
        """
        # init input variables to retrieve cache
        input_ids = jnp.ones((batch_size, max_length))
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        init_variables = self.module.init(
            jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True
        )
        return unfreeze(init_variables["cache"])

    @add_start_docstrings_to_model_forward(GPT_NEOX_INPUTS_DOCSTRING)
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        params: dict = None,
        past_key_values: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        upcoder_input=None,
        retriever_supervision=None,
        encoded_neighbors=None,
        # use_cca_cache:bool=False,
        chunk_index: Optional[jnp.ndarray] = None,
    ):
        rpt_kwargs = dict(upcoder_input = upcoder_input,
                    retriever_supervision = retriever_supervision,
                    encoded_neighbors= encoded_neighbors,
                    # use_cca_cache=use_cca_cache,
                    chunk_index=chunk_index,
                    )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        batch_size, sequence_length = input_ids.shape

        if position_ids is None:
            if past_key_values is not None:
                if attention_mask is None:
                    raise ValueError("Make sure to provide `position_ids` when passing `past_key_values`.")
                else:
                    position_ids = jnp.broadcast_to(
                        jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0),
                        (batch_size, sequence_length)
                    )
            else:
                position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length))

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        inputs = {"params": params or self.params}

        # if past_key_values are passed then cache is already initialized a private flag init_cache has to be passed down to ensure cache is used. It has to be made sure that cache is marked as mutable so that it can be changed by FlaxGPTNeoXAttention module
        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        outputs = self.module.apply(
            inputs,
            input_ids = jnp.array(input_ids, dtype="i4"),
            attention_mask=jnp.array(attention_mask, dtype="i4"),
            position_ids=jnp.array(position_ids, dtype="i4"),
            deterministic=not train,
            init_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            rngs=rngs,
            mutable=mutable,
            **rpt_kwargs
        )

        # add updated cache to model output
        if past_key_values is not None and return_dict:
            outputs, past_key_values = outputs
            outputs["past_key_values"] = unfreeze(past_key_values["cache"])
            return outputs
        elif past_key_values is not None and not return_dict:
            outputs, past_key_values = outputs
            outputs = outputs[:1] + (unfreeze(past_key_values["cache"]),) + outputs[1:]

        return outputs



    #almost the same as the _sample in FlaxGenerationMixin, except we return SampleState instead of FlaxSampleOutput
    def _sample(
        self,
        input_ids: None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        prng_key: Optional[jnp.ndarray] = None,
        logits_processor: Optional[FlaxLogitsProcessorList] = None,
        logits_warper: Optional[FlaxLogitsProcessorList] = None,
        trace: bool = True,
        params: Optional[Dict[str, jnp.ndarray]] = None,
        model_kwargs: Optional[Dict[str, jnp.ndarray]] = None,
    ):
        # init values
        max_length = max_length if max_length is not None else self.generation_config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        prng_key = prng_key if prng_key is not None else jax.random.PRNGKey(0)

        batch_size, cur_len = input_ids.shape

        eos_token_id = jnp.array(eos_token_id, dtype=jnp.int32 if eos_token_id is not None else None)
        pad_token_id = jnp.array(pad_token_id, dtype=jnp.int32)
        cur_len = jnp.array(cur_len)

        # per batch-item holding current token in loop.
        sequences = jnp.full((batch_size, max_length), pad_token_id, dtype=jnp.int32)
        sequences = lax.dynamic_update_slice(sequences, input_ids, (0, 0))

        # per batch-item state bit indicating if sentence has finished.
        is_sent_finished = jnp.zeros((batch_size,), dtype=jnp.bool_)

        # For Seq2Seq generation, we only need to use the decoder instead of the whole model in generation loop
        # and pass it the `encoder_outputs`, which are part of the `model_kwargs`.
        model = self.decode if self.config.is_encoder_decoder else self

        # initialize model specific kwargs
        model_kwargs = self.prepare_inputs_for_generation(input_ids, max_length, **model_kwargs)

        # initialize state
        state = SampleState(
            cur_len=cur_len,
            sequences=sequences,
            running_token=input_ids,
            is_sent_finished=is_sent_finished,
            prng_key=prng_key,
            model_kwargs=model_kwargs,
        )

        def sample_search_cond_fn(state):
            """state termination condition fn."""
            has_reached_max_length = state.cur_len == max_length
            all_sequence_finished = jnp.all(state.is_sent_finished)
            finish_generation = jnp.logical_or(has_reached_max_length, all_sequence_finished)
            return ~finish_generation

        def sample_search_body_fn(state):
            """state update fn."""
            prng_key, prng_key_next = jax.random.split(state.prng_key)
            model_outputs = model(state.running_token, params=params, **state.model_kwargs)

            logits = model_outputs.logits[:, -1]

            # apply min_length, ...
            logits = logits_processor(state.sequences, logits, state.cur_len)
            # apply top_p, top_k, temperature
            logits = logits_warper(logits, logits, state.cur_len)

            next_token = jax.random.categorical(prng_key, logits, axis=-1)

            next_is_sent_finished = state.is_sent_finished | (next_token == eos_token_id)
            next_token = next_token * ~next_is_sent_finished + pad_token_id * next_is_sent_finished
            next_token = next_token[:, None]

            next_sequences = lax.dynamic_update_slice(state.sequences, next_token, (0, state.cur_len))
            next_model_kwargs = self.update_inputs_for_generation(model_outputs, state.model_kwargs)

            return SampleState(
                cur_len=state.cur_len + 1,
                sequences=next_sequences,
                running_token=next_token,
                is_sent_finished=next_is_sent_finished,
                model_kwargs=next_model_kwargs,
                prng_key=prng_key_next,
            )

        # The very first prompt often has sequence length > 1, so run outside of `lax.while_loop` to comply with TPU
        if input_ids.shape[1] > 1:
            state = sample_search_body_fn(state)
        # state = lax.cond(model_kwargs['attention_mask'].sum() > 1, lambda: sample_search_body_fn(state), lambda: state )

        if not trace:
            state = self._run_loop_in_debug(sample_search_cond_fn, sample_search_body_fn, state)
        else:
            state = lax.while_loop(sample_search_cond_fn, sample_search_body_fn, state)

        past_key_values = state.model_kwargs['past_key_values']
        last_lowcoder_states = past_key_values['transformer']['cached_array']

        encoded_lowcoder_states = self.preret_forward(
                           hidden_states=last_lowcoder_states,
                           attention_mask = jnp.ones(last_lowcoder_states.shape[:-1]),
                           params=params)

        return state, encoded_lowcoder_states


    def _greedy_search(
        self,
        input_ids: None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        logits_processor: Optional[FlaxLogitsProcessorList] = None,
        trace: bool = True,
        end_seq = [187,535],
        params: Optional[Dict[str, jnp.ndarray]] = None,
        model_kwargs: Optional[Dict[str, jnp.ndarray]] = None,
    ):
        # init values
        end_seq = jnp.array(end_seq).astype(int)
        max_length = max_length if max_length is not None else self.generation_config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id

        batch_size, cur_len = input_ids.shape

        eos_token_id = jnp.array(eos_token_id, dtype=jnp.int32 if eos_token_id is not None else None)
        pad_token_id = jnp.array(pad_token_id, dtype=jnp.int32)
        cur_len = jnp.array(cur_len)

        # per batch-item holding current token in loop.
        sequences = jnp.full((batch_size, max_length), pad_token_id, dtype=jnp.int32)
        sequences = lax.dynamic_update_slice(sequences, input_ids, (0, 0))

        # per batch-item state bit indicating if sentence has finished.
        is_sent_finished = jnp.zeros((batch_size,), dtype=jnp.bool_)

        # For Seq2Seq generation, we only need to use the decoder instead of the whole model in generation loop
        # and pass it the `encoder_outputs`, which are part of the `model_kwargs`.
        model =  self
        # initialize model specific kwargs
        model_kwargs = self.prepare_inputs_for_generation(input_ids, max_length, **model_kwargs)

        # initialize state
        state = GreedyState(
            cur_len=cur_len,
            sequences=sequences,
            running_token=input_ids,
            is_sent_finished=is_sent_finished,
            model_kwargs=model_kwargs,
        )

        def greedy_search_cond_fn(state):
            """state termination condition fn."""
            has_reached_max_length = state.cur_len == max_length
            all_sequence_finished = jnp.all(state.is_sent_finished)
            finish_generation = jnp.logical_or(has_reached_max_length, all_sequence_finished)
            return ~finish_generation

        def greedy_search_body_fn(state):
            """state update fn."""
            model_outputs = model(state.running_token, params=params, **state.model_kwargs)
            logits = model_outputs.logits[:, -1]

            # apply min_length, ...
            if logits_processor is not None:
                logits = logits_processor(state.sequences, logits, state.cur_len)

            next_token = jnp.argmax(logits, axis=-1)

            next_token = next_token * ~state.is_sent_finished + pad_token_id * state.is_sent_finished
            # next_is_sent_finished = state.is_sent_finished | (next_token == eos_token_id)
            next_is_sent_finished = state.is_sent_finished | (next_token == eos_token_id) | jnp.isin(next_token,end_seq)
            next_token = next_token[:, None]

            next_sequences = lax.dynamic_update_slice(state.sequences, next_token, (0, state.cur_len))
            next_model_kwargs = self.update_inputs_for_generation(model_outputs, state.model_kwargs)
            return GreedyState(
                cur_len=state.cur_len + 1,
                sequences=next_sequences,
                running_token=next_token,
                is_sent_finished=next_is_sent_finished,
                model_kwargs=next_model_kwargs,
            )

        # The very first prompt often has sequence length > 1, so run outside of `lax.while_loop` to comply with TPU
        if input_ids.shape[1] > 1:
            state = greedy_search_body_fn(state)

        if not trace:
            state = self._run_loop_in_debug(greedy_search_cond_fn, greedy_search_body_fn, state)
        else:
            state = lax.while_loop(greedy_search_cond_fn, greedy_search_body_fn, state)

        return state.sequences

    def single_lowcoder_forward(
            self,
            input_ids,
            attention_mask=None,
            position_ids=None,
            chunk_index=None,
            params: dict = None,
            train: bool = False,
            past_key_values: dict = None,
            output_attentions:bool = False,
            dropout_rng = None,
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
            params: dict = None,
            train: bool = False,
            past_key_values: dict = None,
            output_attentions:bool = False,
            dropout_rng = None,
            ):

        input_ids = with_sharding_constraint(input_ids, PS(('dp', 'fsdp')))
        attention_mask = with_sharding_constraint(attention_mask, PS(('dp', 'fsdp')))
        apply_kwargs = self.create_apply_kwargs(params, dropout_rng, past_key_values)
        # input_ids, attention_mask = jax.tree_map(reshape_for_vmap, (input_ids, attention_mask))
        return self.module.apply(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                deterministic=not train,
                output_attentions=output_attentions,
                method=self.module._lowcoder_forward,
                **apply_kwargs
        )


    def preret_forward(
            self,
            hidden_states,
            attention_mask=None,
            params: dict = None,
            train: bool = False,
            output_attentions:bool = False,
            dropout_rng = None,
        ):

        apply_kwargs = self.create_apply_kwargs(params, dropout_rng)


        outputs = self.module.apply(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                deterministic= not train,
                output_attentions=output_attentions,
                method=self.module._encode_forward,
                **apply_kwargs
        )

        return outputs

    def create_apply_kwargs(self, params, dropout_rng, past_key_values=None):
        rngs = {}
        # Handle any PRNG if needed
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        variables = {"params": params or self.params}

        # if past_key_values are passed then cache is already initialized a private flag init_cache
        # has to be passed down to ensure cache is used. It has to be made sure that cache is marked as mutable
        # so that it can be changed by FlaxGPTJAttention module
        if past_key_values is not None:
            variables["cache"] = past_key_values
            mutable = ["cache","intermediates"]
        else:
            mutable = False
        return dict(rngs=rngs, variables=variables, mutable=mutable)

class FlaxGPTNeoXBlockCollection(nn.Module):
    config: GPTNeoXConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.first_upcoder_layer = self.config.num_hidden_layers//2
        self.lowcoder_layer_idxs = np.arange(self.first_upcoder_layer)
        self.upcoder_layer_idxs = np.arange(self.first_upcoder_layer, self.config.num_hidden_layers)
        if self.config.cca_freq>0:
            self.cca_layer_idxs = np.arange(self.first_upcoder_layer,
                                         self.config.num_hidden_layers,
                                         self.config.cca_freq)
        else:
            self.cca_layer_idxs = set()

        self.blocks = [
            FlaxGPTNeoXBlock(self.config, name=str(i), dtype=self.dtype, param_dtype=self.param_dtype,
                             has_cca=i in list(self.cca_layer_idxs),
                             ) for i in range(self.config.num_hidden_layers)
        ]


    def __call__(
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
        mode:str = "all",
        cca_kwargs: Optional[Dict] = None,
        # use_cca_cache:bool=False
    ):
        if mode =="all":
            blocks = self.blocks
        elif mode=="lowcoder":
            blocks = [self.blocks[i] for i in self.lowcoder_layer_idxs]
        elif mode=="upcoder":
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

        return FlaxBaseModelOutputCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


from collections import namedtuple
RetrieverSupervision = namedtuple('RetrieverSupervision', ['nei_scores', 'nei_idx'])




class FlaxGPTNeoXQueryAugmentorLayer(nn.Module):
    config: GPTNeoXConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    def setup(self):
        kwargs = dict(width=self.config.hidden_size, dtype=self.dtype, param_dtype=self.param_dtype, num_blocks=self.config.gate_num_blocks, after_refactor=self.config.gate_refactor)
        self.nei_gate = GriffinGate(**kwargs, a_init=self.config.a_init_nei) if self.config.apply_gating else None
        self.nei_to_query_att = FlaxGPTNeoXCrossAttention(self.config, dtype=self.dtype, param_dtype=self.param_dtype)
        self.nei_att_ln = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.nei_lowrank_mlp = FlaxGPTNeoXMLP(self.config, dtype=self.dtype, param_dtype=self.param_dtype, intermediate_size=self.config.query_aug_dim)
        self.nei_lowrank_mlp_ln = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

        self.query_att_ln = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        if self.config.apply_query_to_nei_att:
            self.query_to_nei_att = FlaxGPTNeoXCrossAttention(self.config, dtype=self.dtype, param_dtype=self.param_dtype)
            self.query_lowrank_mlp = FlaxGPTNeoXMLP(self.config, dtype=self.dtype, param_dtype=self.param_dtype, intermediate_size=self.config.query_aug_dim)
            self.query_lowrank_mlp_ln = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
            self.query_gate = GriffinGate(**kwargs, a_init=self.config.a_init_query) if self.config.apply_gating else None
        else:
            self.query_to_nei_att = None
            self.query_lowrank_mlp = None
            self.query_lowrank_mlp_ln = None


    def __call__(self,
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


        original_query,original_nei = query, nei
        query,nei = self.query_att_ln(query), self.nei_att_ln(nei)
        num_document_chunks, num_neighbors, ret_size, hidden_dim = nei.shape

        if nei_position_ids is None:
            nei_position_ids = jnp.arange(ret_size)
            nei_position_ids = jnp.broadcast_to(nei_position_ids[None,None, :], (num_document_chunks, num_neighbors, ret_size))
        else:
            nei_position_ids = nei_position_ids.reshape([num_document_chunks, num_neighbors, ret_size])
        query = query.reshape([-1, chunk_size, hidden_dim])
        query_attention_mask = query_attention_mask.reshape([-1, chunk_size])
        # ###
        if self.config.apply_query_to_nei_att:
            query_w_nei = self.query_to_nei_att(query,
                                                key_value_states=rearrange(nei, 'b k r d-> b (k r) d'),
                                                attention_mask=rearrange(nei_attention_mask, 'b k r -> b (k r)'),
                                                kv_position_ids=rearrange(nei_position_ids, 'b k r -> b (k r)'))
            aug_query = query + query_w_nei[0]
            aug_query = self.query_lowrank_mlp(self.query_lowrank_mlp_ln(aug_query))+aug_query
            aug_query = aug_query.reshape(original_query.shape)
            if self.query_gate is not None:
                # if using griffin gate, beta is between [0 ,  U[0.1, 0.7] ] at init.
                # alpha is in [U[0.3, 0.9] , 1] at init.
                ######
                # at init query_alpha should be at least ~0.99
                query_alpha, query_beta = self.query_gate(aug_query)
            else:
                query_alpha, query_beta = 0,1
            query_gate = query_alpha*original_query + query_beta*aug_query

        copied_query = einops.repeat(query, "b c d -> (b k) c d", k=num_neighbors)
        copied_query = with_sharding_constraint(copied_query, PS(("dp", "fsdp"), None, "mp"))
        copied_q_mask = einops.repeat(query_attention_mask, "b c -> (b k) c", k=num_neighbors)
        copied_q_mask = with_sharding_constraint(copied_q_mask, PS(("dp", "fsdp"), None))
        nei = rearrange(nei, 'b k r d -> (b k) r d')
        nei_w_query = self.nei_to_query_att(nei,
                                            key_value_states=copied_query,
                                            attention_mask=copied_q_mask
                                            )
        print(f"{nei_w_query=}",flush=True)
        aug_nei = nei + nei_w_query[0]
        aug_nei = self.nei_lowrank_mlp(self.nei_lowrank_mlp_ln(aug_nei))+aug_nei
        aug_nei = aug_nei.reshape(original_nei.shape)
        if self.nei_gate is not None:
            # at init nei_alpha should be at ~0.9 in first layer, 0.95 in second and 0.99 in third.
            nei_alpha, nei_beta = self.nei_gate(aug_nei)
        else:
            nei_alpha, nei_beta = 0, 1
        nei_gate = nei_alpha*original_nei + nei_beta*aug_nei
        if self.config.apply_query_to_nei_att:
            return query_gate, nei_gate
        else:
            return original_query, nei_gate


class FlaxGPTNeoXQueryAugmentor(nn.Module):
    config: GPTNeoXConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    def setup(self):
        self.layers = [FlaxGPTNeoXQueryAugmentorLayer(self.config, dtype=self.dtype, param_dtype=self.param_dtype) for _ in range(self.config.n_query_aug_layers)]

    @partial(
        nn.vmap,
        in_axes=(0, 0, 0, 0),
        out_axes=0,
        variable_axes={'params': None, 'intermediates': 0, "cache" : 0},
        split_rngs={'dropout': True, "params": False},
    )
    def __call__(self, query, query_attention_mask, nei, nei_attention_mask):
        original_nei = nei

        for i,layer in enumerate(self.layers):
            query, nei = layer(query,
                            query_attention_mask,
                            nei,
                            nei_attention_mask,
                            None,
                            None,
                            is_last=i==self.config.n_query_aug_layers-1
                            )



        return nei

class FlaxGPTNeoXModule(nn.Module):
    config: GPTNeoXConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.embed_dim = self.config.hidden_size

        self.embed_in = nn.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            param_dtype=self.param_dtype,
        )
        self.emb_dropout = nn.Dropout(self.config.hidden_dropout)
        self.layers = FlaxGPTNeoXBlockCollection(self.config, dtype=self.dtype, param_dtype=self.param_dtype)
        self.final_layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        if self.config.cca_freq>0:
            self.retriever = FlaxGPTNeoXRetriever(self.config, dtype=self.dtype, param_dtype=self.param_dtype)
        else:
            self.retriever = None
        if self.config.n_query_aug_layers is not None and self.config.n_query_aug_layers>0:
            self.query_augmentor = FlaxGPTNeoXQueryAugmentor(self.config, dtype=self.dtype, param_dtype=self.param_dtype)
        else:
            self.query_augmentor = None


    def lowcoder(self,
        input_ids,
        attention_mask,
        position_ids,
        deterministic=True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        # use_cca_cache:bool=False,
        ):
        input_embeds = self.embed_in(input_ids.astype("i4"))

        jax.debug.print('input_embeds={x}', x=input_embeds)

        hidden_states = self.emb_dropout(input_embeds, deterministic=deterministic)

        jax.debug.print('emb_dropout={x}', x=hidden_states)

        lowcoder_outputs = self.layers(
            hidden_states,
            attention_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            mode="lowcoder",
            # use_cca_cache=use_cca_cache,
        )
        return lowcoder_outputs



    # @nn.compact
    # def _concatenate_to_lowcoder_cache(self, array):
    #     chunk_size = self.config.chunk_size
    #     is_initialized = self.has_variable("cache", "cached_array")
    #     *batch_dims, _, hidden_dim = array.shape
    #     cached_array = self.variable("cache", "cached_array",
    #                                  jnp.zeros,
    #                                  tuple(batch_dims)+(self.config.chunk_size, hidden_dim),
    #                                  array.dtype)
    #     if is_initialized:
    #         last_chunk = array[...,-chunk_size:,:]

    #         num_updated_cache_vectors = last_chunk.shape[-2]
    #         shift = self.config.chunk_size-num_updated_cache_vectors #will need to update if I change retrieval stride
    #         indices = (0,) * len(batch_dims) + (shift,  0)

    #         array_operand = jnp.roll(cached_array.value, shift=-num_updated_cache_vectors, axis=-2)
    #         cached_array.value = lax.dynamic_update_slice(array_operand,
    #                                     last_chunk,
    #                                     indices)
    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids=None,
        deterministic=True,
        retriever_supervision: RetrieverSupervision = None,
        lowonly_input_ids = None,
        lowonly_attention_mask = None,
        train_step: Optional[int] = None,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        upcoder_input = None,
        encoded_neighbors: Optional[EncodedNeighbors] = None,
        # use_cca_cache:bool=False,
        chunk_index: Optional[jnp.array] = None,
        mode:str = "all",
        retrieval_kwargs:Optional[dict] = None,
    ):
        lowcoder_outputs = None
        retriever_output = None
        neighbor_hidden_states = None
        neighbor_mask = None
        nei_position_ids = None
        retriever_input = None
        att_scores = None
        cca_kwargs = None

        if retrieval_kwargs is None:
            retrieval_kwargs = dict()
        if position_ids is None:
            position_ids = jnp.broadcast_to(
                jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0),
                input_ids.shape
            )

        if not self.is_initializing() and lowonly_input_ids is not None:
            # if we have lowonly_input_ids this means we are in SLED/FID mode
            # and we want to encode those inputs first so we could retrieve and attend to them.
            lo_encoded_output = self(lowonly_input_ids,
                                    lowonly_attention_mask,
                                     deterministic=deterministic,
                                     retrieval_kwargs=dict(pooling_size=lowonly_input_ids.shape[-1]),
                                     mode="encoded_output",
                                     )
            assert self.config.pooling_size>0
            retrieval_kwargs = dict(pooling_size=self.config.pooling_size,
                                    append_next_chunk=False,
                                    n_skip_chunks=0,
                                    num_neighbors=retrieval_kwargs.get("num_neighbors",None),
                                    )
            cca_kwargs = dict(chunk_size=self.config.pooling_size, n_chunks_per_window=1)
        else:
            lo_encoded_output = None


        input_embeds = self.embed_in(input_ids.astype("i4"))
        hidden_states = self.emb_dropout(input_embeds, deterministic=deterministic)

        jax.debug.print('emb_dropout: {x}', x=hidden_states)

        def lowcoder(hidden_states, attention_mask, position_ids):
            return self.layers(
                hidden_states,
                attention_mask,
                position_ids,
                deterministic=deterministic,
                init_cache=init_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mode="lowcoder",
            )
        lowcoder_outputs = lowcoder(hidden_states, attention_mask, position_ids)

        hidden_states = lowcoder_outputs.last_hidden_state if  return_dict else lowcoder_outputs[0]
        # if self.has_variable("cache", "cached_array") or init_cache:
        #     self._concatenate_to_lowcoder_cache(hidden_states)

        retriever_input = hidden_states
        if self.retriever is not None and np.prod(hidden_states.shape[:-1])>1:
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
                    encoded_output = FlaxGPTNeoXRetrieverEncodedOutput(
                                            original_hidden_states=None,
                                            encoded_hidden_states=lo_encoded_output.encoded_hidden_states,
                                            attention_mask=lo_encoded_output.attention_mask,
                                            key_chunks=lo_encoded_output.key_chunks,
                                            query_chunks=encoded_output.query_chunks,
                                            chunk_mask=None,
                                            preret_attention=None)
                retriever_output = self.retriever(encoded_output,
                                                retriever_supervision=retriever_supervision,
                                                deterministic=deterministic,
                                                train_step=train_step,
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
                                    init_cache=init_cache,
                                    output_attentions=output_attentions,
                                    output_hidden_states=output_hidden_states,
                                    return_dict=return_dict,
                                    chunk_index=chunk_index,
                                    cca_kwargs=cca_kwargs,
                                    mode="upcoder")


        hidden_states = upcoder_outputs.last_hidden_state if  return_dict else upcoder_outputs[0]
        hidden_states = self.final_layer_norm(hidden_states)


        if not return_dict:
            return (hidden_states,) + upcoder_outputs + lowcoder_outputs

        return FlaxGPTNeoXModelOutput(
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
    "The bare GPTNeoX Model transformer outputting raw hidden-states without any specific head on top.",
    GPT_NEOX_START_DOCSTRING,
)
class FlaxGPTNeoXModel(FlaxGPTNeoXPreTrainedModel):
    module_class = FlaxGPTNeoXModule


append_call_sample_docstring(
    FlaxGPTNeoXModel,
    _CHECKPOINT_FOR_DOC,
    FlaxCausalLMOutput,
    _CONFIG_FOR_DOC,
)


class FlaxGPTNeoXForCausalLMModule(nn.Module):
    config: GPTNeoXConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32


    def setup(self):
        self.gpt_neox = FlaxGPTNeoXModule(self.config, dtype=self.dtype,param_dtype=self.param_dtype)
        self.embed_out = nn.Dense(
            self.config.vocab_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )
    def _lowcoder_forward(self, input_ids, attention_mask, position_ids, deterministic, output_attentions):

        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        if position_ids is None:
            position_ids = jnp.broadcast_to(
                jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0),
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
            input_ids.shape[-1], #note this assumes we are chunked
            output_attentions,
        )
        return outputs


    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        retriever_supervision: RetrieverSupervision = None,
        lowonly_input_ids = None,
        lowonly_attention_mask = None,
        train_step: Optional[int] = None,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        upcoder_input = None,
        encoded_neighbors: Optional[EncodedNeighbors] = None,
        chunk_index: Optional[jnp.array] = None,
        retrieval_kwargs: Optional[dict] = None,
    ):

        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        else:
            assert len(input_ids.shape)==len(attention_mask.shape), "input_ids and attention_mask must have the same number of dimensions"

        if position_ids is None:
            position_ids = jnp.broadcast_to(
                jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0),
                input_ids.shape
            )
        else:
            assert len(input_ids.shape)==len(position_ids.shape), "input_ids and position_ids must have the same number of dimensions"
        should_squeeze = False
        if len(input_ids.shape)==2:
            input_ids = input_ids[None, ...]
            attention_mask = attention_mask[None, ...]
            position_ids = position_ids[None, ...]
            if retriever_supervision is not None:
                retriever_supervision =jax.tree.map(lambda x: x[None,...], retriever_supervision)
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
        if retriever_supervision is not None:
            transformer_input.update(retriever_supervision=retriever_supervision)

        def transformer(**kwargs):
            return self.gpt_neox(
                deterministic=deterministic,
                train_step=train_step,
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
            output = FlaxGPTNeoXLMOutput(
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
            output = jax.tree_map(lambda x: x.squeeze(0), output)
        return output

@add_start_docstrings(
    """
    The GPTNeoX Model transformer with a language modeling head on top.
    """,
    GPT_NEOX_START_DOCSTRING,
)
# Copied from transformers.models.gpt_neo.modeling_flax_gpt_neo.FlaxGPTNeoForCausalLM with GPTNeo->GPTNeoX
class FlaxGPTNeoXForCausalLM(FlaxGPTNeoXPreTrainedModel):
    module_class = FlaxGPTNeoXForCausalLMModule

    def prepare_inputs_for_generation(self,
                                    input_ids,
                                    max_length,
                                    attention_mask: Optional[jax.Array] = None,
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
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
        else:
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))
        chunk_index = jnp.full([batch_size,1],fill_value=63,dtype=jnp.int32)
        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
            "chunk_index":chunk_index,
            **kwargs
        }


    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        if "chunk_index" in model_kwargs:
            model_kwargs["chunk_index"] = jnp.clip(model_kwargs["chunk_index"] + 1, a_max=2*(self.config.chunk_size-1)) #will need to modify later
        if "encoded_neighbors" in model_kwargs:
            encoded_neighbors = model_kwargs["encoded_neighbors"]
            if encoded_neighbors is not None:
                model_kwargs["encoded_neighbors"] = EncodedNeighbors(
                                        neighbor_hidden_states=encoded_neighbors.neighbor_hidden_states[-1:,...], #assumes bs=1
                                        neighbor_mask=encoded_neighbors.neighbor_mask[-1:,...], #assumes bs=1
                                        nei_position_ids=encoded_neighbors.nei_position_ids[-1:,...], #assumes bs=1
                                        )

        return model_kwargs


append_call_sample_docstring(
    FlaxGPTNeoXForCausalLM,
    _CHECKPOINT_FOR_DOC,
    FlaxCausalLMOutput,
    _CONFIG_FOR_DOC,
)


class FlaxGPTNeoXRetriever(nn.Module):
    config: GPTNeoXConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype=jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]]=None

    def setup(self):
        self.preret_bidir_attention = FlaxGPTNeoXCrossAttention(self.config, dtype=self.dtype, param_dtype=self.param_dtype, precision=self.precision)
        self.preret_bi_attention_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype,
                                                     scale_init=jax.nn.initializers.constant(self.config.cca_layernorm_init_scale),
                                                     )
        self.pre_key_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps,
                                         dtype=self.dtype,
                                         scale_init=jax.nn.initializers.constant(self.config.cca_layernorm_init_scale),
                                         )
        self.key_projection = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=True,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            precision=self.precision,
        )
        self.pre_query_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps,
                                           dtype=self.dtype,
                                           scale_init=jax.nn.initializers.constant(self.config.cca_layernorm_init_scale),
                                           )
        self.query_projection = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=True,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            precision=self.precision,
        )
        self.fill_value = self.config.retriever_fill_value
        self.n_skip_chunks = self.config.max_position_embeddings//self.config.chunk_size
        self.num_neighbors = self.config.num_neighbors
        self.threshold_nei_scores = self.config.threshold_nei_scores
        self.num_sequence_chunks = self.config.max_position_embeddings//self.config.chunk_size
        self.learned_margin = self.param('learned_margin', jax.nn.initializers.constant(0), (1,))


        if self.config.ss_schedule_steps is not None and \
                        self.config.scheduled_sampling_max_prob is not None \
                        and self.config.scheduled_sampling_min_prob is not None \
                            and self.has_rng("dropout"):
            self.ss_rng = self.make_rng("dropout")
            self.scheduled_sampling_schedule_fn = m1_cosine_decay_schedule(decay_steps=self.config.ss_schedule_steps,
                                                                            min_value=self.config.scheduled_sampling_min_prob,
                                                                            max_value=self.config.scheduled_sampling_max_prob)
        else:
            self.scheduled_sampling_schedule_fn = None

    @partial(
        nn.vmap,
        in_axes=(0, None, None),
        out_axes=0,
        variable_axes={'params': None, 'intermediates': 0, "cache" : 0},
        split_rngs={'dropout': True, "params": False},
    )
    def compute_query_scores(self, encoded_output, n_skip_chunks, num_neighbors):
        query_based_scores = jnp.einsum('qd,kd->qk', encoded_output.query_chunks,
                                                        encoded_output.key_chunks,
                                                        precision=self.precision)

        if n_skip_chunks>0:
            chunk_mask = encoded_output.chunk_mask
            segment_mask = create_segment_mask(query_based_scores.shape[0], n_skip_chunks)
            chunk_mask &= segment_mask
        else:
            chunk_mask = jnp.ones_like(query_based_scores).astype(bool)
        query_score_based_idx = topk_chunks(query_based_scores, num_candidates=num_neighbors, where=chunk_mask)
        return query_based_scores, query_score_based_idx, chunk_mask

    @partial(
        nn.vmap,
        in_axes=(0, 0),
        out_axes=0,
        variable_axes={'params': None, 'intermediates': 0, "cache" : 0},
        split_rngs={'dropout': True, "params": False},
    )
    def apply_scaling(self, scores, chunk_mask):
        scaled_scores = ranksigrelu(scores/self.config.score_temp,
                                       chunk_mask,
                                        margin=jax.nn.softplus(self.learned_margin),
                                        offset=self.config.atsc_margin_min)

        return scaled_scores

    @partial(
        nn.vmap,
        in_axes=(0, 0, 0),
        out_axes=0,
        variable_axes={'params': None, 'intermediates': 0, "cache" : 0},
        split_rngs={'dropout': True, "params": False},
    )
    def batch_take_along_axis(self, scaled_scores, chunk_mask, idxs):
        idx_scores = jnp.take_along_axis(scaled_scores, idxs, axis=-1)
        idx_mask = jnp.take_along_axis(chunk_mask, idxs, axis=-1)
        return idx_scores, idx_mask



    def __call__(
        self,
        encoded_output,
        retriever_supervision: RetrieverSupervision = None,
        train_step: Optional[int] = None,
        deterministic: bool = True,
        append_next_chunk:Optional[bool] = None,
        n_skip_chunks: Optional[int] = None,
        num_neighbors: Optional[int] = None,
    ):
        if append_next_chunk is None:
            append_next_chunk = self.config.append_next_chunk
        if n_skip_chunks is None:
            n_skip_chunks = self.n_skip_chunks
        if num_neighbors is None:
            num_neighbors = self.num_neighbors

        query_based_scores, query_score_based_idx, chunk_mask = self.compute_query_scores(encoded_output, n_skip_chunks, num_neighbors)
        scaled_scores = self.apply_scaling(query_based_scores, chunk_mask)
        query_att_scores, query_neighbor_mask = self.batch_take_along_axis(scaled_scores, chunk_mask, query_score_based_idx)

        self.sow('intermediates', f'query_att_scores', query_att_scores)
        if retriever_supervision is not None:
            ret_loss_obj = self.compute_retriever_loss(query_based_scores,
                                                        retriever_supervision,
                                                        chunk_mask)
            target_score_based_idx = ret_loss_obj.target_score_based_idx
            aux_loss = ret_loss_obj.aux_loss
            ret_metrics = ret_loss_obj.ret_metrics
            target_att_scores, target_neighbor_mask = self.batch_take_along_axis(scaled_scores, chunk_mask, ret_loss_obj.target_score_based_idx)
            if self.config.use_allowed_tar_mask:
                target_neighbor_mask = ret_loss_obj.target_neighbor_mask #?????

            self.sow('intermediates', f'target_att_scores', target_att_scores)

            top_nei_idx, nei_mask, att_scores = self.apply_scheduled_sampling(
                                query_score_based_idx=query_score_based_idx,
                                chunk_mask=query_neighbor_mask,
                                target_score_based_idx=target_score_based_idx,
                                target_neighbor_mask=target_neighbor_mask,
                                train_step=train_step,
                                query_att_scores=query_att_scores,
                                target_att_scores=target_att_scores,
                                deterministic=deterministic)
        else:
            top_nei_idx, nei_mask, att_scores = query_score_based_idx, query_neighbor_mask, query_att_scores
            aux_loss = None
            ret_metrics = {}


        att_scores = jnp.where(att_scores>0, att_scores, 0)
        if self.config.stop_grad_trick:
            att_scores = att_scores - jax.lax.stop_gradient(att_scores)


            # if self.config.log_debug_metrics and not self.is_initializing():
            #     self.sow('intermediates', f'nei_mask', nei_mask)
            #     self.sow('intermediates', f'pre_nei_mask', pre_nei_mask)
            #     self.sow('intermediates', f'neighbor_attention_mask', neighbor_attention_mask)
        neighbor_hidden_states, neighbor_mask, nei_position_ids, att_scores = self.select_nei(encoded_output, top_nei_idx, nei_mask, att_scores, num_neighbors, append_next_chunk)

        if aux_loss is not None:
            if self.config.aux_scale>0:
                loss_scale = self.config.aux_scale
            else:
                loss_scale = 0
                aux_loss = jax.lax.stop_gradient(aux_loss)
        else:
            loss_scale = None
        return FlaxGPTNeoXRetrieverNeighborOutput(aux_loss=aux_loss,
                                              neighbor_hidden_states=neighbor_hidden_states,
                                              loss_scale=loss_scale,
                                              neighbor_mask=neighbor_mask,
                                              retrieval_metrics=ret_metrics,
                                              att_scores=att_scores,
                                              encoded_output=encoded_output,
                                              nei_position_ids=nei_position_ids,
                                              )

    @partial(
        nn.vmap,
        in_axes=(0, 0, 0, 0, None, None),
        out_axes=0,
        variable_axes={'params': None, 'intermediates': 0, "cache" : 0},
        split_rngs={'dropout': True, "params": False},
    )
    def select_nei(self, encoded_output, top_nei_idx, nei_mask, att_scores, num_neighbors, append_next_chunk):
        cand_hidden_states = encoded_output.encoded_hidden_states
        cand_attention_mask = encoded_output.attention_mask
        if not self.config.apply_refactor:
            chunk_size = self.config.chunk_size
            num_document_chunks = top_nei_idx.shape[0]
            shifted_hidden_states = jnp.pad(cand_hidden_states[1:,...],((0,1),(0,0),(0,0)))
            curr_neighbor_hidden_states = cand_hidden_states[top_nei_idx.reshape(-1)]
            next_neighbor_hidden_states = shifted_hidden_states[top_nei_idx.reshape(-1)]
            neighbor_hidden_states = jnp.concatenate((curr_neighbor_hidden_states, next_neighbor_hidden_states), axis=-2)
            neighbor_hidden_states = einops.rearrange(neighbor_hidden_states, '(b k) r d -> b k r d', b=num_document_chunks)
            neighbor_mask = jnp.broadcast_to(jnp.expand_dims(nei_mask, axis=-1), neighbor_hidden_states.shape[:-1])
            nei_position_ids = jnp.arange(2*chunk_size)
            nei_position_ids = jnp.broadcast_to(nei_position_ids[None, :], (num_document_chunks*num_neighbors, 2*chunk_size))

        else:
            neighbor_hidden_states,neighbor_mask,nei_position_ids =  new_lookup_neighbors(top_nei_idx,
                                                                                          cand_hidden_states,
                                                                                          cand_attention_mask,
                                                                                          append_next_chunk,
                                                                                          module=jnp,
                                                                                          nei_mask=nei_mask)





            if self.config.ret_score_in_cca:
                att_scores = jnp.broadcast_to(jnp.expand_dims(att_scores, axis=-1), neighbor_hidden_states.shape[:-1])
        return neighbor_hidden_states, neighbor_mask, nei_position_ids, att_scores

    @partial(
        nn.vmap,
        in_axes=(0, 0, None, None, None),
        out_axes=0,
        variable_axes={'params': None, 'intermediates': 0, "cache" : 0},
        split_rngs={'dropout': True, "params": False},
    )
    def preret_encode(self,
               hidden_states,
               attention_mask,
               deterministic,
               pooling_size:int,
               output_attentions: bool = False,):
        original_hidden_states_shape = hidden_states.shape
        original_attention_mask_shape = attention_mask.shape
        # print(hidden_states.shape)
        
        
        original_hidden_states, attention_mask = jax.tree_map(
                lambda x: einops.rearrange(x, 'b (l c) ... -> (b l) c ... ', c=pooling_size), 
                (hidden_states, attention_mask)
                ) # add a chunk dimension
        #1. apply bi-dir attention 
        preret_bi_output  = self.preret_bidir_attention(
                                                self.preret_bi_attention_norm(original_hidden_states),
                                                attention_mask=attention_mask,
                                                deterministic=deterministic,
                                                output_attentions=output_attentions)
        encoded_hidden_states = preret_bi_output[0] + original_hidden_states
        
        #2. pool
        pooled_hidden_states = encoded_hidden_states.mean(axis=-2)
        
        #3. project to query chunks and key chunks
        key_chunks = self.key_projection(self.pre_key_norm(pooled_hidden_states))
        query_chunks = self.query_projection(self.pre_query_norm(pooled_hidden_states))
        chunk_mask = attention_mask.astype(bool).any(-1)[...,None]
        if chunk_mask.shape[0]!=pooled_hidden_states.shape[0]:
            chunk_mask = chunk_mask[:pooled_hidden_states.shape[0],...]
            
        # nei_pos = jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0, a_max=2*self.config.chunk_size-1)
        original_hidden_states = original_hidden_states.reshape(original_hidden_states_shape)
        attention_mask = attention_mask.reshape(original_attention_mask_shape)
            
        key_chunks = key_chunks/jnp.linalg.norm(key_chunks,axis=-1,keepdims=True)
        query_chunks = query_chunks/jnp.linalg.norm(query_chunks,axis=-1,keepdims=True)
        
        return FlaxGPTNeoXRetrieverEncodedOutput(
                                            original_hidden_states=original_hidden_states,
                                            encoded_hidden_states=encoded_hidden_states,
                                            attention_mask=attention_mask,
                                            key_chunks=key_chunks,
                                            query_chunks=query_chunks,
                                            chunk_mask=chunk_mask,
                                            preret_attention=preret_bi_output[1:],
                                            # nei_position_ids=nei_pos,
                                            )

    def apply_scheduled_sampling(self,
                              query_score_based_idx,
                              chunk_mask,
                              target_score_based_idx,
                              target_neighbor_mask, 
                              train_step,
                              query_att_scores,
                              target_att_scores,
                              deterministic):
        if deterministic or self.is_initializing() or target_score_based_idx is None or self.scheduled_sampling_schedule_fn is None:
            top_nei_idx,top_nei_mask, top_att_scores = query_score_based_idx, chunk_mask, query_att_scores
        else:
            if self.config.lowres_ss:
                # n_doc_chunks = 
                selector = jax.random.bernoulli(key=self.ss_rng,
                                    p=self.scheduled_sampling_schedule_fn(train_step if not self.is_initializing() else 1),
                                    shape=tuple(query_score_based_idx.shape[:-1])+(1,)
                                    # shape=(n_doc_chunks,1)
                                    ).astype(bool)
                top_nei_idx = jnp.where(selector, query_score_based_idx, target_score_based_idx)
                top_nei_mask = jnp.where(selector, chunk_mask, target_neighbor_mask)
                top_att_scores = jnp.where(selector, query_att_scores, target_att_scores)
            else:
                rv = jax.random.bernoulli(key=self.ss_rng,
                                    p=self.scheduled_sampling_schedule_fn(train_step if not self.is_initializing() else 1),
                                    shape=()) #this is a boolean of shape [1]
                top_nei_idx, top_nei_mask, top_att_scores = jax.lax.cond(rv,
                                        (), lambda args: (query_score_based_idx, chunk_mask, query_att_scores),
                                        (),  lambda args: (target_score_based_idx, target_neighbor_mask, target_att_scores)
                                        )
        return top_nei_idx, top_nei_mask, top_att_scores

    @partial(
        nn.vmap,
        in_axes=(0,0,0),
        out_axes=0,   
        variable_axes={'params': None, 'intermediates': 0, "cache" : 0},
        split_rngs={'dropout': True, "params": False},
    )
    def compute_retriever_loss(self, raw_query_scores, retriever_supervision, chunk_mask): 
        def f(x):
            return x.reshape((-1,self.config.num_scored_neighbors))
        retriever_supervision = jax.tree_map(f, retriever_supervision)
        
        nei_idx = retriever_supervision.nei_idx #[num_sequence_chunks, num_scored_neighbors]
        nei_scores = retriever_supervision.nei_scores
        
        raw_target_scores = create_target_scores(raw_query_scores, nei_idx, nei_scores, fill_value=self.fill_value)
        raw_target_scores_wz = create_target_scores(raw_query_scores, nei_idx, nei_scores, fill_value=0)
        
        threshold_mask = self.threshold_nei_scores<raw_target_scores
        allowed_neighbor_mask = combine_masks(threshold_mask, chunk_mask,dtype=bool) #allowed neighbors
        
        pairs_diff = -compute_pairs(raw_query_scores, operator.sub)
        raw_target_scores = raw_target_scores/self.config.score_temp
        pairs_diff = pairs_diff/self.config.score_temp
        pair_loss = jax.nn.sigmoid(pairs_diff)
            
        
        #one of the scores needs to be above the threshold for the pair to be valid
        valid_pairs =  combine_masks(compute_pairs(raw_target_scores, lambda x,y: x>y),
                                    compute_pairs(threshold_mask, lambda x, y: x),
                                    compute_pairs(chunk_mask, operator.and_)
                                    )
        any_mask = combine_masks(threshold_mask.any(axis=-1), chunk_mask.any(axis=-1),dtype=bool)
        ndcg_lambda = compute_ndcg_lambda(raw_query_scores, raw_target_scores_wz,
                                           query_mask=chunk_mask,
                                           target_mask=allowed_neighbor_mask,)
    
        pair_loss  = jnp.where(valid_pairs, pair_loss, 0.0)

        metrics = compute_retrieval_metrics(raw_query_scores, raw_target_scores_wz,
                                        query_mask=chunk_mask,
                                        target_mask=allowed_neighbor_mask)
        metrics = jax.tree_map(lambda x: x.mean(), metrics)
        
        
        per_chunk_pair_loss = (ndcg_lambda*pair_loss).sum(axis=-1)
        
        
        raw_aux_loss = jnp.where(any_mask, per_chunk_pair_loss, 0.0).sum()
        
        
        target_idx = topk_chunks(raw_target_scores, num_candidates=self.num_neighbors, where=chunk_mask)
        target_nei_mask = jnp.take_along_axis(allowed_neighbor_mask, target_idx,axis=-1)
        return FlaxGPTNeoXRetrieverLossOutput(
                aux_loss=(raw_aux_loss, valid_pairs.sum(), ),
                target_neighbor_mask=target_nei_mask,
                target_score_based_idx=target_idx,
                ret_metrics=metrics,
                )
        
#loss is calculated as lm_loss + (raw_aux_loss/valid_pairs.sum())* self.get_loss_scale(train_step)


@jax.vmap
def _ranksigrelu(scores, mask, pair_mask, weights=None):
  score_i = jnp.expand_dims(scores, axis=-1)
  score_j = jnp.expand_dims(scores, axis=-2)
  score_pairs = jax.nn.sigmoid((score_i-score_j))
  if weights is not None:
    score_pairs = weights.reshape(score_pairs.shape)*score_pairs
  x = jnp.sum(score_pairs, axis=-1, where=pair_mask.reshape(score_pairs.shape), initial=0.0)
  x = x - jax.lax.stop_gradient(jnp.max(x,axis=-1,where=mask,initial=0.0))
  return jnp.where(jnp.expand_dims(mask.any(-1),axis=-1), x, 0.0)

@gin.configurable
def ranksigrelu(
    scores, mask, pair_mask=None, weights=None, margin=0, offset=1.0,substract_max=True,
    ):
  mask = mask.astype(bool)
  if pair_mask is None:
    pair_mask = compute_pairs(mask, operator.and_)
  if weights is None:
    weights = jnp.ones_like(pair_mask)
  if substract_max:
      scores = scores - jax.lax.stop_gradient(jnp.max(scores,axis=-1,where=mask,initial=0.0,keepdims=True))
  scores = _ranksigrelu(scores, mask, pair_mask, weights)
  
  scores = scores + offset+margin
  scores = jax.nn.softplus(scores)
  scores = jnp.where(scores>0, scores, 0)
  return scores



def m1_cosine_decay_schedule(
    decay_steps: int,
    min_value:float,
    max_value:int,
    exponent: float = 1.0,
):
  if not decay_steps > 0:
    raise ValueError('The cosine_decay_schedule requires positive decay_steps!')

  def schedule(count):
    count = jnp.minimum(count, decay_steps)
    cosine_decay = 0.5 * (1 + jnp.cos(jnp.pi * count / decay_steps))
    decayed = 1-(cosine_decay ** exponent)
    decayed = (1 - min_value) * decayed + min_value
    return max_value*decayed

  return schedule

def topk_chunks(retriever_scores,num_candidates,*,where=None):
    @jax.vmap
    def _topk_chunks(retriever_scores):
        return (-retriever_scores).argsort()[:num_candidates] #k = num_candidates
    if where is not None:
        retriever_scores = jnp.where(where,retriever_scores,-jnp.inf)
    return _topk_chunks(retriever_scores)

def create_segment_mask(total_num_chunks,n_skip_chunks):
    @jax.vmap
    def _create_segment_mask(chunk_index):
        max_chunk = n_skip_chunks*(chunk_index//n_skip_chunks)
        return jnp.arange(total_num_chunks)<max_chunk - 2 
    return _create_segment_mask(jnp.arange(total_num_chunks))

def compute_pairs(a, op):
  """Computes pairs based on values of `a` and the given pairwise `op`.

  Args:
    a: The array used to form pairs. The last axis is used to form pairs.
    op: The binary op to map a pair of values to a single value.

  Returns:
    A :class:`jax.Array` with the same leading dimensions as `a`, but with the
    last dimension expanded so it includes all pairs `op(a[..., i], a[..., j])`
  """
  a_i = jnp.expand_dims(a, -1)
  a_j = jnp.expand_dims(a, -2)
  result_shape = jnp.broadcast_shapes(a_i.shape, a_j.shape)
  result = jnp.broadcast_to(op(a_i, a_j), result_shape)
  out_shape = tuple(result.shape[:-2]) + (result.shape[-2] * result.shape[-1],)
  return jnp.reshape(result, out_shape)
import rax

def compute_retrieval_metrics(scores_pred, scores,query_mask,target_mask):
    scores_pred = jnp.where(query_mask, scores_pred,-jnp.inf)
    scores = jnp.where(target_mask, scores, 0.0)
    def recall(n):
        res = rax.recall_metric(scores=scores_pred,
                labels=(scores>0).astype(np.float32),
                topn=n,
                reduce_fn=None,
                where= query_mask)
        res_mask = combine_masks(
                        jnp.isfinite(res),
                        jnp.any(query_mask,axis=-1))
        return res.mean(where=res_mask)
    return {f"recall@{i}":recall(i) for i in [2,5,10,20]}

@jax.vmap
def compute_ndcg_lambda(scores_pred, scores, query_mask, target_mask):
    """
    Compute the NDCG delta matrix for given scores and predicted scores.
    
    Args:
        score_pred (numpy.array): predicted scores
        score (numpy.array): true scores
        
    Returns:
        numpy.array: NDCG delta matrix
    """
    # Get the descending order of indices based on scores
    scores_pred = jnp.where(query_mask, scores_pred,-jnp.inf)
    scores = jnp.where(target_mask, scores, 0.0)

    argsort_score = jnp.argsort(scores)[::-1]
    argsort_score_pred = jnp.argsort(scores_pred)[::-1]

    # Calculate rank plus one for true scores and predicted scores
    rank_plus_one = jnp.argsort(argsort_score) + 2
    rank_plus_one_pred = jnp.argsort(argsort_score_pred) + 2

    # Calculate the numerator, which is the same for both IDCG and DCG
    numerator = 2 ** scores - 1

    # Calculate the denominators for IDCG and DCG
    idcg_denominator = jnp.log2(rank_plus_one)
    dcg_denominator = jnp.log2(rank_plus_one_pred)

    # Calculate IDCG and DCG
    idcg = numerator / idcg_denominator

    # Calculate the difference between numerators
    numerator_ij = numerator[:, None] - numerator

    # Calculate the difference between DCG denominators
    dcg_denominator_ij = dcg_denominator[:, None] - dcg_denominator
    # Calculate the NDCG delta matrix
    ndcg_delta_ij = jnp.abs((numerator_ij * dcg_denominator_ij) / jnp.maximum(jnp.sum(idcg), 0.001)).reshape(-1)
    return jnp.where(jnp.isfinite(ndcg_delta_ij), ndcg_delta_ij, 0.0)

def make_cross_mask(q_len, kv_len, extra_batch_dims: int = 0, dtype=bool):
    source_idxs = jnp.arange(kv_len-q_len, kv_len, dtype=jnp.int32)
    target_idxs = jnp.arange(kv_len, dtype=jnp.int32)
    return nn.make_attention_mask(source_idxs,target_idxs, lambda x,y:x>=y,
                                  extra_batch_dims=extra_batch_dims, dtype=dtype)
    
    
    