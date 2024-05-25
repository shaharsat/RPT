from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import einops
from transformers.utils import ModelOutput
import torch

@dataclass
class EncodedNeighbors(ModelOutput):
    neighbor_hidden_states:torch.Tensor = None
    neighbor_mask:torch.Tensor = None
    chunk_index:torch.Tensor = None
    att_scores:torch.Tensor = None
    nei_position_ids:torch.Tensor = None

@dataclass
class GPTNeoXRetrieverEncodedOutput(ModelOutput):
    original_hidden_states: torch.Tensor = None
    encoded_hidden_states: torch.Tensor = None
    attention_mask: torch.Tensor = None
    key_chunks: torch.Tensor = None
    query_chunks: torch.Tensor = None
    chunk_mask: torch.Tensor = None
    preret_attention: Optional[torch.Tensor] = None
@dataclass
class GPTNeoXRetrieverNeighborOutput(ModelOutput):
    aux_loss: torch.Tensor = None
    loss_scale: torch.Tensor = None
    neighbor_hidden_states: torch.Tensor = None
    neighbor_mask: torch.Tensor = None
    retrieval_metrics: Optional[Dict[str, torch.Tensor]] = None
    att_scores: torch.Tensor = None
    encoded_output: Optional[GPTNeoXRetrieverEncodedOutput] = None
    nei_position_ids: Optional[torch.Tensor] = None

@dataclass
class FlaxGPTNeoXRetrieverEncodedOutput(ModelOutput):
    original_hidden_states: torch.Tensor = None
    encoded_hidden_states: torch.Tensor = None
    attention_mask: torch.Tensor = None
    key_chunks: torch.Tensor = None
    query_chunks: torch.Tensor = None
    chunk_mask: torch.Tensor = None
    preret_attention: Optional[torch.Tensor] = None

@dataclass
class GPTNeoXLMOutput(ModelOutput):
    logits: torch.Tensor = None
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None
    upcoder_hidden_states: Optional[Tuple[torch.Tensor]] = None
    upcoder_attentions: Optional[Tuple[torch.Tensor]] = None
    cross_attentions: Optional[Tuple[torch.Tensor]] = None
    lowcoder_last_hidden_state: Optional[torch.Tensor] = None
    lowcoder_hidden_states: Optional[Tuple[torch.Tensor]] = None
    lowcoder_attentions: Optional[Tuple[torch.Tensor]] = None
    retriever_output: GPTNeoXRetrieverNeighborOutput = None
    retriever_input: Optional[torch.Tensor] = None
    
def new_lookup_neighbors(top_nei_idx, cand_hidden_states, cand_attention_mask, append_next_chunk, module,
                         nei_mask=None):
    cand_attention_mask = cand_attention_mask.reshape(cand_hidden_states.shape[:-1])
    num_document_chunks = top_nei_idx.shape[0]

    curr_neighbor_hidden_states = cand_hidden_states[top_nei_idx.reshape(-1)]
    curr_neighbor_attention_mask = cand_attention_mask[top_nei_idx.reshape(-1)]
    if append_next_chunk:
        shifted_hidden_states = module.pad(cand_hidden_states[1:, ...], ((0, 1), (0, 0), (0, 0)))
        shifted_attention_mask = module.pad(cand_attention_mask[1:, ...], ((0, 1), (0, 0)))
        next_neighbor_hidden_states = shifted_hidden_states[top_nei_idx.reshape(-1)]
        next_neighbor_attention_mask = shifted_attention_mask[top_nei_idx.reshape(-1)]
        neighbor_hidden_states = module.concatenate((curr_neighbor_hidden_states, next_neighbor_hidden_states), axis=-2)
        neighbor_attention_mask = module.concatenate((curr_neighbor_attention_mask, next_neighbor_attention_mask),
                                                     axis=-1)
    else:
        neighbor_hidden_states = curr_neighbor_hidden_states
        neighbor_attention_mask = curr_neighbor_attention_mask

    neighbor_hidden_states = einops.rearrange(neighbor_hidden_states, '(b k) r d -> b k r d', b=num_document_chunks)
    neighbor_attention_mask = einops.rearrange(neighbor_attention_mask, '(b k) r -> b k r', b=num_document_chunks)
    bkr_shape = tuple(neighbor_hidden_states.shape[:-1])
    if nei_mask is not None:
        pre_nei_mask = module.broadcast_to(module.expand_dims(nei_mask, axis=-1), bkr_shape)
        neighbor_mask = neighbor_attention_mask.astype(bool) & pre_nei_mask.astype(bool)
    else:
        neighbor_mask = neighbor_attention_mask.astype(bool)
    nei_position_ids = neighbor_mask.astype(module.int32).cumsum(axis=-1) - 1

    return neighbor_hidden_states, neighbor_mask, nei_position_ids