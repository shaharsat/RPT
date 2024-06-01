from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import einops
import numpy as np
from more_itertools.more import chunked
import tqdm
from transformers.utils import ModelOutput
import torch

from EasyLM.sliding_window import sliding_window


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
class GPTNeoXRetrieverEncodedOutput(ModelOutput):
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

@dataclass
class GPTNeoXModelOutput(ModelOutput):
    last_hidden_state: torch.Tensor = None
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


def batch_lookup_neighbors(query_chunks, memories, num_neighbors, append_next_chunk):
    # mask_shape = (1,1, chunk_size*2)
    query_chunks = list(query_chunks)
    bs = len(query_chunks)
    memories = list(memories)

    neighbor_hidden_states, neighbor_mask, nei_position_ids \
        = zip(*[lookup_neighbors(q, mem, num_neighbors, append_next_chunk) for q, mem in zip(query_chunks, memories)])
    neighbor_hidden_states = np.array(neighbor_hidden_states)
    neighbor_mask = np.array(neighbor_mask)
    nei_position_ids = np.array(nei_position_ids)

    # neighbor_mask = np.where(neighbor_mask[...,None],pos_mask,neg_mask)
    output = EncodedNeighbors(neighbor_hidden_states=neighbor_hidden_states,
                              neighbor_mask=neighbor_mask,
                              nei_position_ids=nei_position_ids)
    # if neighbor_hidden_states.shape[1]==1:
    # output = jax.tree.map(lambda x: x.squeeze(axis=1), output)
    return output


def lookup_neighbors(query_chunks, memories, num_neighbors, append_next_chunk):
    key_chunks = np.concatenate([x.key_chunks for x in memories], axis=0)
    key_chunks = key_chunks.reshape(-1, key_chunks.shape[-1])

    value_chunks = np.concatenate([x.encoded_hidden_states for x in memories], axis=0)
    value_chunks = value_chunks.reshape((-1,) + value_chunks.shape[-2:])

    # chunk_mask = np.concatenate([x.chunk_mask for x in memories], axis=0)
    attention_mask = np.concatenate([x.attention_mask for x in memories], axis=0)
    # chunk_mask = chunk_mask.reshape(-1)

    scores = query_chunks @ key_chunks.T
    if len(scores.shape) == 1:
        scores = scores[None, :]
    top_results = (-scores).argsort(axis=-1)[:, :num_neighbors]
    # top_results_p1 = np.clip(top_results+1,a_min=0, a_max=value_chunks.shape[0]-1)
    # neighbor_mask = chunk_mask[top_results]
    # neighbor_hidden_states = np.concatenate([value_chunks[top_results],value_chunks[top_results_p1]],axis=-2)


def create_prepare_inputs(prefix_tokenizer, input_length):
    def tokenize(x):
        batch = prefix_tokenizer(x, return_tensors='np')["input_ids"][0]
        batch = sliding_window(batch,
                               width=input_length,
                               stride=input_length,
                               bos_token_id=prefix_tokenizer.eos_token_id,
                               padding_value=prefix_tokenizer.eos_token_id,
                               append_eos=True
                               )
        batch = list(batch)
        return batch[-1]

    def prepare_inputs(prefix_text, split_by_newline):
        if split_by_newline:
            prefix_text = prefix_text.split("\n\n")
            inputs = map(tokenize, prefix_text)
        else:
            inputs = prefix_tokenizer(prefix_text, return_tensors='np')
            inputs = sliding_window(inputs["input_ids"][0],
                                    width=input_length,
                                    stride=input_length,
                                    bos_token_id=prefix_tokenizer.eos_token_id,
                                    padding_value=prefix_tokenizer.eos_token_id,
                                    )
        inputs = map(rename_fields, inputs)

        inputs = list(inputs)
        return inputs

    return prepare_inputs


def rename_fields(x):
    x["input_ids"]=x.pop("input_tokens")
    x["attention_mask"] = x["attention_mask"].astype(int)
    return x


def batch_encode_memories(enc_func, windows, batch_size):
    input_ids, attention_mask, memories, window_list = zip(*[encode_memories(enc_func, x, batch_size) for x in windows])
    input_ids = np.array(input_ids)
    attention_mask = np.array(attention_mask)
    return input_ids, attention_mask, memories, window_list


def encode_memories(enc_func, windows, batch_size):
    lowcoder_forward, hf_model = enc_func
    #TODO batching: min_device_batch = max(batch_size // torch.local_device_count(), 1)
    memories = []
    window_list = []

    def iterate_batches(inputs):
        inputs, last_inputs = inputs[:-1], inputs[-1]
        for win in chunked(inputs, batch_size):
            input_ids = np.array([x["input_ids"].squeeze() for x in win])
            attention_mask = np.array([x["attention_mask"].squeeze() for x in win])
            yield win, (input_ids, attention_mask)
        yield last_inputs, (last_inputs["input_ids"].squeeze(), last_inputs["attention_mask"].squeeze())

    windows = iterate_batches(windows)
    windows = list(windows)

    for j, (win, (input_ids, attention_mask)) in enumerate(tqdm.tqdm(windows)):
        if j < len(windows) - 1:
            encoded_output = lowcoder_forward(hf_model, input_ids, attention_mask) # TODO: Handle batching, min_device_batch=min_device_batch)
            memories.append(encoded_output)
        window_list.extend(win)

    return input_ids, attention_mask, memories, window_list

def add_batch_index(x,j):
    for window_index,el in enumerate(x):
        el["batch_index"]=j
        el["window_index"]=window_index
    return x

def collate_fn(batch):
    input_ids = torch.Tensor([x["input_ids"].squeeze() for x in batch])
    attention_mask = torch.Tensor([x["attention_mask"].squeeze() for x in batch])
    return input_ids, attention_mask