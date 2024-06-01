import os
import time

import flax
import torch
import transformers

from EasyLM.models.neox.neox_model import FlaxGPTNeoXForCausalLM
from EasyLM.models.neox.neox_model_torch import GPTNeoXForCausalLM
from EasyLM.models.neox.rpt_torch_utils import create_prepare_inputs, batch_encode_memories, batch_lookup_neighbors, \
    add_batch_index, collate_fn
from EasyLM.models.rpt.torch_utils import reshape_for_vmap

if "DEBUG" in os.environ:
    #   os.system('kill -9 $(lsof -t -i tcp:5678)')
    time.sleep(2)
    print("Waiting for debugger attach",flush=True)
    import debugpy
    debugpy.listen(5678)
    debugpy.wait_for_client()

import numpy as np

from transformers import GenerationConfig, LogitsProcessorList, TemperatureLogitsWarper
import pickle
import base64
from more_itertools import chunked

from dataclasses import dataclass

@dataclass
class Config:
    seed: int
    initialize_jax_distributed: bool
    mesh_dim: str
    dtype: str
    input_length: int
    lowcoder_batch_size: int
    seq_length: int
    top_k: int
    top_p: float
    do_sample: bool
    num_beams: int
    add_bos_token: bool
    model_config_path: str
    load_checkpoint: str
    num_neighbors: int
    keep_fields: str
    split_by_newline: bool
    port: int
    pre_compile: str
    lm_server_batch_size: int
    return_empty: bool

def load_model(config):
    hf_model = GPTNeoXForCausalLM.from_pretrained(pretrained_model_name_or_path='/Users/shahar.satamkar/Desktop/Masters/RPT/EasyLM/models/neox/neox_model_torch')

    prefix_tokenizer = hf_model.config.get_tokenizer(truncation_side='left', padding_side='left')
    tokenizer = hf_model.config.get_tokenizer(truncation_side='right', padding_side='right')

    """
    model_cfg = GPTNeoXConfig.load_config(config.model_config_path)
    model_cfg.update(parse_overrides(config.config_override))
    prefix_tokenizer = model_cfg.get_tokenizer(truncation_side='left', padding_side='left')
    tokenizer = model_cfg.get_tokenizer(truncation_side='right', padding_side='right')

    with jax.default_device(jax.devices("cpu")[0]):
        if config.model_config_path.startswith("hf::"):

            hf_model = FlaxGPTNeoXForCausalLM.from_pretrained(config.model_config_path.split("::")[1],
                    from_pt=True,
                    config=model_cfg,
                    # seed=config.seed,
                    )
            params = dict(params=hf_model.params)
            
        else:
        
            _, params = StreamingCheckpointer.load_trainstate_checkpoint(
                config.load_checkpoint, disallow_trainstate=True
            )
            params = params["params"]

            hf_model = FlaxGPTNeoXForCausalLM(
                model_cfg,
                input_shape=(1, config.seq_length),
                seed=config.seed,
                _do_init=False
            )
        rng_keys = model_cfg.rng_keys()
     
    model_ps = match_partition_rules(
        GPTNeoXConfig.get_partition_rules(), params
    )
    shard_fns, _ = make_shard_and_gather_fns(
        model_ps, get_float_dtype_by_name(config.dtype)
    )
    """

    def softmax_cross_entropy_with_integer_labels(logits, labels):
        # This is like jnp.take_along_axis(jax.nn.log_softmax(...), ...) except that
        # we avoid subtracting the normalizer from all values, just from the values
        # for the correct labels.
        logits = torch.max(logits, dim=-1, keepdim=True).values
        label_logits = torch.take_along_dim(logits, labels[..., None].type(torch.long), dim=-1)[..., 0]
        log_normalizers = torch.log(torch.sum(torch.exp(logits), dim=-1))
        return log_normalizers - label_logits
        

    def forward_loglikelihood(hf_model, batch):
        input_tokens = batch['input_tokens']
        output_tokens = batch['output_tokens']
        input_mask = batch['input_mask']
        output_mask = batch['output_mask']

        logits = hf_model(input_tokens, attention_mask=input_mask, deterministic=True).logits
        loglikelihood = -softmax_cross_entropy_with_integer_labels(
            logits, output_tokens
        )
        loglikelihood = torch.sum(loglikelihood * output_mask, dim=-1)
        match_count = torch.sum(
            (torch.argmax(logits, dim=-1) == output_tokens) * output_mask,
            dim=-1
        )
        total = torch.sum(output_mask, dim=-1)
        is_greedy = match_count == total
        return loglikelihood, is_greedy


    def forward_generate(hf_model, batch, temperature):
        output = hf_model.generate(
            batch['input_tokens'],
            attention_mask=batch['attention_mask'],
            logits_processor=LogitsProcessorList(
                [TemperatureLogitsWarper(temperature)]
            ),
            generation_config=GenerationConfig(
                max_new_tokens=config.seq_length - config.input_length,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=config.do_sample,
                num_beams=config.num_beams,
                top_k=config.top_k,
                top_p=config.top_p,
            )
        ).sequences[:, batch['input_tokens'].shape[1]:]
        return output


    def forward_greedy_generate(hf_model, batch):
        output = hf_model.generate(
            batch['input_tokens'],
            attention_mask=batch['attention_mask'],
            generation_config=GenerationConfig(
                max_new_tokens=config.seq_length - config.input_length,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
                num_beams=1,
            )
        ).sequences[:, batch['input_tokens'].shape[1]:]
        return output
    

    def lowcoder_forward(hf_model, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        return hf_model.batch_lowcoder_forward(input_ids, attention_mask)



    def create_greedy():
        def apply_forward(hf_model, batch):
            def fwd(input_ids, attention_mask, encoded_neighbors):
                return hf_model.generate(input_ids,max_length=config.seq_length,
                        do_sample=False,
                        pad_token_id=0,
                        encoded_neighbors=encoded_neighbors,
                        attention_mask=attention_mask,
                        )
            fwd = torch.vmap(fwd, in_dims=(0,0,0))
            batch = reshape_for_vmap(batch)
            result = fwd(batch["input_ids"], batch["attention_mask"], batch["encoded_neighbors"])
            return result
        def forward(input_ids, attention_mask, params, encoded_neighbors):
            batch = {"input_ids": input_ids, "encoded_neighbors":encoded_neighbors, "attention_mask": attention_mask}
            output = apply_forward(params, batch)
            # TODO: torch to device
            assert len(output.shape)==3
            assert output.shape[1]==1
            output = output.squeeze(1)
            
            
            all_outputs = []
            for tokens,out in zip(list(input_ids),list(output)):
                all_outputs.append(out[len(tokens):])
            all_outputs = tokenizer.batch_decode(all_outputs, skip_special_tokens=True)
            return [x.strip() for x in all_outputs]
            
        return forward

    greedy_w_cache = create_greedy()
    prepare_inputs = create_prepare_inputs(prefix_tokenizer, input_length=config.input_length)


    def _loglikelihood(prefix_text, text):
        prefix = prefix_tokenizer(
            prefix_text,
            padding='max_length',
            truncation=True,
            max_length=config.input_length,
            return_tensors='np',
        )
        inputs = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=config.seq_length - config.input_length,
            return_tensors='np',
        )
        output_tokens = np.concatenate([prefix.input_ids, inputs.input_ids], axis=1)
        bos_tokens = np.full(
            (output_tokens.shape[0], 1), tokenizer.bos_token_id, dtype=np.int32
        )
        input_tokens = np.concatenate([bos_tokens, output_tokens[:, :-1]], axis=-1)
        input_mask = np.concatenate(
            [prefix.attention_mask, inputs.attention_mask], axis=1
        )
        if config.add_bos_token:
            bos_mask = np.ones_like(input_mask[:, :1])
        else:
            bos_mask = np.zeros_like(input_mask[:, :1])

        input_mask = np.concatenate([bos_mask, input_mask[:, :-1]], axis=1)
        output_mask = np.concatenate(
            [np.zeros_like(prefix.attention_mask), inputs.attention_mask], axis=1
        )
        batch = dict(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_mask=input_mask,
            output_mask=output_mask,
        )
        loglikelihood, is_greedy = forward_loglikelihood(hf_model, batch)
        # TODO: Move to torch device
        return loglikelihood, is_greedy


    def _loglikelihood_rolling(text):
        inputs = tokenizer(
            text,
            padding='longest',
            truncation=False,
            max_length=np.iinfo(np.int32).max,
            return_tensors='np',
        )
        batch_size = inputs.input_ids.shape[0]
        output_tokens = inputs.input_ids
        attention_mask = inputs.attention_mask

        if output_tokens.shape[1] < config.seq_length:
            padding_length = config.seq_length - output_tokens.shape[1]
            pad_tokens = np.full(
                (batch_size, padding_length), tokenizer.pad_token_id, dtype=np.int32
            )
            output_tokens = np.concatenate([output_tokens, pad_tokens], axis=-1)
            pad_mask = np.zeros(
                (batch_size, padding_length), dtype=inputs.attention_mask.dtype
            )
            attention_mask = np.concatenate([attention_mask, pad_mask], axis=-1)

        bos_tokens = np.full(
            (batch_size, 1), tokenizer.bos_token_id, dtype=np.int32
        )
        input_tokens = np.concatenate([bos_tokens, output_tokens[:, :-1]], axis=-1)
        bos_mask = np.ones((batch_size, 1), dtype=inputs.attention_mask.dtype)
        total_seq_length = output_tokens.shape[1]

        total_loglikelihood = 0.0
        total_is_greedy = True
        # Sliding window
        for i in range(0, total_seq_length, config.seq_length):
            # Last window
            if i + config.seq_length > total_seq_length:
                last_output_mask = np.copy(attention_mask[:, -config.seq_length:])
                last_output_mask[:, :i - total_seq_length] = 0.0

                batch = dict(
                    input_tokens=input_tokens[:, -config.seq_length:],
                    output_tokens=output_tokens[:, -config.seq_length:],
                    input_mask=attention_mask[:, -config.seq_length:],
                    output_mask=last_output_mask,
                )

            # Normal window
            else:
                batch = dict(
                    input_tokens=input_tokens[:, i:i + config.seq_length],
                    output_tokens=output_tokens[:, i:i + config.seq_length],
                    input_mask=attention_mask[:, i:i + config.seq_length],
                    output_mask=attention_mask[:, i:i + config.seq_length],
                )

            loglikelihood, is_greedy = forward_loglikelihood(hf_model, batch)
            # TODO: torch to device

            total_loglikelihood += loglikelihood
            total_is_greedy = np.logical_and(is_greedy, total_is_greedy)

        return total_loglikelihood, total_is_greedy


    def _generate(text, temperature):
        if config.return_empty:
            return ["" for _ in text]
        
        inputs = prefix_tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=config.input_length,
            return_special_tokens_mask=True,
            return_tensors='np',
        )
        input_tokens = inputs.input_ids
        input_mask = inputs.attention_mask
        if config.add_bos_token:
            bos_position = np.maximum(inputs["special_tokens_mask"].sum(-1)-1,0)[:,None]
            np.put_along_axis(input_tokens, bos_position, tokenizer.bos_token_id, axis=1)
            np.put_along_axis(input_mask, bos_position, 1, axis=1)
        batch = dict(
            input_tokens=input_tokens,
            attention_mask=input_mask,
        )
        print(batch)
        output, sharded_rng = forward_generate(
            hf_model, batch, temperature
        )
        # TODO: to torch device
        output_text = []
        for text in list(tokenizer.batch_decode(output)):
            if tokenizer.eos_token in text:
                text = text.split(tokenizer.eos_token, maxsplit=1)[0]
            output_text.append(text)

        return output_text


    def _old_greedy_until(prefix_text, until, max_length, pre_compile=False):
        
        all_outputs = []
        for pf, ut in zip(prefix_text, until):
            if isinstance(ut, str):
                ut = [ut]
            total_length = 0
            total_generated = ''

            while total_length < max_length:
                pf_tokens = tokenizer(
                    pf,
                    padding=False,
                    truncation=False,
                    max_length=np.iinfo(np.int32).max,
                    return_tensors='np',
                )
                input_tokens = pf_tokens.input_ids
                attention_mask = pf_tokens.attention_mask

                if input_tokens.shape[1] < config.input_length:
                    extra = config.input_length - input_tokens.shape[1]
                    pad_tokens = np.full(
                        (1, extra), tokenizer.pad_token_id, dtype=np.int32
                    )
                    input_tokens = np.concatenate(
                        [pad_tokens, input_tokens], axis=1
                    )
                    pad_attention = np.zeros((1, extra), dtype=attention_mask.dtype)
                    attention_mask = np.concatenate(
                        [pad_attention, attention_mask], axis=1
                    )
                elif input_tokens.shape[1] > config.input_length:
                    # if False:
                    input_tokens = input_tokens[:, -config.input_length:]
                    attention_mask = attention_mask[:, -config.input_length:]
                

                if config.add_bos_token:
                    input_tokens[:, 0] = tokenizer.bos_token_id
                    attention_mask[:, 0] = 1

                batch = dict(input_tokens=input_tokens, attention_mask=attention_mask)

                # print(batch)
                output, sharded_rng = forward_greedy_generate(
                   hf_model, sharded_rng, batch
                )
                # print(output)
                # TODO: to torch device
                # print(output)

                total_length += output.shape[1]
                output_text = tokenizer.batch_decode(output)[0]
                total_generated = total_generated + output_text
                pf = pf + output_text

                done = False
                for s in ut:
                    if s in total_generated:
                        total_generated = total_generated.split(s, maxsplit=1)[0]
                        done = True
                if done or pre_compile:
                    break

            all_outputs.append(total_generated)

        return all_outputs
    
    

    def _greedy_until(prefix_text, until, max_length, pre_compile=False):
        if config.return_empty:
            return ["" for _ in prefix_text]
        
        num_neighbors = config.num_neighbors
        split_by_newline = config.split_by_newline
        # if False:
        #     num_neighbors = 2
        #     split_by_newline=True
        #     prefix_text = [top_n(x,10) for x in prefix_text]
        # else:
        #     num_neighbors = 0
            
        disable_neighbors = config.num_neighbors == 0
        inputs = [prepare_inputs(prefix, split_by_newline) for prefix in prefix_text]
        if pre_compile:
            inputs = [[x[0],x[0]] for x in inputs]

        if disable_neighbors:
            input_ids=np.array([x[-1]["input_ids"].squeeze() for x in inputs])
            attention_mask=np.array([x[-1]["attention_mask"].squeeze() for x in inputs])
            encoded_neighbors=None
        else:
            input_ids, attention_mask, memories, _ = batch_encode_memories((lowcoder_forward, hf_model), inputs, config.lowcoder_batch_size)
            encoded_output = lowcoder_forward(hf_model, input_ids, attention_mask)
            encoded_neighbors = batch_lookup_neighbors(encoded_output.query_chunks, memories, num_neighbors, config.append_next_chunk)
            # encoded_neighbors = jax.tree.map(lambda x:x[None,:], encoded_neighbors)
        return greedy_w_cache(input_ids, attention_mask, hf_model, encoded_neighbors)

    

    def _encode(text):
        lowcoder_bs = config.lowcoder_batch_size
        n_examples = len(text)
        inputs = [prefix[:50000] for prefix in text]
        inputs = [prepare_inputs(prefix, config.split_by_newline) for prefix in text]
        inputs = [add_batch_index(x,j) for j,x in enumerate(inputs)]
        inputs = sum(inputs,[])
        keep_fields = config.keep_fields.split(",")
        if len(keep_fields)==1 and keep_fields[0]=="":
            keep_fields = []
        def format_enc(x):
            new_x = {}
            for key in x.keys():
                if key in keep_fields:
                    new_x[key] = x[key]
            return new_x
        output_dict = {}
        for batch in chunked(inputs, lowcoder_bs):
            input_ids, attention_mask = collate_fn(batch)
            encoded_output = []
            for input_id, attention_mask_id in zip(input_ids, attention_mask):
                single_encoded_output = lowcoder_forward(hf_model, input_id.unsqueeze(0), attention_mask_id.unsqueeze(0)) # TODO: Handle multi device: min_device_batch=max(lowcoder_bs//jax.local_device_count(),1))
                encoded_output.append(single_encoded_output)

            # TODO: Torch to device
            #encoded_output = tree_unstack(encoded_output)
            # TODO: Handle tree unpacking from torch
            for batch_el, enc in zip(batch, encoded_output):
                enc = dict(enc)
                enc["input_ids"] = batch_el["input_ids"]
                batch_index = batch_el["batch_index"]
                window_index = batch_el["window_index"]
                if batch_index not in output_dict:
                    output_dict[batch_index] = []
                if len(keep_fields)>0:
                    enc = format_enc(enc)
                output_dict[batch_index].append((window_index, enc))
        encoded_output = [
            [enc for _, enc in sorted(output_dict[i], key=lambda x: x[0])]
            for i in range(n_examples)
        ]
        encoded_output = [pickle.dumps(enc) for enc in encoded_output]
        encoded_output = [base64.b64encode(enc).decode() for enc in encoded_output]
        return encoded_output

    return _loglikelihood, _loglikelihood_rolling, _generate, _greedy_until, _encode, _old_greedy_until




# batch.keys()
# dict_keys(['attention_mask', 'input_tokens', 'loss_masks', 'target_tokens', 'lowonly_input_ids', 'lowonly_attention_mask'])
# special variables
