
import jax
import gin
from EasyLM.jax_utils import cross_entropy_loss_and_accuracy
import numpy as np
import jax.numpy as jnp
from typing import List, Optional
import optax
import dataclasses
from optax._src import base
from optax._src import transform
from optax._src import numerics
from optax._src import combine
from optax._src import alias
import flax
from flax.traverse_util import path_aware_map
import copy
from flax.traverse_util import flatten_dict,unflatten_dict


def parse_overrides(config_override: List[str]):
    config_override = dict(map(lambda x:x.split("="), config_override))
    config_override = {k:gin.config.parse_value(v) for k,v in config_override.items() }
    return config_override


@gin.configurable
@dataclasses.dataclass
class TrainConfig:
    init_from_pretrained:bool = False
    size:str = gin.REQUIRED
    milestones: List[int] = dataclasses.field(default_factory=lambda:[])
    init_lr:float = 0.0
    lr:float=1e-4
    lrmod:float = 1.0
    lr_warmup_steps:int=1000
    lr_decay_steps:int = 10000
    end_lr: Optional[float] = None
    b1: float = 0.9
    b2: float = 0.95
    clip_gradient:float = 1.0
    weight_decay:float = 0.01
    bf16_momentum:bool = False
    adamw_eps:float = 1e-8 
    adabelief_eps:float = 1e-16
    eps_root:float = 1e-16 
    optimizer_name:str = "adamw"
    pt_mult:float = -1
    accumulate_gradient_steps:int = 1
    flatten_optimizer:bool = True
    unroll_dataloader:bool = False
    independant_wd:bool = False
    dataset_type: str = "zstd::"
    a_param_modifier: Optional[float] = None
    
    

    def create_optimizer(self, hidden_size):
        
        # if self.independant_wd:
            #See section 3.2.2 of https://arxiv.org/pdf/2309.14322.pdf
            # if lr= 5e-4 and wd=0.01 then Loshchilov and Hutter WD is 5e-4*0.01 = 5.0e-6
            # learning_rate_schedule = optax.warmup_cosine_decay_schedule(
            #     init_value=0.0002,
            #     peak_value=1,
            #     warmup_steps=self.lr_warmup_steps,
            #     decay_steps=self.lr_decay_steps,
            #     end_value=0.1,
            # )
        # else:
        
        
        def build_opt(lr, return_info=True):
            weight_decay_mask = lambda params: jax.tree_map(lambda x: jnp.array(x).size > hidden_size, params)
            learning_rate_schedule = optax.warmup_cosine_decay_schedule(
                init_value=self.init_lr,
                peak_value=lr,
                warmup_steps=self.lr_warmup_steps,
                decay_steps=self.lr_decay_steps,
                end_value=lr*0.1 if self.end_lr is None else self.end_lr,
            )
            optimizer_info = dict(
                learning_rate_schedule=learning_rate_schedule,
            )
            if self.optimizer_name=="adamw":
                inner_opt = optax.adamw(
                    learning_rate=learning_rate_schedule,
                    weight_decay=self.weight_decay,
                    b1=self.b1,
                    b2=self.b2,
                    eps=self.adamw_eps,
                    mu_dtype=jnp.bfloat16 if self.bf16_momentum else jnp.float32,
                )
            elif self.optimizer_name=="adabelief":
                inner_opt = optax.chain(
                    transform.scale_by_belief(b1=self.b1,
                                            b2=self.b2,
                                            eps=self.adabelief_eps,
                                            eps_root=self.eps_root,
                                            ),
                    transform.add_decayed_weights(self.weight_decay, weight_decay_mask),
                    alias._scale_by_learning_rate(learning_rate_schedule),
                )
            else:
                raise ValueError(f"Unknown optimizer {self.optimizer_name}")
            
            optimizer = optax.chain(
                    optax.clip_by_global_norm(self.clip_gradient),
                    inner_opt
                )
            if self.flatten_optimizer:
                transforms={True:optimizer,
                            False:optax.flatten(optimizer)
                            }
                optimizer = optax.multi_transform(transforms=transforms, param_labels=weight_decay_mask)
            if return_info:
                return optimizer, optimizer_info
            else:
                return optimizer
        optimizer, optimizer_info = build_opt(self.lr*self.lrmod)
        if self.a_param_modifier is not None and self.a_param_modifier>0:
            a_optimzer = build_opt(self.lr*self.a_param_modifier*self.lrmod, return_info=False)
            label_func = lambda params: path_aware_map(lambda p,v: "a_param" in "/".join(p), params)
            optimizer = optax.multi_transform(transforms={True:a_optimzer,
                                                            False:optimizer
                                                            },
                                                    param_labels=label_func)
                
        if self.init_from_pretrained:
            if self.pt_mult>=0:
                if self.pt_mult==0:
                    other_opt = optax.set_to_zero()
                else:
                    other_opt = build_opt(self.lr*self.pt_mult*self.lrmod, return_info=False)
                def label_func(params):
                    def func(p,val):
                        p = "/".join(p)
                        return "retriever" in p or "cca" in p or "query_augmentor" in p
                    return path_aware_map(func, params)
                optimizer = optax.multi_transform(transforms={True:optimizer,
                                                                False: other_opt
                                                                },
                                                        param_labels=label_func)
        if self.accumulate_gradient_steps > 1:
            optimizer = optax.MultiSteps(
                optimizer, self.accumulate_gradient_steps
            )
        optimizer = optax.apply_if_finite(optimizer,20)

        return optimizer, optimizer_info


def print_shard(path, array):
    print("/".join(path))
    print(f"{array.shape=} {array.dtype=} each device is holding {array.addressable_shards[0].data.shape}\n{str(array.sharding)}")
    return None


@gin.configurable
def calculate_loss(output, batch, norm_loss:bool=False, epsilon:float=1e-8):
    logits = output.logits
    target_tokens = batch['target_tokens'].reshape(-1)
    loss_masks = batch['loss_masks'].reshape(-1)
    logits = logits.reshape(-1, logits.shape[-1])
    
    loss, accuracy = cross_entropy_loss_and_accuracy(
        logits, target_tokens, loss_masks
    )
    print(loss)
    metrics = {"loss": loss, "accuracy": accuracy, "perplexity":jnp.exp(loss)}
    if output.retriever_output is not None and output.retriever_output.aux_loss is not None:
        raw_aux_loss, valid_pairs = output.retriever_output.aux_loss
        loss_scale = output.retriever_output.loss_scale
        if hasattr(loss_scale,"shape") and np.prod(loss_scale.shape)>1: #for now...
            loss_scale = loss_scale.mean()
        if norm_loss:
            aux_loss = raw_aux_loss.sum()
            scaled_aux_loss = (aux_loss/(epsilon+jax.lax.stop_gradient(aux_loss)))*loss_scale
            loss = loss/(epsilon+jax.lax.stop_gradient(loss))
            
        else:
            aux_loss = raw_aux_loss.sum()/valid_pairs.sum()
            scaled_aux_loss = loss_scale*aux_loss
        metrics['loss_scale'] = loss_scale
        
        metrics['aux_loss'] = aux_loss
        metrics['scaled_aux_loss'] = scaled_aux_loss
        if output.retriever_output.retrieval_metrics is not None:
            retrieval_metrics = jax.tree_map(lambda x:x.mean(),
                                            output.retriever_output.retrieval_metrics)
            metrics = {**metrics,**retrieval_metrics}
        
        loss = loss+scaled_aux_loss
        
    return loss.squeeze(), metrics
    
    

def create_cca(layer):
    new_tree = {}
    attention = layer["attention"]
    
    qkv_part = attention["query_key_value"]
    o_part = attention["dense"]
    for name in ["kernel"] + ["bias"] if "bias" in qkv_part else []:
        mat = qkv_part[name]
        for qkv,val in zip(["wq","wk","wv"],np.split(mat, 3, axis=-1)):
            new_tree[f"{qkv}/{name}"] = val
    
    for name in ["kernel"] + ["bias"] if "bias" in o_part else []:
        new_tree[f"wo/{name}"] = o_part[name]
    return new_tree

@gin.configurable
def create_rpt_params(model,kv_index=0,norm_weights=False):
    tree = model.params
    new_tree = {}
    n_layers = model.config.num_hidden_layers
    min_len =  n_layers//2
    prefix = "gpt_neox/layers/{i}/cca/cross_attention/{suffix}"
    bidir_prefix = "gpt_neox/retriever/preret_bidir_attention/{suffix}"
    for i in range(min_len, model.config.num_hidden_layers):
        layer = tree["gpt_neox"]["layers"][str(i)]
        if i==min_len:
            bidir_params = {bidir_prefix.format(suffix=k):v for k,v in create_cca(layer).items()}
            new_tree.update(bidir_params)
        cca_params= {prefix.format(i=i,suffix=k):v for k,v in create_cca(layer).items()}
        new_tree.update(cca_params)
    first_layer = create_cca(tree["gpt_neox"]["layers"][str(kv_index)])
    new_tree["gpt_neox/retriever/query_projection/kernel"] =first_layer["wq/kernel"]
    new_tree["gpt_neox/retriever/key_projection/kernel"] =first_layer["wk/kernel"]
    new_tree = {k:copy.deepcopy(v) for k,v in new_tree.items()}
    if norm_weights:
        new_tree = {k: v/(np.linalg.norm(v,keepdims=True)+1e-8) for k,v in new_tree.items()}
    return new_tree

def set_rpt_params(model):
    params = model.params
    flat_params = flatten_dict(params,sep="/")
    rpt_params = create_rpt_params(model)
    for k,v in rpt_params.items():
        if k in flat_params:
            flat_params[k] = v
    model.params = unflatten_dict(flat_params,sep="/")
    return model