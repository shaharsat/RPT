from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn
import functools
from collections.abc import Sequence
import einops
from typing import Optional
import gin

@functools.partial(jax.custom_vjp, nondiff_argnums=(1,))
def sqrt_bound_derivative(
    x: jax.Array,
    max_gradient: float | jax.Array,
) -> jax.Array:
  """Computes a square root with a gradient clipped at `max_gradient`."""
  del max_gradient  # unused
  return jnp.sqrt(x)



def stable_sqrt_fwd(
    x: jax.Array,
    _: float | jax.Array
) -> tuple[jax.Array, tuple[jax.Array]]:  # pylint: disable=g-one-element-tuple
  return jnp.sqrt(x), (x,)


def rnn_param_init(
    min_rad: float,
    max_rad: float,
    transform: str = "softplus",
    eps: float = 1e-8,
) -> nn.initializers.Initializer:
  """Initializes the `A` parameter of the RG-LRU uniformly on a ring."""

  def init(
      key: jax.Array,
      shape: Sequence[int],
      dtype = jnp.float32,
  ):
    unif = jax.random.uniform(key, shape=shape)
    # Proportional to area in a ring.
    a_real = 0.5 * jnp.log(unif * (max_rad**2 - min_rad**2) + min_rad**2 + eps)

    if transform == "softplus":
      # Inverse transform.
      return jnp.log(jnp.exp(-a_real) - 1.0).astype(dtype)
    else:
      raise NotImplementedError()

  return init


def stable_sqrt_bwd(
    max_gradient: float | jax.Array,
    res: tuple[jax.Array],  # pylint: disable=g-one-element-tuple
    g: jax.Array,
) -> tuple[jax.Array]:  # pylint: disable=g-one-element-tuple
  (x,) = res
  x_pre = jnp.maximum(x, 1 / (4 * max_gradient**2))
  return jax.vjp(jnp.sqrt, x_pre)[1](g)


sqrt_bound_derivative.defvjp(stable_sqrt_fwd, stable_sqrt_bwd)

@gin.configurable
class BlockDiagonalLinear(nn.Module):
  """Block-diagonal linear layer.

  Attributes:
    width: The number of dimensions of the input and output.
    num_blocks: The number of diagonal blocks in the layer.
    w_init_variance_scale: A parameters that scales the variance of the
      initialization of the weights.
    dtype: dtype used for computation.
    param_dtype: dtype used for initializing parameters.
  """

  width: int
  num_blocks: int
  w_init_variance_scale: float = 1.0
  dtype = None
  param_dtype = jnp.float32
  b_init: float = 0.0

  @property
  def kernel_init(self) -> nn.initializers.Initializer:
    """Initializer for the weight `w` of the layer."""
    return nn.initializers.variance_scaling(
        scale=self.w_init_variance_scale,
        mode="fan_in",
        distribution="normal",
    )

  def setup(self):
    assert self.width % self.num_blocks == 0
    block_width = self.width // self.num_blocks

    # Parameters.
    self.w = self.param(
        "w",
        self.kernel_init,
        [self.num_blocks, block_width, block_width],
        self.param_dtype,
    )
    self.b = self.param(
        "b",
        nn.initializers.constant(self.b_init),
        [self.num_blocks, block_width],
        self.param_dtype,
    )

  def __call__(
      self, x
  ):
    """Calls the BlockDiagonalLinear."""
    x, w, b = nn.dtypes.promote_dtype(x, self.w, self.b, dtype=self.dtype)

    # Split x to blocks.
    x = einops.rearrange(x, "... (h i) -> ... h i", h=self.num_blocks)

    # Linear layer over each block + bias.
    y = jnp.einsum("... h i, h i j -> ... h j", x, w) + b

    # Flatten the output.
    return einops.rearrange(y, "... h j -> ... (h j)", h=self.num_blocks)


@gin.configurable
class GriffinGate(nn.Module):
    width: int
    num_blocks: int
    w_init_variance_scale: float = 1.0
    dtype:jnp.dtype = jnp.float32
    param_dtype:jnp.dtype = jnp.float32
    after_refactor:bool = False
    a_init: Optional[float] = None
    c_value: float = 8
    d_value: float = 1
    gate_type: str = "gru"
    
    def setup(self):
        # Parameters and layers.
        if self.a_init is None or self.a_init <= 0:
          kwargs = dict(min_rad=0.9, max_rad=0.999)
        else:
          kwargs = dict(min_rad=self.a_init, max_rad=self.a_init)
        self.a_param = self.param(
            "a_param",
            rnn_param_init(**kwargs),
            [self.width],
            self.param_dtype,
        )
        if self.after_refactor:
          self.a_param_proj = BlockDiagonalLinear(width=self.width, num_blocks=self.num_blocks)
          self.a_param_proj_ln = nn.LayerNorm(dtype=self.dtype)
        else:
          self.proj = BlockDiagonalLinear(width=self.width, num_blocks=self.num_blocks)
          self.ln = nn.LayerNorm(dtype=self.dtype)
        

    def __call__(self, x):  
        if self.after_refactor:
          gate_input = self.a_param_proj(self.a_param_proj_ln(x))
        else:
          gate_input = self.proj(self.ln(x))
        # Compute the parameter `A` of the recurrence.
        # todo: replace with squareplus and hard_sigmoid?
        log_a = -self.c_value * jax.nn.sigmoid(self.d_value * gate_input) * jax.nn.softplus(self.a_param)
        alpha = jnp.exp(log_a)
        if self.gate_type=="gru":
          alpha = jnp.clip(alpha, 0.05, 0.95)
          beta = 1-alpha
          return alpha,beta
        elif self.gate_type=="griffin":
          a_squared = jnp.exp(2 * log_a) 
          beta = sqrt_bound_derivative(1 - a_squared, 1000) #at init this is  between 0 and 0.1
          return alpha,beta
        else:
          raise NotImplementedError(f"Gate type {self.gate_type} not implemented")
        # new_alpha = alpha/jax.lax.stop_gradient(alpha+beta)
        # new_beta = 1-new_alpha
        
        # new_beta = 1-new_alpha
        # return new_alpha,new_beta
        # return ((alpha*residual_stream)+(beta*x))