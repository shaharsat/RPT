from __future__ import annotations

import torch
from torch import nn
import functools
from collections.abc import Sequence
import einops
from typing import Optional
import gin


def sqrt_bound_derivative(
        x: torch.Tensor,
        max_gradient: float | torch.Tensor,
) -> torch.Tensor:
    """Computes a square root with a gradient clipped at `max_gradient`."""
    del max_gradient  # unused
    return torch.sqrt(x)


def stable_sqrt_fwd(
        x: torch.Tensor,
        _: float | torch.Tensor
) -> tuple[torch.Tensor, tuple[torch.Tensor]]:  # pylint: disable=g-one-element-tuple
    return torch.sqrt(x), (x,)



def stable_sqrt_bwd(
        max_gradient: float | torch.Tensor,
        res: tuple[torch.Tensor],  # pylint: disable=g-one-element-tuple
        g: torch.Tensor,
) -> tuple[torch.Tensor]:  # pylint: disable=g-one-element-tuple
    (x,) = res
    x_pre = torch.maximum(x, 1 / (4 * max_gradient ** 2))
    return torch.autograd.functional.vjp(torch.sqrt, x_pre)[1](g)


# TODO: Used in inference?
#sqrt_bound_derivative.defvjp(stable_sqrt_fwd, stable_sqrt_bwd)


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

    def __init__(self, width, num_blocks, w_init_variance_scale=1.0, dtype=None, b_init=0.0):
        self.width = width
        self.num_blocks = num_blocks
        self.w_init_variance_scale = w_init_variance_scale
        self.dtype = dtype

        super().__init__()
        assert self.width % self.num_blocks == 0
        block_width = self.width // self.num_blocks

        # TODO: random intialization (is this required for inference?)
        # Parameters.
        self.w = nn.Parameter(
            torch.Tensor((self.num_blocks, block_width, block_width)),
        )
        self.b = nn.Parameter(
            torch.full((self.num_blocks, block_width), b_init)
        )

    def forward(
            self, x
    ):
        """Calls the BlockDiagonalLinear."""
        # TODO: Promote types
        # x, w, b = nn.dtypes.promote_dtype(x, self.w, self.b, dtype=self.dtype)

        # Split x to blocks.
        x = einops.rearrange(x, "... (h i) -> ... h i", h=self.num_blocks)

        # Linear layer over each block + bias.
        y = torch.einsum("... h i, h i j -> ... h j", x, w) + b

        # Flatten the output.
        return einops.rearrange(y, "... h j -> ... (h j)", h=self.num_blocks)


@gin.configurable
class GriffinGate(nn.Module):
    def __init__(self,
                 hidden_size,
                 width,
                 num_blocks,
                 w_init_variance_scale = 1.0,
                 dtype: torch.dtype = torch.float32,
                 after_refactor: bool = False,
                 a_init: Optional[float] = None,
                 c_value: float = 8,
                 d_value: float = 1,
                 gate_type: str = "gru"
                 ):
        # Parameters and layers.
        super().__init__()
        self.width = width
        self.num_blocks = num_blocks
        self.w_init_variance_scale = w_init_variance_scale
        self.dtype = dtype
        self.after_refactor = after_refactor
        self.a_init = a_init
        self.c_value = c_value
        self.d_value = d_value
        self.gate_type = gate_type

        if self.a_init is None or self.a_init <= 0:
            kwargs = dict(min_rad=0.9, max_rad=0.999)
        else:
            kwargs = dict(min_rad=self.a_init, max_rad=self.a_init)
        self.a_param = torch.nn.Parameter(torch.Tensor(self.width)) # TODO: Fix initialization
        if self.after_refactor:
            self.a_param_proj = BlockDiagonalLinear(width=self.width, num_blocks=self.num_blocks)
            self.a_param_proj_ln = nn.LayerNorm(normalized_shape=hidden_size, dtype=self.dtype)
        else:
            self.proj = BlockDiagonalLinear(width=self.width, num_blocks=self.num_blocks)
            self.ln = nn.LayerNorm(normalized_shape=hidden_size, dtype=self.dtype)

    def forward(self, x):
        if self.after_refactor:
            gate_input = self.a_param_proj(self.a_param_proj_ln(x))
        else:
            gate_input = self.proj(self.ln(x))
        # Compute the parameter `A` of the recurrence.
        # todo: replace with squareplus and hard_sigmoid?
        log_a = -self.c_value * torch.sigmoid(self.d_value * gate_input) * torch.nn.functional.softplus(self.a_param)
        alpha = torch.exp(log_a)
        if self.gate_type == "gru":
            alpha = torch.clip(alpha, 0.05, 0.95)
            beta = 1 - alpha
            return alpha, beta
        elif self.gate_type == "griffin":
            a_squared = torch.exp(2 * log_a)
            beta = sqrt_bound_derivative(1 - a_squared, 1000)  # at init this is  between 0 and 0.1
            return alpha, beta
        else:
            raise NotImplementedError(f"Gate type {self.gate_type} not implemented")
        # new_alpha = alpha/jax.lax.stop_gradient(alpha+beta)
        # new_beta = 1-new_alpha

        # new_beta = 1-new_alpha
        # return new_alpha,new_beta
        # return ((alpha*residual_stream)+(beta*x))