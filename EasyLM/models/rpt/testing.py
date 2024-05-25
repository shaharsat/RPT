import torch
import jax.numpy as jnp
from jax import lax

print(torch.maximum(torch.arange(-5, 5), torch.tensor(0)))