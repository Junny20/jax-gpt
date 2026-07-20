import jax
import jax.numpy as jnp
import optax
from functools import partial
from jaxgpt.config import GPTConfig
from jaxgpt.model import loss
from jax import jit, vmap


def batched_loss(params: dict, cfg: GPTConfig, token_ids: jax.Array, targets: jax.Array) -> jax.Array:
    return jnp.mean(vmap(loss, in_axes=(None, None, 0, 0))(params, cfg, token_ids, targets))


@partial(jit, static_argnums=(0, 2))
def update(cfg: GPTConfig, params: dict, optimizer, opt_state, token_ids: jax.Array, targets: jax.Array):
    loss_value, grads = jax.value_and_grad(batched_loss)(params, cfg, token_ids, targets)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_value