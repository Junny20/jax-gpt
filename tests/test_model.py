import jax.numpy as jnp
import jax.random as random
from jaxgpt.config import GPTConfig
from jaxgpt.model import init_params, gpt2_forward, loss

def test_forward_shape():
    cfg = GPTConfig(vocab_size=65, max_seq_len=32, model_dim=16, n_layers=2, n_heads=4, ff_dim=64)
    params = init_params(cfg, random.key(0))
    ids = jnp.arange(8, dtype=jnp.int32)
    assert gpt2_forward(cfg, params, ids).shape == (8, 65)


def test_causal():
    # perturbing token t+1 must not change logits at position t
    cfg = GPTConfig(vocab_size=65, max_seq_len=32, model_dim=16, n_layers=2, n_heads=4, ff_dim=64)
    params = init_params(cfg, random.key(0))
    a = jnp.array([1,2,3,4,5], dtype=jnp.int32)
    b = a.at[4].set(9)
    la, lb = gpt2_forward(cfg, params, a), gpt2_forward(cfg, params, b)
    assert jnp.allclose(la[:4], lb[:4], atol=1e-5)


def test_loss_at_init():
    # untrained loss should be ~ln(vocab_size)
    cfg = GPTConfig(vocab_size=65, max_seq_len=32, model_dim=16, n_layers=2, n_heads=4, ff_dim=64)
    params = init_params(cfg, random.key(0))
    ids = jnp.arange(8, dtype=jnp.int32)
    l = loss(cfg, params, ids, ids)
    assert abs(float(l) - jnp.log(65)) < 0.5