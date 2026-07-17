import jax
import jax.numpy as jnp
import jax.random as random
from config import GPTConfig, TrainConfig

def init_params(cfg: GPTConfig, key: jax.Array) -> dict:
    # n layers, each layers has an mlp, 3 projection matrixes, 2 layernorm g/b matrixes
    # model has a final layernorm, an embedding and unembedding matrix
    params = {}

    vocab_size = cfg.vocab_size
    max_seq_len = cfg.max_seq_len
    model_dim = cfg.model_dim
    n_layers = cfg.n_layers
    ff_dim = cfg.ff_dim

    token_embed_key, position_embed_key, unembed_key, key = random.split(key, 4)
    params["token_embed"] = random.normal(token_embed_key, (vocab_size, model_dim), dtype=jnp.float32) * 0.02
    params["position_embed"] = random.normal(position_embed_key, (max_seq_len, model_dim), dtype=jnp.float32) * 0.02

    blocks = []
    
    for _ in range(n_layers):
        W_q_key, W_k_key, W_v_key, W_o_key, mlp_in_key, mlp_out_key, key = random.split(key, 7)
        block = {}
        block["ln_1_gamma"] = jnp.ones((model_dim,), dtype=jnp.float32)
        block["ln_1_beta"] = jnp.zeros((model_dim,), dtype=jnp.float32)
        block["W_q"] = random.normal(W_q_key, (model_dim, model_dim), dtype=jnp.float32) * (model_dim ** -0.5)
        block["W_k"] = random.normal(W_k_key, (model_dim, model_dim), dtype=jnp.float32) * (model_dim ** -0.5)
        block["W_v"] = random.normal(W_v_key, (model_dim, model_dim), dtype=jnp.float32) * (model_dim ** -0.5)
        block["W_o"] = random.normal(W_o_key, (model_dim, model_dim), dtype=jnp.float32) * (model_dim ** -0.5)
        block["ln_2_gamma"] = jnp.ones((model_dim,), dtype=jnp.float32)
        block["ln_2_beta"] = jnp.zeros((model_dim,), dtype=jnp.float32)
        block["mlp_in"] = random.normal(mlp_in_key, (model_dim, ff_dim), dtype=jnp.float32) * (model_dim ** -0.5)
        block["b_in"] = jnp.zeros((ff_dim,), dtype=jnp.float32)
        block["mlp_out"] = random.normal(mlp_out_key, (ff_dim, model_dim), dtype=jnp.float32) * (ff_dim ** -0.5)
        block["b_out"] = jnp.zeros((model_dim,), dtype=jnp.float32)
        blocks.append(block)

    params["blocks"] = blocks
    params["ln_f_gamma"] = jnp.ones((model_dim,), dtype=jnp.float32)
    params["ln_f_beta"] = jnp.zeros((model_dim,), dtype=jnp.float32)
    params["unembed"] = random.normal(unembed_key, (model_dim, vocab_size), dtype=jnp.float32) * (model_dim ** -0.5)

    return params






