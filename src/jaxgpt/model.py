import jax
import jax.numpy as jnp
import jax.random as random
from jax import vmap
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


def embed(cfg: GPTConfig, params: dict, token_ids: jax.Array) -> jax.Array:
    # embedding for a (seq_len,) sequence - returns an array of (seq_len, model_dim)
    seq_len = token_ids.shape[0]

    assert seq_len <= cfg.max_seq_len, "Sequence length exceeds maximum sequence length"

    token_embed = params["token_embed"]
    position_embed = params["position_embed"]
    embeddings = token_embed[token_ids] + position_embed[:seq_len] # both (seq_len, model_dim) arrays 
    return embeddings


def layernorm(x: jax.Array, gamma: jax.Array, beta: jax.Array, eps=1e-5):
    norm = (x - jnp.mean(x, axis=-1, keepdims=True)) / jnp.sqrt(jnp.var(x, axis=-1, keepdims=True) + eps)
    return gamma * norm + beta


def causal_mask(seq_len): # only creates the mask
    # uses jnp.tril, first creates a (seq_len, seq_len) matrix
    return jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool))


def single_head_attention(q, k, v, mask):
    # q, k, v are all (seq_len, n_dim) matrixes
    assert q.shape[1] == k.shape[1] == v.shape[1], "Invalid matrix shape[1]"
    assert q.shape[0] == k.shape[0] == v.shape[0], "Invalid matrix shape[0]"

    n_dim = q.shape[1]

    S = q @ k.T * (n_dim ** -0.5) # (seq_len, seq_len) - divide by sqrt(fan_in)
    S = jnp.where(mask, S, -jnp.inf)
    S = jax.nn.softmax(S)
    return S @ v


def multi_head_attention(cfg: GPTConfig, mha_params: dict, x: jax.Array, mask: jax.Array) -> jax.Array:
    # x is the res stream, (seq_len, model_dim)
    # batches single head attention function
    # reshapes q, k, v, transposes, and funnel into batched function
    # pre-norm
    mha = vmap(single_head_attention, in_axes=(0, 0, 0, None))

    seq_len = x.shape[0]
    n_heads = cfg.n_heads
    head_dim = cfg.head_dim
    model_dim = cfg.model_dim

    Q = x @ mha_params["W_q"]
    Q = Q.reshape(seq_len, n_heads, head_dim).transpose(1, 0, 2)
    K = x @ mha_params["W_k"]
    K = K.reshape(seq_len, n_heads, head_dim).transpose(1, 0, 2)
    V = x @ mha_params["W_v"]
    V = V.reshape(seq_len, n_heads, head_dim).transpose(1, 0, 2)
    # Q, K, V all shape (n_heads, seq_len, head_dim)

    O = mha(Q, K, V, mask) # shape (n_heads, seq_len, head_dim)
    O = O.transpose(1, 0, 2).reshape(seq_len, model_dim)
    return O @ mha_params["W_o"]


def mlp(mlp_params, x):
    # x is shape (seq_len, model_dim)
    x = x @ mlp_params["mlp_in"] + mlp_params["b_in"]
    x = jax.nn.gelu(x)
    return x @ mlp_params["mlp_out"] + mlp_params["b_out"]

def transformer_block(cfg, block_params, x, mask): ...

def gpt2_forward(cfg, params, token_ids): ...

def loss(cfg, params, token_ids, targets): ...




