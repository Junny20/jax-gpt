import jax
import jax.numpy as jnp
import jax.random as random
import urllib.request

def build_tokenizer(text: str) -> tuple[dict, dict, int]:
    """Tuple returns an encoder, decoder, and vocab size"""
    vocab = sorted(set(text))
    vocab_size = len(vocab)
    encoder = {c: i for i, c in enumerate(vocab)}
    decoder = {i: c for c, i in encoder.items()}
    return encoder, decoder, vocab_size


def encode(text: str, encoder: dict) -> jax.Array:
    return jnp.array([encoder[c] for c in text], dtype=jnp.int32)


def decode(tokens: jax.Array, decoder) -> str:
    return ''.join(decoder[int(t)] for t in tokens)


def get_batch(data: jax.Array, batch_size: int, block_size: int, key) -> tuple[jax.Array, jax.Array]:
    """samples batch_size starting positions within data, returns tuple of (batch_size, block_size) tokens_ids, targets"""
    assert block_size < data.shape[0], f"block_size {block_size} >= data_len {data.shape[0]}"

    data_len = data.shape[0]
    min_idx = 0
    max_idx = data_len - block_size # exclusive
    
    starting_idxs = random.randint(key, (batch_size,), min_idx, max_idx, dtype=jnp.int32) # (batch_size,)
    offsets = jnp.arange(block_size, dtype=jnp.int32) # (block_size,)
    token_ids = starting_idxs[:, None] + offsets
    targets = token_ids + 1

    
    return data[token_ids], data[targets] 


def split_data(data: jax.Array, val_frac: float = 0.1) -> tuple[jax.Array, jax.Array]:
    n = int(data.shape[0] * (1 - val_frac))
    return data[:n], data[n:]


def load_tinyshakespeare() -> str:
    urllib.request.urlretrieve("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", "src/jaxgpt/input.txt")
    return "input.txt"

