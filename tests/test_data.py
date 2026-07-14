import jax.numpy as jnp
import jax.random as random
from jaxgpt.data import build_tokenizer, encode, decode, get_batch, split_data


def test_roundtrip():
    text = "hello world"
    enc, dec, _ = build_tokenizer(text)
    assert decode(encode(text, enc), dec) == text


def test_vocab_size():
    enc, dec, V = build_tokenizer("aabbc")
    assert V == 3 and len(enc) == len(dec) == 3


def test_batch_shapes_and_alignment():
    data = jnp.arange(1000, dtype=jnp.int32)
    x, y = get_batch(data, batch_size=4, block_size=8, key=random.key(0))
    assert x.shape == (4, 8) and y.shape == (4, 8)
    assert jnp.all(y[:, :-1] == x[:, 1:])


def test_batch_in_bounds():
    data = jnp.arange(100, dtype=jnp.int32)
    for seed in range(50):
        x, y = get_batch(data, 8, 16, random.key(seed))
        assert int(jnp.max(y)) < 100


def test_split_no_overlap():
    data = jnp.arange(100, dtype=jnp.int32)
    tr, val = split_data(data, 0.1)
    assert tr.shape[0] == 90 and val.shape[0] == 10
    assert int(tr[-1]) < int(val[0])