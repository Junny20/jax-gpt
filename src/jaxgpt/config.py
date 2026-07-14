from dataclasses import dataclass

@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    max_seq_len: int
    model_dim: int
    n_layers: int
    n_heads: int
    ff_dim: int

    @property
    def head_dim(self) -> int:
        return self.model_dim // self.n_heads

    def __post_init__(self):
        assert self.vocab_size > 0
        assert self.max_seq_len > 0
        assert self.model_dim > 0
        assert self.n_layers > 0
        assert self.n_heads > 0
        assert self.ff_dim > 0
        assert self.model_dim % self.n_heads == 0, f"model_dim {self.model_dim} not divisible by n_heads {self.n_heads}"


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int
    block_size: int
    lr: float
    n_steps: int

    def __post_init__(self):
        assert self.batch_size > 0
        assert self.block_size > 0
        assert self.lr > 0.0
        assert self.n_steps > 0