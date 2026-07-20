import jax.random as random
import optax
import csv, os
from jaxgpt.data import build_tokenizer, encode, get_batch, split_data, load_tinyshakespeare
from jaxgpt.config import GPTConfig, TrainConfig
from jaxgpt.model import init_params
from jaxgpt.optimizer import update, batched_loss

def main():
    text = load_tinyshakespeare()
    encoder, decoder, vocab_size = build_tokenizer(text)
    data = encode(text, encoder)
    train_data, val_data = split_data(data, val_frac=0.1)

    cfg = GPTConfig(vocab_size=vocab_size, max_seq_len=256, model_dim=384, n_layers=6, n_heads=6, ff_dim=1536)
    train_cfg = TrainConfig(batch_size=32, block_size=256, lr=3e-4, n_steps=5000)

    key = random.key(0)
    key, init_key = random.split(key)
    params = init_params(cfg, init_key)

    optimizer = optax.adamw(train_cfg.lr)
    opt_state = optimizer.init(params)

    history = []
    for step in range(train_cfg.n_steps):
        key, batch_key = random.split(key)
        token_ids, targets = get_batch(train_data, train_cfg.batch_size, train_cfg.block_size, batch_key)
        params, opt_state, loss_value = update(cfg, params, optimizer, opt_state, token_ids, targets)
        if step % 100 == 0:
            key, val_key = random.split(key)
            val_token_ids, val_targets = get_batch(val_data, train_cfg.batch_size, train_cfg.block_size, val_key)
            val_loss = batched_loss(params, cfg, val_token_ids, val_targets)
            history.append((step, loss_value, val_loss))

            print(f"Step {step}, loss: {loss_value}, validation loss: {val_loss}")

    os.makedirs("logs", exist_ok=True)
    with open("logs/loss.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "train_loss", "val_loss"])
        w.writerows(history)


if __name__ == "__main__":
    main()