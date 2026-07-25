from pathlib import Path

import argparse
import torch
import torch.nn as nn

from aft.model import AFTLanguageModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-steps", type=int, default=10000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--aft-type", type=str, default="local")
    parser.add_argument("--local-window-size", type=int, default=32)
    parser.add_argument("--kernel-size", type=int, default=None)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--output-path", type=str, default="outputs/aft_enwik8.pt")
    parser.add_argument("--log-path", type=str, default="outputs/train_enwik8_log.csv")
    return parser.parse_args()

args = parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = Path("data/enwik8")
train_path = data_dir / "train.pt"
val_path = data_dir / "val.pt"
output_path = Path(args.output_path)
log_path = Path(args.log_path)

seq_len = args.seq_len
batch_size = args.batch_size
d_model = args.d_model
hidden_dim = args.hidden_dim
n_layers = args.n_layers
dropout = args.dropout
num_steps = args.num_steps
eval_interval = args.eval_interval
save_interval = args.save_interval
learning_rate = args.learning_rate
aft_type = args.aft_type
local_window_size = args.local_window_size
kernel_size = args.kernel_size
causal = args.causal
vocab_size = 256

train_data = torch.load(train_path, map_location="cpu")
val_data = torch.load(val_path, map_location="cpu")

def make_batch(data, batch_size, seq_len, device):
    starts = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[start:start + seq_len] for start in starts])
    y = torch.stack([data[start + 1:start + seq_len + 1] for start in starts])
    return x.to(device), y.to(device)

print("device:", device)
print("train tokens:", len(train_data))
print("val tokens:", len(val_data))
print("config:")
print("  seq_len:", seq_len)
print("  batch_size:", batch_size)
print("  d_model:", d_model)
print("  hidden_dim:", hidden_dim)
print("  n_layers:", n_layers)
print("  dropout:", dropout)
print("  num_steps:", num_steps)
print("  eval_interval:", eval_interval)
print("  save_interval:", save_interval)
print("  learning_rate:", learning_rate)
print("  aft_type:", aft_type)
print("  local_window_size:", local_window_size)
print("  kernel_size:", kernel_size)
print("  causal:", causal)
print("  output_path:", output_path)
print("  log_path:", log_path)

model = AFTLanguageModel(
    vocab_size=vocab_size,
    d_model=d_model,
    hidden_dim=hidden_dim,
    n_layers=n_layers,
    max_seq_len=seq_len,
    dropout=dropout,
    aft_type=aft_type,
    local_window_size=local_window_size,
    kernel_size=kernel_size,
    causal=causal,
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
start_step = 0

def move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)

if output_path.exists():
    checkpoint = torch.load(output_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    move_optimizer_state_to_device(optimizer, device)
    start_step = checkpoint["step"] + 1
    print("resumed from checkpoint:", output_path)
    print("start step:", start_step)

if start_step == 0:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("step,train_loss,val_loss\n", encoding="utf-8")

@torch.no_grad()
def estimate_loss(data, num_batches=20):
    model.eval()
    losses = []

    for _ in range(num_batches):
        x, y = make_batch(data, batch_size, seq_len, device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)

def save_checkpoint(step):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {
                "vocab_size": vocab_size,
                "seq_len": seq_len,
                "batch_size": batch_size,
                "d_model": d_model,
                "hidden_dim": hidden_dim,
                "n_layers": n_layers,
                "dropout": dropout,
                "num_steps": num_steps,
                "learning_rate": learning_rate,
                "aft_type": aft_type,
                "local_window_size": local_window_size,
                "kernel_size": kernel_size,
                "causal": causal,
            },
        },
        output_path,
    )

def write_log(step, train_loss, val_loss):
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"{step},{train_loss},{val_loss}\n")

for step in range(start_step, num_steps):
    x, y = make_batch(train_data, batch_size, seq_len, device)

    logits = model(x)
    loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % eval_interval == 0:
        val_loss = estimate_loss(val_data)
        print(
            "step:",
            step,
            "train loss:",
            loss.item(),
            "val loss:",
            val_loss,
        )
        write_log(step, loss.item(), val_loss)

    if step > 0 and step % save_interval == 0:
        save_checkpoint(step)
        print("saved checkpoint at step", step)

save_checkpoint(num_steps)
print("saved final checkpoint:", output_path)
