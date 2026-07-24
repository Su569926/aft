from pathlib import Path

import torch
import torch.nn as nn

from aft.model import AFTLanguageModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = Path("data/enwik8")
train_path = data_dir / "train.pt"
val_path = data_dir / "val.pt"
output_path = Path("outputs/aft_enwik8.pt")

vocab_size = 256
seq_len = 128
batch_size = 32
d_model = 128
hidden_dim = 512
n_layers = 4
dropout = 0.1
num_steps = 10
eval_interval = 5
learning_rate = 3e-4
aft_type = "local"
local_window_size = 32
kernel_size = None

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

model = AFTLanguageModel(
    vocab_size=vocab_size,
    d_model=d_model,
    hidden_dim=hidden_dim,
    n_layers=n_layers,
    max_seq_len=seq_len,
    dropout=dropout,
    aft_type=aft_type,
    local_window_size=local_window_size,
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

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

for step in range(num_steps):
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

output_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "config": {
            "vocab_size": vocab_size,
            "seq_len": seq_len,
            "d_model": d_model,
            "hidden_dim": hidden_dim,
            "n_layers": n_layers,
            "dropout": dropout,
            "aft_type": aft_type,
            "local_window_size": local_window_size,
            "kernel_size": kernel_size,
        },
    },
    output_path,
)