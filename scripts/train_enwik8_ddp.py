from pathlib import Path
import os

import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

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
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    return parser.parse_args()

def setup_ddp():
    dist.init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"]) #当前进程使用的第几张gpu
    rank = dist.get_rank() #当前进程在所有进程的编号
    world_size = dist.get_world_size() #总进程数 = 总GPU数

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    return device, rank, local_rank, world_size

def cleanup_ddp():
    dist.destroy_process_group()

args = parse_args()

device, rank, local_rank, world_size = setup_ddp()
is_main_process = rank == 0

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
amp = args.amp
use_checkpoint = args.use_checkpoint
grad_accum_steps = args.grad_accum_steps
vocab_size = 256

train_data = torch.load(train_path, map_location="cpu")
val_data = torch.load(val_path, map_location="cpu")

def make_batch(data, batch_size, seq_len, device):
    starts = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[start:start + seq_len] for start in starts])
    y = torch.stack([data[start + 1:start + seq_len + 1] for start in starts])
    return x.to(device), y.to(device)

if is_main_process:
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
    print("  amp:", amp)
    print("  use_checkpoint:", use_checkpoint)
    print("  grad_accum_steps:", grad_accum_steps)
    print("  effective_batch_size:", batch_size * world_size * grad_accum_steps)

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
    use_checkpoint=use_checkpoint,
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scaler = torch.amp.GradScaler("cuda", enabled=amp)
start_step = 0

def move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)

if output_path.exists():# resume逻辑
    checkpoint = torch.load(output_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    move_optimizer_state_to_device(optimizer, device)
    start_step = checkpoint["step"] + 1
    if is_main_process:
        print("resumed from checkpoint:", output_path)
        print("start step:", start_step)

model = DDP(model, device_ids=[local_rank], output_device=local_rank)

if is_main_process and start_step == 0:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("step,train_loss,val_loss\n", encoding="utf-8")

@torch.no_grad()
def estimate_loss(data, num_batches=20):
    model.eval()
    total_loss = 0.0

    for _ in range(num_batches):
        x, y = make_batch(data, batch_size, seq_len, device)
        with torch.amp.autocast("cuda", enabled=amp):
            logits = model(x)
            loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
        total_loss += loss.item()

    local_avg_loss = total_loss / num_batches

    loss_tensor = torch.tensor(local_avg_loss, device=device)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    global_avg_loss = loss_tensor.item() / world_size

    model.train()
    return global_avg_loss

def save_checkpoint(step):
    if not is_main_process:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state_dict": model.module.state_dict(),
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
                "grad_accum_steps": grad_accum_steps,
                "world_size": world_size,
                "effective_batch_size": batch_size * world_size * grad_accum_steps,
            },
        },
        output_path,
    )

def write_log(step, train_loss, val_loss):
    if not is_main_process:
        return
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"{step},{train_loss},{val_loss}\n")

for step in range(start_step, num_steps):
    optimizer.zero_grad()
    total_loss = 0.0

    for micro_step in range(grad_accum_steps):
        x, y = make_batch(train_data, batch_size, seq_len, device)


        with torch.amp.autocast("cuda", enabled=amp):
            logits = model(x)
            loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
            loss = loss / grad_accum_steps

        if amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        total_loss += loss.item()

    if amp:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    train_loss = total_loss

    if step % eval_interval == 0:
        val_loss = estimate_loss(val_data)

        if is_main_process:
            print(
                "step:",
                step,
                "train loss:",
                train_loss,
                "val loss:",
                val_loss,
            )
            write_log(step, train_loss, val_loss)

    if is_main_process and step > 0 and step % save_interval == 0:
        save_checkpoint(step)
        print("saved checkpoint at step", step)

save_checkpoint(num_steps)
if is_main_process:
    print("saved final checkpoint:", output_path)

cleanup_ddp()
