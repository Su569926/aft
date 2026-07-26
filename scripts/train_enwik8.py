from pathlib import Path

import argparse
import torch
import torch.nn as nn
import math

from aft.model import AFTLanguageModel

def parse_args():
    # 用命令行参数控制实验配置，避免每次上云调参都手动改代码。
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
    return parser.parse_args()

args = parse_args()

# 自动选择 GPU；如果本地没有 CUDA，就退回 CPU 做语法/小规模调试。
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
amp = args.amp
use_checkpoint = args.use_checkpoint
vocab_size = 256

train_data = torch.load(train_path, map_location="cpu")
val_data = torch.load(val_path, map_location="cpu")

def make_batch(data, batch_size, seq_len, device):
    # 从长序列中随机抽取 batch_size 个起点。
    starts = torch.randint(0, len(data) - seq_len - 1, (batch_size,))

    # x 是从 start 开始的 seq_len 个 token，y 是整体向后错一位的目标。
    x = torch.stack([data[start:start + seq_len] for start in starts])
    y = torch.stack([data[start + 1:start + seq_len + 1] for start in starts])

    # 整份数据保存在 CPU，只把当前 batch 移到训练设备，节省 GPU 显存。
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
print("  amp:", amp)
print("  use_checkpoint:", use_checkpoint)

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
# AMP 开启时用 GradScaler 缩放 loss，降低 float16 梯度下溢风险。
scaler = torch.amp.GradScaler(enabled=amp)
start_step = 0

def move_optimizer_state_to_device(optimizer, device):
    # resume 时 optimizer 状态可能先加载到 CPU，这里把其中的 tensor 搬到当前设备。
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)

if output_path.exists():
    # 如果 checkpoint 已存在，就恢复模型、优化器和 step，从中断处继续训练。
    checkpoint = torch.load(output_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    move_optimizer_state_to_device(optimizer, device)
    start_step = checkpoint["step"] + 1
    print("resumed from checkpoint:", output_path)
    print("start step:", start_step)

if start_step == 0:
    # 从头训练时创建新日志；resume 时保留旧日志并继续追加。
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("step,train_loss,val_loss,val_bpc\n", encoding="utf-8")

@torch.no_grad()
def estimate_loss(data, num_batches=20):
    # 验证阶段不需要梯度，关闭 dropout 等训练行为后估计平均 loss。
    model.eval()
    losses = []

    for _ in range(num_batches):
        x, y = make_batch(data, batch_size, seq_len, device)
        # 评估也可以使用 autocast，节省显存并加速前向计算。
        with torch.amp.autocast(enabled=amp):
            logits = model(x)
            loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)

def save_checkpoint(step):
    # 保存模型参数、优化器状态和复现实验所需的关键配置。
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

def write_log(step, train_loss, val_loss, val_bpc):
    # CSV 日志用于后续画曲线和对照论文指标。
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"{step},{train_loss},{val_loss},{val_bpc}\n")

for step in range(start_step, num_steps):
    x, y = make_batch(train_data, batch_size, seq_len, device)

    # logits: [B, T, vocab_size]，y: [B, T]。
    with torch.amp.autocast(enabled=amp):
        logits = model(x)
        # CrossEntropyLoss 需要 [N, C] 和 [N]，所以把 B 和 T 展平。
        loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))

    optimizer.zero_grad()

    # AMP 和普通训练的反向传播/参数更新写法不同。
    if amp:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    if step % eval_interval == 0:
        val_loss = estimate_loss(val_data)
        # PyTorch 交叉熵单位是 nats；论文 Enwik8 指标 BPC 用 bits，所以除以 ln(2)。
        val_bpc = val_loss / math.log(2)
        print(
            "step:",
            step,
            "train loss:",
            loss.item(),
            "val loss:",
            val_loss,
            "val bpc:",
            val_bpc,
        )
        write_log(step, loss.item(), val_loss, val_bpc)

    if step > 0 and step % save_interval == 0:
        # 定期保存，避免云训练中断后从头再来。
        save_checkpoint(step)
        print("saved checkpoint at step", step)

save_checkpoint(num_steps)
print("saved final checkpoint:", output_path)
