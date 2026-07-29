from pathlib import Path
import os
import math

import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from aft.model import AFTLanguageModel

def parse_args():
    # DDP = DistributedDataParallel，中文常叫“分布式数据并行”。
    # DDP 版本保留单卡脚本的所有训练参数，方便两套脚本使用同一套命令习惯。
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
    # NCCL 是 NVIDIA GPU 多卡通信常用后端，适合 CUDA 上的 DDP 训练。
    # torchrun 会为每个 GPU 启动一个 Python 进程；这里初始化这些进程之间的通信。
    dist.init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"]) #当前进程使用的第几张gpu
    rank = dist.get_rank() #当前进程在所有进程的编号
    world_size = dist.get_world_size() #总进程数 = 总GPU数

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    return device, rank, local_rank, world_size

def cleanup_ddp():
    # 训练结束后关闭分布式进程组，释放通信资源。
    dist.destroy_process_group()

args = parse_args()

device, rank, local_rank, world_size = setup_ddp()
# 只让 rank 0 负责打印、写日志、保存 checkpoint，避免多个进程同时写同一个文件。
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
    # 每个 rank 都会独立随机抽 batch；DDP 会在反向传播时自动同步各 rank 的梯度。
    # x: [B, T] 是输入 token；y: [B, T] 是向后错一位的下一个 token 目标。
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
# AMP = Automatic Mixed Precision，中文常叫“自动混合精度”。
# AMP 开启时用 GradScaler 缩放 loss，降低 float16 梯度下溢风险。
scaler = torch.amp.GradScaler("cuda", enabled=amp)
start_step = 0

def move_optimizer_state_to_device(optimizer, device):
    # checkpoint 中 optimizer 状态可能先被加载到 CPU；恢复训练前要搬到当前 rank 的 GPU。
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)

if output_path.exists():# resume逻辑
    # 注意：这里在 DDP 包装前加载模型参数，因此 checkpoint 里保存的是原始模型 state_dict。
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
    # 从头训练时创建新日志；resume 时不覆盖旧日志，而是继续追加。
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("step,train_loss,val_loss,val_bpc\n", encoding="utf-8")

@torch.no_grad()
def estimate_loss(data, num_batches=20):
    # 所有 rank 都参与验证，并用 all_reduce 求平均；否则 rank 0 验证时其他 rank 继续训练会卡住。
    model.eval()
    total_loss = 0.0

    for _ in range(num_batches):
        x, y = make_batch(data, batch_size, seq_len, device)
        with torch.amp.autocast("cuda", enabled=amp):
            logits = model(x)
            # logits: [B, T, vocab_size]，y: [B, T]。
            # CrossEntropyLoss 需要 [样本数, 类别数] 和 [样本数]，所以把 B*T 展平。
            loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
        total_loss += loss.item()

    local_avg_loss = total_loss / num_batches

    # 把每个 rank 的验证 loss 求和，再除以 world_size 得到全局平均验证 loss。
    loss_tensor = torch.tensor(local_avg_loss, device=device)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    global_avg_loss = loss_tensor.item() / world_size

    model.train()
    return global_avg_loss

def save_checkpoint(step):
    # 只有主进程保存，避免多个 rank 同时写同一个 checkpoint 文件。
    if not is_main_process:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            # DDP 包装后，真正的原始模型在 model.module 里。
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

def write_log(step, train_loss, val_loss, val_bpc):
    # 日志同样只让主进程写，避免多进程重复写入。
    if not is_main_process:
        return
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"{step},{train_loss},{val_loss},{val_bpc}\n")

for step in range(start_step, num_steps):
    optimizer.zero_grad()
    total_loss = 0.0

    # Gradient Accumulation，中文常叫“梯度累积”：
    # 多个 micro batch 只 backward，不 step；循环结束后统一更新一次参数。
    for micro_step in range(grad_accum_steps):
        x, y = make_batch(train_data, batch_size, seq_len, device)


        with torch.amp.autocast("cuda", enabled=amp):
            logits = model(x)
            loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
            # 除以累积步数，保证累积后的梯度相当于大 batch 的平均梯度。
            loss = loss / grad_accum_steps

        if amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        total_loss += loss.item()

    # 所有 micro batch 的梯度都累积完以后，才真正更新一次参数。
    if amp:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    train_loss = total_loss

    if step % eval_interval == 0:
        # estimate_loss 内部有 all_reduce，所以所有 rank 必须一起进入。
        val_loss = estimate_loss(val_data)
        # BPC = bits per character，中文可理解为“每字符比特数”。
        # CrossEntropyLoss 的单位是 nats，除以 ln(2) 后转换成 bits。
        val_bpc = val_loss / math.log(2)

        if is_main_process:
            print(
                "step:",
                step,
                "train loss:",
                train_loss,
                "val loss:",
                val_loss,
                "val bpc:",
                val_bpc,
            )
            write_log(step, train_loss, val_loss, val_bpc)

    if is_main_process and step > 0 and step % save_interval == 0:
        save_checkpoint(step)
        print("saved checkpoint at step", step)

save_checkpoint(num_steps)
if is_main_process:
    print("saved final checkpoint:", output_path)

cleanup_ddp()
