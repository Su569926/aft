from pathlib import Path
import os
import argparse

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

from aft.vision import AFTImageClassifier

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", type=str, default="data/imagenet")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--num-classes", type=int, default=1000)

    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=8) #用多少个 CPU 子进程在后台加载图片、做 transforms、拼 batch

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad-accum-steps", type=int, default=1)

    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument("--output-path", type=str, default="outputs/aft_imagenet.pt")
    parser.add_argument("--log-path", type=str, default="outputs/train_imagenet_log.csv")

    return parser.parse_args()

def setup_ddp():
    dist.init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"]) #当前进程使用的第几张gpu，哪台机器上的几号位
    rank = dist.get_rank() #当前进程在所有进程的编号
    world_size = dist.get_world_size() #总进程数 = 总GPU数

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    return device, rank, local_rank, world_size

def cleanup_ddp():
    dist.destroy_process_group()

def accuracy(logits, targets, topk=(1, 5)):
    max_k = max(topk)

    _, pred = logits.topk(max_k, dim=1) #pred: [B, max_k]，取最高的 5 个类别编号
    pred = pred.t() #[max_k, B]

    correct = pred.eq(targets.reshape(1, -1).expand_as(pred)) #targets是真实类别[B]，correct: [max_k, B]，里面是true/false

    results = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum() #把前 k 行摊平成一维，然后数 True 的数量
        results.append(correct_k)

    return results #返回每个 K 值对应的正确样本数

def build_dataloaders(args, rank, world_size):
    train_dir = Path(args.data_dir) / "train"
    val_dir = Path(args.data_dir) / "val"

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.image_size), #训练集随机裁剪成 224 x 224，这是 ImageNet 常用训练增强
        transforms.RandomHorizontalFlip(), #随机左右翻转图片，提高泛化
        transforms.ToTensor(), #[H, W, C] -> [C, H, W]，并且像素值从 0~255 变成 0~1
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ), #标准均值和方差归一化
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256), #缩放到[256, 256]
        transforms.CenterCrop(args.image_size), #从中间裁[224, 224]
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ])

    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=train_transform,
    ) #图像分类的加载器

    val_dataset = datasets.ImageFolder(
        val_dir,
        transform=val_transform,
    )

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size, #GPU数量
        rank=rank, #当前GPU
        shuffle=True, #训练时打乱数据
    ) #分布式采样器，把整个数据集平均分给所有 GPU

    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True, #训练时丢掉最后一个不完整 batch
    ) #数据加载器

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size, #不是总的batch，而是每一张GPU的batch
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True, #把数据放在 CUDA 锁页内存中，加速从 CPU 复制到 GPU 的速度
        drop_last=False,
    )

    return train_loader, val_loader, train_sampler, val_sampler

def build_model_and_train_state(args, device):
    model = AFTImageClassifier(
        image_size=args.image_size,
        patch_size=args.patch_size,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        d_model=args.d_model,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=args.amp) #AMP 混合精度用的梯度缩放器

    return model, criterion, optimizer, scaler

def move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)

def save_checkpoint(
        path,
        epoch,
        model,
        optimizer,
        scaler,
        args,
        world_size,
        is_main_process,
):
    if not is_main_process:
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(model, DDP):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "config": vars(args),
            "world_size": world_size,
            "effective_batch_size": args.batch_size * world_size * args.grad_accum_steps,
        },
        path,
    )

def load_checkpoint(path, model, optimizer, scaler, device):
    path = Path(path)

    if not path.exists():
        return 0

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    move_optimizer_state_to_device(optimizer, device)

    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    start_epoch = checkpoint["epoch"] + 1

    return start_epoch

def write_log(log_path, epoch, train_loss, val_loss, top1, top5, is_main_process):
    if not is_main_process:
        return

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"{epoch},{train_loss},{val_loss},{top1},{top5}\n")

@torch.no_grad()
def evaluate(model, val_loader, criterion, device, amp, world_size):
    model.eval()

    total_loss = 0.0 #total指当前 GPU
    total_top1 = 0.0
    total_top5 = 0.0
    total_samples = 0

    for images, labels in val_loader:
        images = images.to(device, non_blocking=True) #异步传输到 GPU
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=amp):
            logits = model(images)
            loss = criterion(logits, labels)

        batch_size = images.shape[0]

        top1_correct, top5_correct = accuracy(logits, labels, topk=(1, 5))

        total_loss += loss.item() * batch_size
        total_top1 += top1_correct.item()
        total_top5 += top5_correct.item()
        total_samples += batch_size

    stats = torch.tensor(
        [total_loss, total_top1, total_top5, total_samples],
        device=device,
        dtype=torch.float64,
    )

    dist.all_reduce(stats, op=dist.ReduceOp.SUM) #把所有卡的统计量加起来

    val_loss = stats[0].item() / stats[3].item()
    top1 = stats[1].item() / stats[3].item() * 100.0
    top5 = stats[2].item() / stats[3].item() * 100.0

    model.train()

    return val_loss, top1, top5

def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, amp, grad_accum_steps,):
    model.train()

    optimizer.zero_grad()

    total_loss = 0.0
    total_samples = 0

    for step, (images, labels) in enumerate(train_loader): #step是这个epoch的第几个batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=amp):
            logits = model(images)
            loss = criterion(logits, labels)
            loss_for_backward = loss / grad_accum_steps #累计了4个micro batch再更新参数

        if amp:
            scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()

        should_step = (step + 1) % grad_accum_steps == 0

        if should_step:
            if amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()

        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    stats = torch.tensor(
        [total_loss, total_samples],
        device=device,
        dtype=torch.float64,
    )

    dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    train_loss = stats[0].item() / stats[1].item()

    return train_loss

def main():
    args = parse_args()

    device, rank, local_rank, world_size = setup_ddp()
    is_main_process = rank == 0

    output_path = Path(args.output_path)
    log_path = Path(args.log_path)

    train_loader, val_loader, train_sampler, val_sampler = build_dataloaders(
        args,
        rank,
        world_size,
    )

    model, criterion, optimizer, scaler = build_model_and_train_state(
        args,
        device,
    )

    start_epoch = load_checkpoint(
        output_path,
        model,
        optimizer,
        scaler,
        device,
    )

    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
    )

    if is_main_process:
        print("device:", device)
        print("world_size:", world_size)
        print("train samples:", len(train_loader.dataset))
        print("val samples:", len(val_loader.dataset))
        print("effective batch size:", args.batch_size * world_size * args.grad_accum_steps)
        print("start epoch:", start_epoch)

    if is_main_process and start_epoch == 0:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            "epoch,train_loss,val_loss,top1,top5\n",
            encoding="utf-8",
        )

    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)

        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp=args.amp,
            grad_accum_steps=args.grad_accum_steps,
        )

        should_eval = (epoch + 1) % args.eval_interval == 0
        if should_eval:
            val_loss, top1, top5 = evaluate(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                device=device,
                amp=args.amp,
                world_size=world_size,
            )
        else:
            val_loss = float("nan")
            top1 = float("nan")
            top5 = float("nan")

        if is_main_process:
            print(
                "epoch:",
                epoch,
                "train loss:",
                train_loss,
                "val loss:",
                val_loss,
                "top1:",
                top1,
                "top5:",
                top5,
            )

            write_log(
                log_path=log_path,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                top1=top1,
                top5=top5,
                is_main_process=is_main_process,
            )

        should_save = (epoch + 1) % args.save_interval == 0
        if should_save:
            save_checkpoint(
                path=output_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                args=args,
                world_size=world_size,
                is_main_process=is_main_process,
            )

            if is_main_process:
                print("saved checkpoint at epoch", epoch)

    save_checkpoint(
        path=output_path,
        epoch=args.epochs - 1,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        args=args,
        world_size=world_size,
        is_main_process=is_main_process,
    )

    if is_main_process:
        print("saved final checkpoint:", output_path)

    cleanup_ddp()

if __name__ == "__main__":
    main()