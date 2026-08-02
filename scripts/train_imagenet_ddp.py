from pathlib import Path
import os
import argparse
from contextlib import nullcontext
import math

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

from aft.vision import AFTImageClassifier

def parse_args():
    # 所有训练超参数都放到命令行里，方便云端复现实验时保存完整命令。
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
    parser.add_argument("--use-position-embedding", action="store_true")

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=90) #整个训练总共跑多少个epoch，所有GPU一起跑
    parser.add_argument("--learning-rate", type=float, default=3e-4) #最大学习率
    parser.add_argument("--min-learning-rate", type=float, default=1e-5) #最后降到的最低学习率
    parser.add_argument("--warmup-epochs", type=int, default=5)#前几个epoch现性升学习率
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=8) #用多少个 CPU 子进程在后台加载图片、做 transforms、拼 batch
    parser.add_argument("--grad-clip", type=float, default=0.0)

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad-accum-steps", type=int, default=1)

    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--cutmix-alpha", type=float, default=0.0)
    parser.add_argument("--mix-prob", type=float, default=0.0)

    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument("--output-path", type=str, default="outputs/aft_imagenet.pt")
    parser.add_argument("--log-path", type=str, default="outputs/train_imagenet_log.csv")

    return parser.parse_args()

def setup_ddp():
    # DDP = DistributedDataParallel，中文常叫“分布式数据并行”。
    # 它的基本方式是：每张 GPU 一个进程，每个进程有一份完整模型，
    # forward/backward 各算自己的 batch，backward 时自动同步梯度。
    # torchrun 会为每张 GPU 启动一个 Python 进程；
    # init_process_group 负责让这些进程建立通信。
    dist.init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"]) #当前进程使用的第几张gpu，哪台机器上的几号位
    rank = dist.get_rank() #当前进程在所有进程的编号
    world_size = dist.get_world_size() #总进程数 = 总GPU数

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    return device, rank, local_rank, world_size

def cleanup_ddp():
    # 训练结束后关闭分布式进程组，释放通信资源。
    dist.destroy_process_group()

def accuracy(logits, targets, topk=(1, 5)):
    # logits: [B, num_classes]，每一行是一张图片对所有类别的预测分数。
    # targets: [B]，每个元素是对应图片的真实类别编号。
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
    # ImageFolder 要求目录格式：
    # data/imagenet/train/class_name/*.JPEG
    # data/imagenet/val/class_name/*.JPEG
    train_dir = Path(args.data_dir) / "train"
    val_dir = Path(args.data_dir) / "val"

    # 训练 transform 带随机性：同一张图每个 epoch 可能裁出不同区域。
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.image_size), #训练集随机裁剪成 224 x 224，这是 ImageNet 常用训练增强
        transforms.RandomHorizontalFlip(), #随机左右翻转图片，提高泛化
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(), #[H, W, C] -> [C, H, W]，并且像素值从 0~255 变成 0~1
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ), #标准均值和方差归一化、
        transforms.RandomErasing(
            p=0.25,
            scale=(0.02, 0.33),
            ratio=(0.3, 3.3)
        )
    ])

    # 验证 transform 不带随机性：保证每次评估结果可比较。
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

    # 验证集也用 DistributedSampler，这样所有 GPU 一起评估不同子集；
    # 后面 evaluate() 会用 all_reduce 汇总所有 GPU 的统计量。
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

    # 每次迭代 val_loader 会返回：
    # images: [batch_size, 3, image_size, image_size]
    # labels: [batch_size]
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
    # AFTImageClassifier 输入图片 [B, 3, H, W]，输出 logits [B, num_classes]。
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
        use_position_embedding=args.use_position_embedding,
    )
    model = model.to(device)

    # CrossEntropyLoss 接收 logits [B, num_classes] 和整数标签 [B]。
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # AdamW 是带 decoupled weight decay 的 Adam，ImageNet/Transformer 训练常用。
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # AMP = Automatic Mixed Precision，中文常叫“自动混合精度”。
    # autocast 让部分计算用 float16/bfloat16 加速，GradScaler 负责降低 float16 梯度下溢风险。
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp) #AMP 混合精度用的梯度缩放器

    return model, criterion, optimizer, scaler

def get_learning_rate(epoch, args): #epoch是当前第几轮
    # Warmup：训练开头几轮把学习率从小逐渐升到最大学习率，避免刚开始更新过猛。
    if epoch < args.warmup_epochs:
        return args.learning_rate * float(epoch + 1) / float(args.warmup_epochs)

    # Cosine decay：warmup 结束后，学习率按余弦曲线从最大值逐渐降到最小值。
    decay_epochs = args.epochs - args.warmup_epochs
    decay_progress = float(epoch - args.warmup_epochs) / float(max(1, decay_epochs))

    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * decay_progress))

    lr = args.min_learning_rate + (args.learning_rate - args.min_learning_rate) * cosine_decay

    return lr

def set_learning_rate(optimizer, lr):
    # PyTorch optimizer 可能有多组参数；这里统一把每组参数的学习率改成当前 epoch 的 lr。
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def move_optimizer_state_to_device(optimizer, device):
    # resume 时 optimizer state 可能先被 torch.load 放到 CPU；
    # 真正继续训练前，要把其中的动量/方差 tensor 搬到当前 rank 的 GPU。
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
    # DDP 下只允许 rank 0 写 checkpoint，避免多个进程同时写同一个文件。
    if not is_main_process:
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(model, DDP):
        # DDP(model) 包了一层，真实模型参数在 model.module 里。
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

    # 这里在 DDP 包装前加载，因此 checkpoint 里保存的是原始模型 state_dict。
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    move_optimizer_state_to_device(optimizer, device)

    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    start_epoch = checkpoint["epoch"] + 1

    return start_epoch

def write_log(log_path, epoch, lr, train_loss, val_loss, top1, top5, is_main_process):
    # CSV 日志同样只让 rank 0 写，避免重复行。
    if not is_main_process:
        return

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"{epoch},{lr},{train_loss},{val_loss},{top1},{top5}\n")

def sample_beta(alpha, device):
    alpha_tensor = torch.tensor(alpha, device=device)
    beta = torch.distributions.Beta(alpha_tensor, alpha_tensor)
    return float(beta.sample().item())

def rand_bbox(images, lam):
    _, _, height, width = images.shape

    cut_ratio = math.sqrt(1.0 - lam)
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)

    cx = int(torch.randint(width, (1,), device=images.device).item())
    cy = int(torch.randint(height, (1,), device=images.device).item())

    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, width)
    y2 = min(cy + cut_h // 2, height)

    return x1, y1, x2, y2

def apply_mixup_cutmix(images, labels, mixup_alpha, cutmix_alpha, mix_prob):
    if mix_prob <= 0.0:
        return images, labels, labels, 1.0

    if mixup_alpha <= 0.0 and cutmix_alpha <= 0.0:
        return images, labels, labels, 1.0

    if float(torch.rand((), device=images.device).item()) > mix_prob:
        return images, labels, labels, 1.0

    batch_size = images.shape[0]
    perm = torch.randperm(batch_size, device=images.device)

    use_cutmix = cutmix_alpha > 0.0
    if mixup_alpha > 0.0 and cutmix_alpha > 0.0:
        use_cutmix = bool(torch.rand((), device=images.device).item() < 0.5)

    if use_cutmix:
        lam = sample_beta(cutmix_alpha, images.device)
        x1, y1, x2, y2 = rand_bbox(images, lam)

        mixed_images = images.clone()
        mixed_images[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]

        area = (x2 - x1) * (y2 - y1)
        total_area = images.shape[2] * images.shape[3]
        lam = 1.0 - area / total_area
    else:
        lam = sample_beta(mixup_alpha, images.device)
        mixed_images = lam * images + (1.0 - lam) * images[perm]

    labels_a = labels
    labels_b = labels[perm]

    return mixed_images, labels_a, labels_b, lam

@torch.no_grad()
def evaluate(model, val_loader, criterion, device, amp, world_size):
    # 验证阶段不更新参数，所以关闭梯度记录并切到 eval 模式。
    model.eval()

    total_loss = 0.0 #total指当前 GPU
    total_top1 = 0.0
    total_top5 = 0.0
    total_samples = 0

    for images, labels in val_loader:
        # images: [B, 3, H, W]，labels: [B]。
        # non_blocking=True 配合 pin_memory=True，可以让 CPU->GPU 拷贝更高效。
        images = images.to(device, non_blocking=True) #异步传输到 GPU
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=amp):
            # logits: [B, num_classes]。
            logits = model(images)
            loss = criterion(logits, labels)

        batch_size = images.shape[0]

        top1_correct, top5_correct = accuracy(logits, labels, topk=(1, 5))

        # loss 是当前 batch 平均值；乘以 batch_size 后变成当前 batch 的 loss 总和。
        # 这样最后才能按样本数求全局平均。
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

    # stats[3] 是所有 GPU 的验证样本总数。
    val_loss = stats[0].item() / stats[3].item()
    top1 = stats[1].item() / stats[3].item() * 100.0
    top5 = stats[2].item() / stats[3].item() * 100.0

    model.train()

    return val_loss, top1, top5

def train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        scaler,
        device,
        amp,
        grad_accum_steps,
        grad_clip,
        mixup_alpha,
        cutmix_alpha,
        mix_prob,
):
    # 训练模式会启用 Dropout 等训练期行为。
    model.train()

    optimizer.zero_grad()

    total_loss = 0.0
    total_samples = 0
    num_batches = len(train_loader)

    for step, (images, labels) in enumerate(train_loader): #step是这个epoch的第几个batch
        # train_loader 每次吐出一个 batch：
        # images: [B, 3, H, W]，labels: [B]。
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        should_sync = (step + 1) % grad_accum_steps == 0
        should_sync = should_sync or (step + 1 == num_batches)
        # 当前梯度累积组里一共有多少个 micro batch。
        # 最后一组可能不足 grad_accum_steps，所以不能永远除以 grad_accum_steps。
        group_start = (step // grad_accum_steps) * grad_accum_steps
        current_accum_steps = min(grad_accum_steps, num_batches - group_start)
        is_ddp = isinstance(model, DDP)

        if is_ddp and not should_sync:
            # no_sync 是 DDP 的通信优化：
            # 梯度累积的前几个 micro batch 只在本 GPU 上累积梯度，暂不同步到其他 GPU。
            sync_context = model.no_sync()
        else:
            # nullcontext 是“什么都不做”的上下文；当不需要 no_sync 时用它统一代码结构。
            sync_context = nullcontext()

        with sync_context:
            mixed_images, labels_a, labels_b, lam = apply_mixup_cutmix(
                images=images,
                labels=labels,
                mixup_alpha=mixup_alpha,
                cutmix_alpha=cutmix_alpha,
                mix_prob=mix_prob
            )
            with torch.amp.autocast("cuda", enabled=amp):
                # 模型输出 logits: [B, num_classes]。
                logits = model(mixed_images)
                loss = (
                    lam * criterion(logits, labels_a)
                    + (1.0 - lam) * criterion(logits, labels_b)
                )
                # 梯度累积时，每个 micro batch 的 loss 要除以累积步数；
                # 否则累积后的梯度会被放大。最后不足一组时，用当前组真实步数。
                loss_for_backward = loss / current_accum_steps #累计了多个micro batch再更新参数

            # backward 必须也放在 no_sync 上下文里面，否则 DDP 仍然会在 backward 时同步梯度。
            if amp:
                scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

        should_step = should_sync

        if should_step:
            # 累积够 grad_accum_steps 个 micro batch 后，才真正更新一次参数。
            if amp:
                scaler.unscale_(optimizer)
                if grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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

    # 同步所有 GPU 的 train_loss 统计量，rank 0 打印的是全局平均训练 loss。
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    train_loss = stats[0].item() / stats[1].item()

    return train_loss

def main():
    args = parse_args()

    # 每个 torchrun 子进程都会执行 main，但 rank/local_rank 不同。
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
        print("use_position_embedding:", args.use_position_embedding)
        print("grad_clip:", args.grad_clip)
        print("start epoch:", start_epoch)

    if is_main_process and start_epoch == 0:
        # 从头训练时创建新日志；resume 时继续追加，避免覆盖旧记录。
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            "epoch,lr,train_loss,val_loss,top1,top5\n",
            encoding="utf-8",
        )

    for epoch in range(start_epoch, args.epochs):
        # epoch 是当前轮数；start_epoch 只是 resume 后的起点，不能拿来计算每一轮的学习率。
        lr = get_learning_rate(epoch, args)
        set_learning_rate(optimizer, lr)
        # DDP 的 DistributedSampler 需要知道当前 epoch，
        # 才能让每个 epoch 使用不同的 shuffle 顺序。
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
            grad_clip=args.grad_clip,
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            mix_prob=args.mix_prob,
        )

        should_eval = (epoch + 1) % args.eval_interval == 0
        if should_eval:
            # evaluate 内部会 all_reduce，因此所有 rank 都必须一起进入。
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
                "lr:",
                lr,
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
                lr=lr,
                train_loss=train_loss,
                val_loss=val_loss,
                top1=top1,
                top5=top5,
                is_main_process=is_main_process,
            )

        should_save = (epoch + 1) % args.save_interval == 0
        if should_save:
            # save_checkpoint 内部只让 rank 0 真正写文件。
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
