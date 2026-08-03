from pathlib import Path
import argparse

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from aft.vision_paper import PaperAFTImageClassifier
from train_imagenet_ddp import (
    build_dataloaders,
    cleanup_ddp,
    evaluate,
    get_learning_rate,
    load_checkpoint,
    save_checkpoint,
    set_learning_rate,
    setup_ddp,
    train_one_epoch,
    write_log,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", type=str, default="data/imagenet")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--num-classes", type=int, default=1000)

    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--hidden-dim", type=int, default=1536)
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--kernel-size", type=int, default=11)
    parser.add_argument("--n-heads", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use-position-embedding", action="store_true")

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad-accum-steps", type=int, default=1)

    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--cutmix-alpha", type=float, default=0.0)
    parser.add_argument("--mix-prob", type=float, default=0.0)

    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument(
        "--output-path",
        type=str,
        default="outputs/aft_imagenet_paper_conv.pt",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default="outputs/aft_imagenet_paper_conv_log.csv",
    )

    args = parser.parse_args()

    if args.d_model % args.n_heads != 0:
        parser.error("--d-model must be divisible by --n-heads")
    if args.mixup_alpha < 0.0:
        parser.error("--mixup-alpha must be >= 0")
    if args.cutmix_alpha < 0.0:
        parser.error("--cutmix-alpha must be >= 0")
    if not 0.0 <= args.mix_prob <= 1.0:
        parser.error("--mix-prob must be between 0 and 1")
    if args.grad_accum_steps < 1:
        parser.error("--grad-accum-steps must be >= 1")
    if not 0.0 <= args.label_smoothing < 1.0:
        parser.error("--label-smoothing must be in [0, 1)")

    return args


def build_model_and_train_state(args, device):
    model = PaperAFTImageClassifier(
        image_size=args.image_size,
        patch_size=args.patch_size,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        d_model=args.d_model,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        kernel_size=args.kernel_size,
        n_heads=args.n_heads,
        dropout=args.dropout,
        use_position_embedding=args.use_position_embedding,
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    return model, criterion, optimizer, scaler


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
    _ = val_sampler

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
        print("paper_aft_conv:", True)
        print("n_heads:", args.n_heads)
        print("head_dim:", args.d_model // args.n_heads)
        print("grad_clip:", args.grad_clip)
        print("label_smoothing:", args.label_smoothing)
        print("mixup_alpha:", args.mixup_alpha)
        print("cutmix_alpha:", args.cutmix_alpha)
        print("mix_prob:", args.mix_prob)
        print("start epoch:", start_epoch)

    if is_main_process and start_epoch == 0:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            "epoch,lr,train_loss,val_loss,top1,top5\n",
            encoding="utf-8",
        )

    for epoch in range(start_epoch, args.epochs):
        lr = get_learning_rate(epoch, args)
        set_learning_rate(optimizer, lr)
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
