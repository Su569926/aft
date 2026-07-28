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
    parser.add_argument("--num-workers", type=int, default=8)

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

    _, pred = logits.topk(max_k, dim=1) #pred: [B, max_k]
    pred = pred.t() #[max_k, B]

    correct = pred.eq(targets.reshape(1, -1).expand_as(pred)) #targets是真实类别[B]，correct: [max_k, B]，里面是true/false

    results = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum()
        results.append(correct_k)

    return results #返回每个 K 值对应的正确样本数

